# -*- coding: utf-8 -*-
"""快速判断 CPU MoE 是否被内存带宽卡住.

思路: 每 token 的专家字节数是模型结构决定的常量(config 可算),
把日志里实测的 CPU MoE 耗时除一下就是"有效带宽", 再和机器实测读带宽比.

用法:
    python tools/moe_bw_check.py --model E:\\models\\xxx
    python tools/moe_bw_check.py --model E:\\models\\xxx --log run.log
    python tools/moe_bw_check.py --model E:\\models\\xxx --log run.log --membw 30.2

--log 里需要 FASTLLM_PROFILE=1 FASTLLM_PROFILE_DETAIL=1 打出的
[fastllm-profile-cpu-moe] 行, 退一步也可以用 [fastllm-host-moe-split] 的 merge=.

判读:
    >= 75%   带宽基本打满, CPU 侧没有软件空间, 只能减字节或把专家搬上卡
    50~75%   混合受限, 先查线程数与解包算力
    < 50%    明显有软件问题(并行度/barrier/数据类型)
"""

import argparse
import json
import os
import re
import statistics
import sys
import time

PROFILE_MOE = "[fastllm-profile-cpu-moe]"
PROFILE_SPLIT = "[fastllm-host-moe-split]"

ROUNDS = 15


def read_config(model_dir):
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        sys.exit("找不到 %s" % path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    text = cfg.get("text_config") or cfg
    quant = (cfg.get("quantization_config")
             or text.get("quantization_config")
             or {})
    return cfg, text, quant


def quant_bytes_per_param(quant):
    """返回 (每参数字节, 每个参数的 group 元数据字节)."""
    weights = None
    groups = quant.get("config_groups") or {}
    for group in groups.values():
        if isinstance(group, dict) and group.get("weights"):
            weights = group["weights"]
            break
    if not weights:
        # 没有量化信息时按 bf16 估
        return 2.0, 0.0
    bits = float(weights.get("num_bits") or 4)
    group_size = float(weights.get("group_size") or 0)
    symmetric = bool(weights.get("symmetric"))
    meta = 0.0
    if group_size > 0:
        # 对称量化只存 scale, 非对称再存一份 min
        meta = (4.0 if symmetric else 8.0) / group_size
    return bits / 8.0, meta


def moe_geometry(text, log_info):
    hidden = text.get("hidden_size")
    inter = text.get("moe_intermediate_size") or text.get("intermediate_size")
    layers = text.get("num_hidden_layers")
    topk = text.get("num_experts_per_tok")
    experts = text.get("num_experts")
    shared = text.get("shared_expert_intermediate_size") or 0
    if log_info:
        topk = log_info.get("topk", topk)
    if not hidden or not inter or not layers or not topk:
        sys.exit("config 里缺少 MoE 结构字段(hidden_size/moe_intermediate_size/"
                 "num_hidden_layers/num_experts_per_tok), 这个模型可能不是 MoE")
    return {
        "hidden": hidden,
        "inter": inter,
        "layers": layers,
        "topk": topk,
        "experts": experts,
        "shared": shared,
    }


def layer_bytes(geo, bytes_per_param, meta_per_param):
    """每层每 token 的专家字节数, 拆成 gate_up / down 两段."""
    scale = 1.0 + meta_per_param
    routed = geo["topk"]
    shared = 1 if geo["shared"] else 0
    gate_up = (routed + shared) * 2 * geo["inter"] * geo["hidden"]
    down = (routed + shared) * geo["inter"] * geo["hidden"]
    if shared:
        # 共享专家的中间维可能和路由专家不同, 用差值修正
        fix = (geo["shared"] - geo["inter"])
        gate_up += 2 * fix * geo["hidden"]
        down += fix * geo["hidden"]
    return gate_up * bytes_per_param * scale, down * bytes_per_param * scale


def parse_log(path):
    """返回 {'moe': [...], 'split': [...]}."""
    found = {"moe": [], "split": []}
    key_value = re.compile(r"(\w+)=([-\d.]+)")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if PROFILE_MOE in line:
                bucket = found["moe"]
            elif PROFILE_SPLIT in line:
                bucket = found["split"]
            else:
                continue
            values = {}
            for key, raw in key_value.findall(line):
                try:
                    values[key] = float(raw)
                except ValueError:
                    pass
            if values:
                bucket.append(values)
    return found


def median_of(rows, key):
    values = [row[key] for row in rows if key in row]
    return statistics.median(values) if values else None


def probe_membw(size_mb=1024, rounds=5):
    """用 numpy 读求和粗测机器读带宽, 取多线程最好值."""
    try:
        import numpy as np
    except ImportError:
        return None
    count = size_mb * 1024 * 1024 // 8
    buffer = np.ones(count, dtype=np.float64)
    best = 0.0
    for threads in (1, 2, 4, 8, 16):
        if threads > (os.cpu_count() or 1):
            continue
        chunk = count // threads
        slices = [buffer[i * chunk:(i + 1) * chunk] for i in range(threads)]
        for _ in range(2):  # 预热, 让页表和缓存状态稳定
            for piece in slices:
                piece.sum()
        cost = 0.0
        for _ in range(rounds):
            begin = time.perf_counter()
            for piece in slices:
                piece.sum()
            cost = time.perf_counter() - begin
        if cost <= 0.0:
            continue
        best = max(best, size_mb / 1024.0 / cost)
    return best or None


def verdict(ratio):
    if ratio >= 0.75:
        return ("带宽已打满 (>=75%)",
                "CPU 侧没有软件空间. 只能减字节(int4/更小 bit)、换更快内存,"
                " 或把专家搬上显存(--moe_device cuda --moe_device_layers N)")
    if ratio >= 0.50:
        return ("混合受限 (50~75%)",
                "先加线程 -t <物理核数>, 再看 profile 里 gate/down 是否仍按字节线性;"
                " 若线性则仍是内存, 否则查解包算力与线程池 barrier")
    return ("软件瓶颈 (<50%)",
            "有明显软件空间: 查 -t 是否给足、是否单线程在跑、"
            " 以及 prepare/reduce 占比是否异常")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型目录(含 config.json)")
    parser.add_argument("--log", default="", help="服务端日志(含 profile 行)")
    parser.add_argument("--membw", type=float, default=0.0,
                        help="机器实测读带宽 GB/s; 不填则用 numpy 现测")
    parser.add_argument("--size", type=int, default=1024,
                        help="现测带宽时的缓冲区大小 MB")
    args = parser.parse_args()

    log_info = {}
    found = {"moe": [], "split": []}
    if args.log:
        if not os.path.isfile(args.log):
            sys.exit("找不到日志 %s" % args.log)
        found = parse_log(args.log)
        if found["moe"]:
            # profile 里的 experts= 是本次实际参与计算的专家数(含共享), 不是总专家数
            log_info = dict((k, int(v)) for k, v in found["moe"][0].items()
                            if k == "topk")

    cfg, text, quant = read_config(args.model)
    bytes_per_param, meta_per_param = quant_bytes_per_param(quant)
    geo = moe_geometry(text, log_info)

    gate_up_bytes, down_bytes = layer_bytes(geo, bytes_per_param, meta_per_param)
    total_bytes = (gate_up_bytes + down_bytes) * geo["layers"]

    print("== 模型 ==")
    print("  layers=%d  experts=%s  topk=%d  hidden=%d  moe_inter=%d  共享专家中间维=%s"
          % (geo["layers"], geo["experts"], geo["topk"], geo["hidden"],
             geo["inter"], geo["shared"] or "无"))
    print("  量化: %.2f 字节/参数 + group 元数据 %.1f%%"
          % (bytes_per_param, meta_per_param * 100.0))
    print("  每层每 token 专家字节: gate_up=%.2fMB  down=%.2fMB"
          % (gate_up_bytes / 1048576.0, down_bytes / 1048576.0))
    print("  每 token 专家字节合计: %.1fMB  (%.2fGB)"
          % (total_bytes / 1048576.0, total_bytes / 1073741824.0))

    ceiling = args.membw
    if ceiling <= 0.0:
        ceiling = probe_membw(args.size) or 0.0
    if ceiling > 0.0:
        print("  机器读带宽: %.1fGB/s%s"
              % (ceiling, "  (现测)" if args.membw <= 0.0 else "  (手动指定)"))
        per_token_ms = total_bytes / 1048576.0 / ceiling
        print("  带宽下限: 每层 %.3fms, 每 token %.1fms"
              % (per_token_ms / geo["layers"], per_token_ms))
    else:
        print("  机器读带宽: 未知(没装 numpy?), 用 --membw 指定, 例如 30.2")
        per_token_ms = None

    if not found["moe"] and not found["split"]:
        print("\n没在日志里找到 profile 行. 加环境变量重跑:")
        print("  set FASTLLM_PROFILE=1")
        print("  set FASTLLM_PROFILE_DETAIL=1")
        return

    merge_ms = None
    gemm_ms = None
    rows = found["moe"]
    if rows:
        gemm_ms = median_of(rows, "total")
        gate_ms = median_of(rows, "gate")
        down_ms = median_of(rows, "down")
        gate_bytes = gate_up_bytes / 1048576.0
        down_bytes_mb = down_bytes / 1048576.0
        print("\n== 实测(profile-cpu-moe, %d 层采样, 每层中位数) ==" % len(rows))
        print("  total=%.3fms  gate=%.3fms  down=%.3fms"
              % (gemm_ms, gate_ms or 0.0, down_ms or 0.0))
        if gate_ms:
            print("  gate 有效带宽: %.1fGB/s" % (gate_bytes / gate_ms))
        if down_ms:
            print("  down 有效带宽: %.1fGB/s" % (down_bytes_mb / down_ms))
        # gate/down 的字节比与实际耗时比是否一致, 一致说明纯按字节计费
        if gate_ms and down_ms:
            byte_ratio = gate_bytes / down_bytes_mb
            time_ratio = gate_ms / down_ms
            print("  字节比 gate:down = %.2f, 耗时比 = %.2f  -> %s"
                  % (byte_ratio, time_ratio,
                     "纯带宽" if abs(byte_ratio - time_ratio) / byte_ratio < 0.15
                     else "有非带宽成分"))

    if found["split"]:
        split_merge = median_of(found["split"], "merge")
        print("\n== 实测(host-moe-split, %d token 采样) ==" % len(found["split"]))
        print("  merge=%.3fms/token  (含 MergeMOEBlock 外框)" % (split_merge or 0.0))
        if gemm_ms:
            gemm_total = gemm_ms * geo["layers"]
            overhead = split_merge - gemm_total
            print("  GEMM 本体 %.1fms, 外框开销 %.1fms (占 merge %.0f%%)"
                  % (gemm_total, overhead,
                     overhead / split_merge * 100.0 if split_merge else 0.0))
            if split_merge and overhead / split_merge > 0.10:
                print("  外框占比偏高, 值得查 executor 分发/线程池唤醒"
                      " (这部分不受带宽限制)")
        merge_ms = split_merge

    if not (gemm_ms or merge_ms):
        return

    # 判带宽用 GEMM 本体, 外框开销不属于带宽
    basis = gemm_ms * geo["layers"] if gemm_ms else merge_ms
    effective = total_bytes / 1048576.0 / basis
    print("\n== 结论 ==")
    print("  MoE 有效带宽: %.1fGB/s  (按 %.1fms/token 计)"
          % (effective, basis))
    if not ceiling:
        return
    ratio = effective / ceiling
    title, advice = verdict(ratio)
    print("  占机器读带宽: %.0f%%" % (ratio * 100.0))
    print("  %s" % title)
    print("  建议: %s" % advice)


if __name__ == "__main__":
    main()
