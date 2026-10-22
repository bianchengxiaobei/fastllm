# -*- coding: utf-8 -*-
"""快速判断 CPU MoE 是否被内存带宽卡住.

思路: 每 token 的专家字节数是模型结构决定的常量(config 可算),
把日志里实测的 CPU MoE 耗时除一下就是"有效带宽", 再和机器实测读带宽比.

用法:
    python tools/moe_bw_check.py --model E:\\models\\xxx
    python tools/moe_bw_check.py --model E:\\models\\xxx --log run.log
    python tools/moe_bw_check.py --model E:\\models\\xxx --log run.log --membw 30.2
    # 部分层专家在显存(对应 --moe_device_layers N)
    python tools/moe_bw_check.py --model ... --log run.log --moe-gpu-layers 16
    # 量化格式认不出来时手动指定字节数
    python tools/moe_bw_check.py --model ... --bytes-per-param 0.5

覆盖范围: 字段名按常见写法兼容(hidden_size/n_embd, moe_intermediate_size/
intermediate_size, num_experts/n_routed_experts/num_local_experts,
shared_expert_intermediate_size 或 n_shared_experts 倍数, first_k_dense_replace);
量化兼容 compressed-tensors / fp8 / GPTQ-AWQ(顶层 bits) / 未量化 dtype.
GGUF 这类 config 里没有 bit 宽度的必须用 --bytes-per-param 指定.

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

MIB = 1048576.0
GIB = 1073741824.0


def bandwidth_gbs(nbytes, cost_ms):
    """字节 / 毫秒 -> GB/s (十进制, 与 membw 等工具一致)."""
    if not cost_ms or cost_ms <= 0.0:
        return 0.0
    return nbytes / 1e6 / cost_ms

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


def quant_bytes_per_param(quant, dtype=""):
    """返回 (每参数字节, 每个参数的 group 元数据字节, 来源说明).

    认不出量化格式时首项为 None, 由调用方决定兜底值.
    """
    method = str(quant.get("quant_method") or "").lower()

    # compressed-tensors: config_groups[..].weights
    groups = quant.get("config_groups") or {}
    for group in groups.values():
        if isinstance(group, dict) and group.get("weights"):
            weights = group["weights"]
            bits = float(weights.get("num_bits") or 4)
            group_size = float(weights.get("group_size") or 0)
            symmetric = bool(weights.get("symmetric"))
            # 对称量化只存 scale, 非对称再存一份 min
            meta = (4.0 if symmetric else 8.0) / group_size if group_size > 0 else 0.0
            return bits / 8.0, meta, "compressed-tensors %gbit" % bits

    # FP8: 没有 config_groups, 一个参数就是一字节
    if method in ("fp8", "fp8_e4m3", "fp8_e5m2", "fbgemm_fp8"):
        return 1.0, 0.0, "fp8"

    # GPTQ / AWQ / QuIP 等: bits 在顶层
    bits = quant.get("bits") or quant.get("w_bit") or quant.get("weight_bits")
    if bits:
        bits = float(bits)
        group_size = float(quant.get("group_size") or 0)
        meta = 8.0 / group_size if group_size > 0 else 0.0
        return bits / 8.0, meta, "%s %gbit" % (method or "gptq/awq", bits)

    # 完全没有量化配置: 按 config 里的 dtype 算
    if not quant:
        bits = {"float32": 32, "fp32": 32, "float16": 16, "fp16": 16,
                "bfloat16": 16, "bf16": 16}.get(str(dtype).lower())
        if bits:
            return bits / 8.0, 0.0, "未量化 %s" % dtype

    # 认不出来: 交给调用方决定兜底值
    return None, 0.0, "unknown(%s)" % (method or "none")


def pick(source, names, default=None):
    for name in names:
        value = source.get(name)
        if value not in (None, "", []):
            return value
    return default


def moe_geometry(text, log_info):
    # 各家字段名不统一, 按常见写法依次尝试
    hidden = pick(text, ("hidden_size", "n_embd", "d_model"))
    inter = pick(text, ("moe_intermediate_size", "intermediate_size",
                        "ffn_hidden_size"))
    layers = pick(text, ("num_hidden_layers", "num_layers", "n_layer"))
    topk = pick(text, ("num_experts_per_tok", "num_experts_per_token",
                       "moe_top_k", "top_k_experts"))
    experts = pick(text, ("num_experts", "n_routed_experts",
                          "num_local_experts", "moe_num_experts"))
    dense_layers = pick(text, ("first_k_dense_replace", "num_dense_layers"), 0) or 0

    # 共享专家有两种写法: 直接给中间维, 或者给"几个专家的倍数"
    shared = pick(text, ("shared_expert_intermediate_size",
                         "shared_expert_size"), 0) or 0
    if not shared and inter:
        shared_count = pick(text, ("n_shared_experts",),
                            text.get("shared_expert_intermediate_size") and 1 or 0) or 0
        shared = int(shared_count) * int(inter)

    if log_info:
        topk = log_info.get("topk", topk)
    if not hidden or not inter or not layers or not topk:
        sys.exit("config 里缺少 MoE 结构字段(hidden=%s, moe_inter=%s, layers=%s, "
                 "topk=%s), 这个模型可能不是 MoE"
                 % (hidden, inter, layers, topk))
    return {
        "hidden": hidden,
        "inter": inter,
        "layers": layers,
        "topk": topk,
        "experts": experts,
        "shared": shared,
        "dense_layers": int(dense_layers),
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
    """用 numpy 多线程读求和粗测机器读带宽, 返回 (GB/s, 线程数).

    注意必须真开线程: numpy 求和在 reduce 里放 GIL, 串行切片测不出多通道带宽.
    """
    try:
        import numpy as np
    except ImportError:
        return None, 0
    import threading

    count = size_mb * 1024 * 1024 // 8
    buffer = np.ones(count, dtype=np.float64)
    total_bytes = count * 8
    limit = os.cpu_count() or 1
    best, best_threads = 0.0, 0
    for threads in (1, 2, 4, 8, 16, 32):
        if threads > limit:
            break
        chunk = count // threads
        pieces = [buffer[i * chunk:(i + 1) * chunk] for i in range(threads)]
        sink = [0.0] * threads

        def work(index):
            sink[index] = float(pieces[index].sum())

        for _ in range(2):  # 预热, 让页表与缓存状态稳定
            warm = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
            for worker in warm:
                worker.start()
            for worker in warm:
                worker.join()

        cost = 0.0
        for _ in range(rounds):
            batch = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
            begin = time.perf_counter()
            for worker in batch:
                worker.start()
            for worker in batch:
                worker.join()
            cost = time.perf_counter() - begin
        if cost <= 0.0:
            continue
        rate = total_bytes / 1e9 / cost
        if rate > best:
            best, best_threads = rate, threads
    return (best, best_threads) if best > 0.0 else (None, 0)


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
    parser.add_argument("--bytes-per-param", type=float, default=0.0,
                        help="量化格式认不出来时手动指定, 例如 int4=0.5 int8=1.0 fp8=1.0")
    parser.add_argument("--moe-gpu-layers", type=int, default=0,
                        help="有多少层专家在显存里(对应 --moe_device_layers), 这些层不计入 CPU 字节")
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
    bytes_per_param, meta_per_param, quant_tag = quant_bytes_per_param(
        quant, text.get("dtype") or cfg.get("torch_dtype") or "")
    if args.bytes_per_param > 0.0:
        bytes_per_param = args.bytes_per_param
        quant_tag = "手动指定 %g" % bytes_per_param
    if bytes_per_param is None:
        print("!! 认不出量化格式(%s). 用 --bytes-per-param 指定,"
              " 例如 int4=0.5 / int8=1.0 / fp8=1.0 / bf16=2.0" % quant_tag)
        sys.exit(2)
    geo = moe_geometry(text, log_info)

    gate_up_bytes, down_bytes = layer_bytes(geo, bytes_per_param, meta_per_param)
    gpu_layers = max(0, args.moe_gpu_layers)
    cpu_layers = geo["layers"] - geo["dense_layers"] - gpu_layers
    if cpu_layers < 1:
        gpu_layers = 0
        cpu_layers = max(1, geo["layers"] - geo["dense_layers"])
        print("!! --moe-gpu-layers 过大, 按全部 MoE 层都留在 CPU 计算")
    total_bytes = (gate_up_bytes + down_bytes) * cpu_layers

    print("== 模型 ==")
    print("  layers=%d  experts=%s  topk=%d  hidden=%d  moe_inter=%d  共享专家中间维=%s"
          % (geo["layers"], geo["experts"], geo["topk"], geo["hidden"],
             geo["inter"], geo["shared"] or "无"))
    print("  量化: %.2f 字节/参数 + group 元数据 %.1f%%  (%s)"
          % (bytes_per_param, meta_per_param * 100.0, quant_tag))
    if geo["dense_layers"] or gpu_layers:
        print("  计入 CPU 的 MoE 层数: %d (总 %d, 稠密层 %d, 专家在显存 %d)"
              % (cpu_layers, geo["layers"], geo["dense_layers"], gpu_layers))
    print("  每层每 token 专家字节: gate_up=%.2fMiB  down=%.2fMiB"
          % (gate_up_bytes / MIB, down_bytes / MIB))
    print("  每 token 专家字节合计: %.1fMiB  (%.2fGiB, %.1fMB)"
          % (total_bytes / MIB, total_bytes / GIB, total_bytes / 1e6))

    ceiling = args.membw
    probe_threads = 0
    if ceiling <= 0.0:
        probed, probe_threads = probe_membw(args.size)
        ceiling = probed or 0.0
    if ceiling > 0.0:
        source = "  (手动指定)" if args.membw > 0.0 else (
            "  (现测, %d 线程)" % probe_threads if probe_threads else "  (现测)")
        print("  机器读带宽: %.1fGB/s%s" % (ceiling, source))
        if args.membw <= 0.0:
            print("  (现测为当前机器的粗测值 +-15%, 服务在别的机器上请用 --membw 传入那边实测值)")
        per_token_ms = total_bytes / 1e6 / ceiling
        print("  带宽下限: 每层 %.3fms, 每 token %.1fms"
              % (per_token_ms / cpu_layers, per_token_ms))
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
        print("\n== 实测(profile-cpu-moe, %d 层采样, 每层中位数) ==" % len(rows))
        print("  total=%.3fms  gate=%.3fms  down=%.3fms"
              % (gemm_ms, gate_ms or 0.0, down_ms or 0.0))
        if gate_ms:
            print("  gate 有效带宽: %.1fGB/s" % bandwidth_gbs(gate_up_bytes, gate_ms))
        if down_ms:
            print("  down 有效带宽: %.1fGB/s" % bandwidth_gbs(down_bytes, down_ms))
        # gate/down 的字节比与实际耗时比是否一致, 一致说明纯按字节计费
        if gate_ms and down_ms:
            byte_ratio = gate_up_bytes / down_bytes
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
            gemm_total = gemm_ms * cpu_layers
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
    basis = gemm_ms * cpu_layers if gemm_ms else merge_ms
    effective = bandwidth_gbs(total_bytes, basis)
    print("\n== 结论 ==")
    print("  MoE 有效带宽: %.1fGB/s  (按 %.1fms/token 计)"
          % (effective, basis))
    if not ceiling:
        return
    ratio = effective / ceiling
    if ratio > 1.15:
        print("  占机器读带宽: %.0f%%" % (ratio * 100.0))
        print("  !! 数据矛盾, 不给结论. 有效带宽超过机器上限, 通常是这两台不是同一台机器:")
        print("     --log 来自服务机, 而带宽是在本机现测的(本机 %.1fGB/s)."
              % ceiling)
        print("     请在服务机上直接跑本脚本, 或用 --membw 传入服务机的实测值"
              " (本脚本现测值只代表当前这台机器).")
        return
    if ratio > 1.0:
        print("  占机器读带宽: 约 100%% (超出 %.0f%%, 在测量误差内)"
              % ((ratio - 1.0) * 100.0))
        ratio = 1.0
    title, advice = verdict(ratio)
    print("  占机器读带宽: %.0f%%" % (ratio * 100.0))
    print("  %s" % title)
    print("  建议: %s" % advice)


if __name__ == "__main__":
    main()
