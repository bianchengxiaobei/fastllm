#!/usr/bin/env python3
"""DeepSeek-V4.1 前缀缓存落盘回归测试（跨进程免重 prefill）。

用 test/basic/deepseek_v41_fixture.npz 的微型 V4.1 模型验证三件事：

  1. record 阶段：--cache_history 请求结束后，<model_dir>/fastllm_prefix_cache/
     出现非空 record_*.bin；
  2. replay 阶段：新进程重载模型后，日志出现 "disk load: ... adopted=" 且本次
     请求命中磁盘记录（"prefix-cache] hit len="），只 prefill 剩余尾巴；
  3. 一致性：同一上下文，"磁盘记录恢复" 与 "FASTLLM_DSV41_PREFIX_CACHE_DIR=off
     全量重算" 的贪心 token 序列完全一致。

提示词设计受 CanTruncateHistory 约束：滑窗(window_size=8)只允许把记录截断到
total-1；ratio-2 压缩层还要求截断点为偶数。所以 replay 的提示词 = 上一轮的
prompt + 生成结果（必要时补 1 个 token 把截断点凑成偶数），对应真实多轮对话
"下一轮带上完整历史"的用法。

每个阶段在独立子进程里跑（C++ 侧的懒加载/指纹是进程内状态）。

用法：
    PYTHONPATH=<fastllm>/build-vs-gpu/tools python test/basic/test_deepseek_v41_prefix_disk.py

退出码：0 = 通过（fixture 缺失则跳过），1 = 失败。
"""

import argparse
import ctypes
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "deepseek_v41_fixture.npz")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", default=FIXTURE)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--keep", action="store_true", help="保留解包出来的模型目录")
    p.add_argument("--phase", default="", choices=("", "record", "replay", "baseline"),
                   help="内部使用：子进程里跑单个阶段")
    p.add_argument("--model-dir", default="", help="内部使用：解包好的模型目录")
    p.add_argument("--tail-file", default="", help="内部使用：record 阶段产出的生成 token")
    p.add_argument("--out", default="", help="内部使用：把本阶段生成 token 写成 json")
    return p.parse_args()


def load_meta(fixture):
    data = np.load(fixture)
    return json.loads(bytes(data["meta"]).decode("utf-8"))


def unpack(fixture, dst):
    data = np.load(fixture)
    meta = json.loads(bytes(data["meta"]).decode("utf-8"))
    for name, key in meta["files"].items():
        with open(os.path.join(dst, name), "wb") as f:
            f.write(bytes(data[key]))
    # fixture 的 image_token_id=127 落在 128 词表内，随机 prompt 撞上它就会被
    # V41HasImageToken 当成"带图像占位却无图像数据"而静默拒绝前缀恢复；迷你
    # 模型没有视觉塔（vision_n_layers=0），这里把顶层与 text_config 里的占位
    # id 都置为 -1 关掉该守卫（加载器以顶层为准）。
    config_path = os.path.join(dst, "config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    config["image_token_id"] = -1
    if isinstance(config.get("text_config"), dict):
        config["text_config"]["image_token_id"] = -1
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, separators=(",", ":"))
    return meta


def record_cache_dir(model_dir):
    return os.path.join(model_dir, "fastllm_prefix_cache")


def wait_record_file(model_dir, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        files = [f for f in glob.glob(os.path.join(record_cache_dir(model_dir), "record_*.bin"))
                 if os.path.getsize(f) > 0]
        if files:
            return files
        time.sleep(0.2)
    return []


def run_response(args, meta, prompt, disk_off):
    os.environ.setdefault("FASTLLM_SKIP_WARMUP", "1")
    os.environ["FASTLLM_DSV41_PREFIX_CACHE_DEBUG"] = "1"
    # 上游 V41PrefixCacheDisabled 用的是"默认开启"的旗子语义，不设时前缀缓存
    # 整体关闭；显式设 0 才启用（内存版与落盘版共用这道门）。
    os.environ["FASTLLM_DSV41_DISABLE_PREFIX_CACHE"] = "0"
    if disk_off:
        os.environ["FASTLLM_DSV41_PREFIX_CACHE_DIR"] = "off"
    else:
        os.environ.pop("FASTLLM_DSV41_PREFIX_CACHE_DIR", None)
    for k, v in meta["fastllm"].get("env", {}).items():
        os.environ.setdefault(k, v)

    from ftllm import llm
    from ftllm.util import make_normal_llm_model, make_normal_parser

    parser = make_normal_parser("v41 prefix disk")
    argv = ["--path", args.model_dir, "--dtype", meta["fastllm"]["dtype"],
            "--device", meta["fastllm"]["device"],
            "--moe_device", meta["fastllm"]["moe_device"],
            "-t", str(args.threads)]
    if meta["fastllm"]["device"] == "cpu":
        argv += ["--cuda_shared_expert", "false"]
    fargs = parser.parse_args(argv)
    if fargs.max_batch <= 0:
        fargs.max_batch = 1
    if fargs.tokens <= 0:
        fargs.tokens = 4096
    model = make_normal_llm_model(fargs)
    model.set_save_history(True)

    steps = int(meta["decode_steps"]) + 1
    array = (ctypes.c_int * len(prompt))(*prompt)
    handle = llm.fastllm_lib.launch_response_llm_model(
        model.model, len(prompt), array, ctypes.c_int(steps), ctypes.c_int(0),
        ctypes.c_bool(False), ctypes.c_float(1.0), ctypes.c_int(1), ctypes.c_float(1.0),
        ctypes.c_float(1.0), ctypes.c_bool(True), ctypes.c_int(0), None)
    tokens = []
    while True:
        token_id = llm.fastllm_lib.fetch_response_llm_model(model.model, handle)
        if token_id < 0:
            break
        tokens.append(int(token_id))
    return tokens


def run_phase(args):
    meta = load_meta(args.fixture)
    if args.phase == "record":
        prompt = list(meta["prompt"])
    else:
        with open(args.tail_file, encoding="utf-8") as f:
            tail = json.load(f)
        prompt = list(meta["prompt"]) + [int(t) for t in tail]
        # 截断点必须是偶数(ratio-2 压缩层)：total-1 为奇数时补 1 个 token，
        # 让 hit 落在 total（偶数）上，剩余 prefill 恰好 1 个 token。
        if (len(prompt) - 1) % 2 == 1:
            prompt.append(int(tail[-1]))
    tokens = run_response(args, meta, prompt, disk_off=(args.phase == "baseline"))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(tokens, f)
    if args.phase == "record":
        files = wait_record_file(args.model_dir)
        print("record files: %d" % len(files))
        for path in files:
            print("  %s (%d bytes)" % (os.path.basename(path), os.path.getsize(path)))
    sys.stdout.flush()
    return 0


def run_child(python, script, args, phase, model_dir, tail_file, out_file):
    cmd = [python, script, "--phase", phase, "--fixture", args.fixture,
           "--model-dir", model_dir, "--threads", str(args.threads), "--out", out_file]
    if tail_file:
        cmd += ["--tail-file", tail_file]
    env = dict(os.environ)
    tools = os.environ.get("FTLLM_TOOLS")
    if tools:
        env["PYTHONPATH"] = tools
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)


def main():
    args = parse_args()
    if not os.path.exists(args.fixture):
        print("SKIP: fixture %s not found." % args.fixture)
        return 0
    if args.phase:
        return run_phase(args)

    work = tempfile.mkdtemp(prefix="v41-prefix-disk-")
    failures = []
    try:
        unpack(args.fixture, work)
        script = os.path.abspath(__file__)
        python = sys.executable
        tail_file = os.path.join(work, "record.tokens.json")
        replay_out = os.path.join(work, "replay.tokens.json")
        baseline_out = os.path.join(work, "baseline.tokens.json")

        print("== record phase ==")
        r = run_child(python, script, args, "record", work, "", tail_file)
        print(r.stdout[-1500:])
        if r.returncode != 0:
            failures.append("record phase exit=%d\n%s" % (r.returncode, r.stderr[-1500:]))
        elif not os.path.exists(tail_file):
            failures.append("record phase wrote no tokens")
        else:
            with open(tail_file, encoding="utf-8") as f:
                if not json.load(f):
                    failures.append("record phase generated no tokens")
        files = [f for f in glob.glob(os.path.join(record_cache_dir(work), "record_*.bin"))
                 if os.path.getsize(f) > 0]
        if not files:
            failures.append("no record_*.bin under %s" % record_cache_dir(work))

        print("== replay phase ==")
        r2 = run_child(python, script, args, "replay", work, tail_file, replay_out)
        print(r2.stdout[-1500:])
        if r2.returncode != 0:
            failures.append("replay phase exit=%d\n%s" % (r2.returncode, r2.stderr[-1500:]))
        if "disk load:" not in r2.stdout:
            failures.append("replay phase has no 'disk load:' log")
        if "prefix-cache] hit len=" not in r2.stdout:
            failures.append("replay phase did not hit a prefix record")

        print("== baseline phase (disk off) ==")
        r3 = run_child(python, script, args, "baseline", work, tail_file, baseline_out)
        print(r3.stdout[-800:])
        if r3.returncode != 0:
            failures.append("baseline phase exit=%d\n%s" % (r3.returncode, r3.stderr[-1500:]))

        if os.path.exists(replay_out) and os.path.exists(baseline_out):
            replay_tokens = json.load(open(replay_out, encoding="utf-8"))
            baseline_tokens = json.load(open(baseline_out, encoding="utf-8"))
            if replay_tokens != baseline_tokens:
                failures.append("restored continuation %s != recomputed %s" % (
                    replay_tokens, baseline_tokens))
            else:
                print("restored == recomputed: %s" % replay_tokens)
        elif not failures:
            failures.append("missing token outputs for equivalence check")
    finally:
        if args.keep:
            print("work dir kept at", work)
        else:
            shutil.rmtree(work, ignore_errors=True)

    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
