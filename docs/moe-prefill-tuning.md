# MoE 主机专家流式 prefill 调优

[返回 README](../README.md) · [Windows CUDA 构建](windows-cuda-build.md) · [混合推理](mixforward.md)

适用场景：`--device cuda` + `--moe_device cpu`（路由专家权重放在主机内存），长输入 prefill。
这套参数的默认值已经调好，正常**只需要给 `--chunked_prefill_size`**，不用记别的。

## 1. 推荐启动命令

```bat
run_ftllm.bat server <模型路径> --device cuda --moe_device cpu --threads 8 --port 8080 ^
  --chunked_prefill_size 4096 --moe_pinned_slots 16 --moe_pinned_slot_mb 4
```

- 不用 `set FTLLM_BUILD=gpu`：`run_ftllm.bat` 会自动挑（有 `build-vs-gpu\tools\ftllm\fastllm_tools.dll` 就用 gpu，否则 cpu）。
- 不用任何 `FASTLLM_*` 环境变量：`prefill_moe_gpu`、快照间隔都按配置自动决定（见第 2 节）。
- 只有当你要量性能、看归因日志时，才加 `set FASTLLM_PROFILE_MOE_STREAM=1`。

## 2. 参数与自动行为

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--chunked_prefill_size` | 模型决定 | 分块 prefill 切片大小。**这是这套里唯一必须显式管的参数** |
| `--moe_pinned_slots` / `--moe_pinned_slot_mb` | 16 × 4MB | 固定内存中转环的槽数与单槽宽；总量上限 256MB |
| `--prefill_moe_gpu` | `-1` 自动 | prefill 把主机专家搬上卡用 GPU 算；`1` 开 / `0` 关 |
| `--moe_pin_host_weights` | `0` 关 | 在主機侧钉住专家权重的额度(MB)，钉住后 DMA 直发，但内存不可换出 |
| `--prefix_cache_snapshot_interval_pages` | 自动 | 前缀快照间隔页数；显式给了就以它为准 |

自动行为（写这些参数就不用管了）：

1. **产物目录**：`FTLLM_BUILD` 不设置时自动挑 gpu / cpu。
2. **prefill 走 GPU**：专家在 **cpu** 内存里时默认开（`numa` / `disk` 有自己的快路径，不替它们决定）。
   老的环境变量 `FASTLLM_PREFILL_MOE_GPU=0/1` 仍然有效，等同于显式指定。
3. **快照间隔跟着 chunk 走**：没显式给 `--prefix_cache_snapshot_interval_pages` 时，
   间隔会被抬到 `ceil(chunk / pageLen) × pageLen`，保证「一个 chunk 正好落在一个快照边界上」。
4. **中转环**：槽位不够放下一个权重块时按行自动分片；环装不下才回退直拷。

## 3. 分块 prefill 与快照间隔（为什么要联动）

线性注意力层要求 chunk 边界落在前缀快照边界上，所以有 `chunk <= 间隔` 这个不变量。
以前的写法是把 chunk 钳回间隔，效果是**显式设的 `--chunked_prefill_size` 被静默忽略**：

```
effective = max(pageLen, min(--chunked_prefill_size, interval_pages × pageLen))
          = max(128,   min(4096,                  16 × 128 = 2048)) = 2048
```

现在反过来：`--chunked_prefill_size 4096` → `pages = max(16, ceil(4096/128)) = 32` → 间隔 4096，
两个不变量都保住（chunk ≤ 间隔、间隔是 pageLen 的整数倍），chunk 原样生效。

显式传 `--prefix_cache_snapshot_interval_pages` 时以它为准（老行为不变）。

验证办法：看进度日志的步长，`(4096/26012, 15%)` → `(8192, 31%)` 才对；如果是 2048，说明 chunk 被钳了。

## 4. 性能归因日志怎么读

```bat
set FASTLLM_PROFILE_MOE_STREAM=1
```

每 8 层打一行 `[fastllm-moe-stream]`：

| 字段 | 含义 |
| --- | --- |
| `experts` / `upload (N calls, X GB, Y GB/s)` | 本层命中的专家数、上传次数与字节；**`Y GB/s` 是 bytes/loop，不是单次拷贝速率** |
| `upload` | 主机侧上传总时长（含下面的 alloc / stageMemcpy / stageEnqueue） |
| `alloc` | 从显存池要一块目的缓冲的时间（延迟释放后每块都要重拿，正常十几 µs/次） |
| `stageMemcpy` | 把权重写进中转槽的主机 memcpy —— 正常情况这里占 `upload` 的 85~95% |
| `stageEnqueue` | `cudaMemcpyAsync` 入队时间（正常几 µs） |
| `cudaMalloc` | 池子没兜住、真调 `cudaMalloc` 的次数；非 0 才需要关心 |
| `pinHost` | 已钉住的主机权重 / 额度(MB) |
| `computeEnqueue` | GPU 侧 MoE 计算的入队时间 |
| `release` | 释放上一专家显存的时间 |
| `waitCur` | 等"当前专家拷贝完成事件"的时间（设备侧等待，主机不阻塞） |
| `find` / `evPrefetch` / `evCompute` | 找下一个专家、两次 `cudaEventRecord` 的时间，正常都是 0.0001 量级 |
| `slotWait` | 中转环被写满导致主机阻塞的时间；**> 0 说明 DMA 跟不上，可以把环加深** |
| `other` | 剩余项。注意它包含循环尾「等整层设备排空」的时间，所以主机比设备快时它会很大 —— 那时它就是设备时间，不是浪费 |
| `loop` | 本层墙钟时间（含排空），**跨版本对比只看这个** |

判读顺序：`loop` 大 → 看 `upload` 和 `other` 谁大：
`upload` 大 = 主机侧 memcpy 是瓶颈；`other` 大 = 设备侧（DMA/计算）是瓶颈。

## 5. DeepSeek-V4.1 的 GGUF 主机专家

V4.1 的路由专家走独立后端（Q8_K/FP8 block-32 数学），主机专家路径由
`--prefill_moe_gpu` 控制，语义与 Qwen3.5 一致：

| 值 | 行为 |
| --- | --- |
| `-1`（默认/auto） | 专家在 `cpu` 内存里就开 |
| `1` | 强制开，条件不满足时每次启动只提示一次并回落 CPU |
| `0` | 关 |

```bat
run_ftllm.bat server <模型路径> --moe_device cpu --prefill_moe_gpu 1 ^
  --chunked_prefill_size 4096 --threads 8
```

- 一批只上传**当前 chunk 路由到的**专家，显存预算 = 空闲显存 − 256MB − 该行块的工作区；
- 行数会再分块（4096/2048/…/64），工作区和专家驻留区互相让位，所以显存很小的机器也能跑；
- 每组、每行块的结果在 FLOAT32 输出上累加，最后由共享专家和 HC 边界按原路径收尾；
- `FASTLLM_PREFILL_MOE_GPU_MIN_TOKENS`（默认 64）决定多短的输入不值得上传；
- `FASTLLM_PROFILE_HOST_MOE=1` 打印每层 `host->gpu` 耗时。

约束：单卡、非 graph、专家为 pageable GGUF（`numa`/`disk` 走各自流水线）、
`--moe_device cpu`。不满足时不会报错，直接保留 CPU MoE。

## 6. 实测结论（RTX 2060 SUPER / PCIe 3.0 x16 / 64GB 主机内存）
长输入 prefill 从 **181 tok/s 提到 ~484 tok/s（2.7 倍）**，各步贡献：

1. 固定内存中转环 + 事件排序预取（摘掉「副本流等计算」的闸门）
2. `--chunked_prefill_size 4096` 生效
3. 中转环改用 `malloc + cudaHostRegister`（可 cache 映射）而不是 `cudaHostAlloc`（WC）

独立探针（`bw_probe.cu` 那类离线测量）量到的边界：

| 场景 | 速率 |
| --- | --- |
| 主机 memcpy 写中转槽（1.18MB，冷源，环上轮转） | WC 环 4.78 GB/s → 注册环 **6.88 GB/s**（4 线程 7.52） |
| 异步 H2D 从固定内存直发（1.18MB） | **9.4~10.0 GB/s** |
| 可换页内存同步 H2D（1.18MB） | 2.7 GB/s |
| 多线程 memcpy（1.18MB） | 2 线程无收益、4 线程 +9%，**不值得** |
| 非临时存储 / `cudaHostAllocMapped` | 无收益 / 更差 |

结论：**主机那一次 memcpy 是硬墙（~4.8~6.9 GB/s）**，它占 `upload` 的 85~95%、占 `loop` 约 81%。
想再上一个数量级只能省掉这次拷贝，也就是第 2 节的 `--moe_pin_host_weights`。

## 6. prefill 的 GEMM 内核选择（为什么换个模型就差 6 倍）

同一份权重形状、同一张卡，两个模型 prefill 能差好几倍，差别几乎全在「这些矩阵用什么内核算」。
按权重类型分三条路，prefill（大 M）要的是**张量核 + 大 M 的 cuBLAS**：

| 权重类型 | prefill 走什么 | 备注 |
| --- | --- | --- |
| NVFP4 / FP8 | 反量化到 fp16 暂存 → cuBLAS（`NVFP4_PREFILL_MIN_ROWS` / `FP8_PREFILL_MIN_ROWS` 以上） | 一直用 `CUBLAS_GEMM_DEFAULT_TENSOR_OP` |
| int4 group | Marlin（解码向内核） | 大 M 效率差；`FASTLLM_CUDA_INT4_GROUP_PREFILL_CUBLAS=1` 可改成反量化+cuBLAS。注意反量化路径的 cuBLAS 之前也是 `CUBLAS_GEMM_DEFAULT`（同样漏了张量核），本轮一并改成 `_TENSOR_OP` —— 所以「和 Marlin 持平」那次的结论是在旧行为下量的，要用 `=1` 重测 |
| fp16（未量化） | `cublasGemmEx` + `CUBLAS_GEMM_DEFAULT` | **以前这里没要张量核**，Turing 上退化成 SIMT fp16，大 M 慢一个数量级；本轮改成 `CUBLAS_GEMM_DEFAULT_TENSOR_OP` |

最后一行就是「KAT-Coder 比 Ornith 慢 6 倍」的主因，而且和模型配置直接相关：**看 `quantization_config.ignore`**。

- `KAT-Coder-V2.5-Dev`：`re:.*\.linear_attn\..*` —— 整个 linear_attn（含 `in_proj_qkvzba`，
  输出 12288/输入 2048，全模型最大的矩阵）都是 **fp16**，于是 prefill 全落在上面那条 fp16 路上。
- `Ornith-1.5-35B-A3B-NVFP4`：只 ignore 了 `conv1d` / `in_proj_a` / `in_proj_b`，`in_proj_qkvzba`
  是 NVFP4 —— 启动日志里那行 `[NVFP4 prefill cuBLAS] rows=4096 N=12288 K=2048` 就是它。

所以遇到「prefill 慢」，先看 ignore 列表和启动日志里有没有 `[* prefill cuBLAS]`，再看 `[fastllm-moe-stream]`
的 `upload/compute/slotWait` 构成，别一上来就怀疑 MoE 流式。

## 7. 已知边界（别踩）

- **钉权重需要约 23GB 空闲主机内存**（35B-A3B 的专家权重 ≈ 22.6GB），而且必须是「可换出」的那部分之外。
  在 commit 已经超过物理内存的机器上全钉会**反向崩**（实测掉到 0.4 GB/s，因为系统被迫换出其它热数据）。
- **部分钉没用**：只钉 8GB 时没钉住的 2/3 专家仍然决定整体节奏，实测和不开一样。
- **多线程、NT 存储、`cudaHostAllocMapped` 都别试**，上面量过。
- 换过钉住/大块测试之后，机器内存状态会漂；要对比就重启一次再测，否则基线自己就在动。

## 8. 退路环境变量

| 变量 | 作用 |
| --- | --- |
| `FASTLLM_CUDA_INT4_GROUP_PREFILL_CUBLAS` | int4 group 权重的 prefill 分流：默认关（保留 Marlin），`1` 改成反量化 + cuBLAS，`0` 显式关 |
| `FASTLLM_MOE_RING_HOSTALLOC=1` | 中转环退回 `cudaHostAlloc`（WC），做 A/B 对照用 |
| `FASTLLM_MOE_PINNED_UPLOAD=0` | 整体关掉固定内存中转，回退直接同步拷（慢很多） |
| `FASTLLM_MOE_PIN_HOST_WEIGHTS_MB=<MB>` | `--moe_pin_host_weights` 的环境变量形式 |
| `FASTLLM_PREFILL_MOE_GPU` | `--prefill_moe_gpu` 的环境变量形式，给了就等同显式指定 |
| `FASTLLM_PREFILL_MOE_GPU_MIN_TOKENS` | 上述路径生效的最小 token 数，默认 1024 |
| `FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES` | 快照间隔页数，优先级高于"跟着 chunk 自动" |
| `FASTLLM_MOE_PINNED_SLOTS` / `_SLOT_MB` | 中转环默认值（被同名命令行参数覆盖） |
| `FASTLLM_PROFILE_MOE_STREAM=1` | 打开第 4 节的归因日志 |
