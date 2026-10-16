# KVMem 参数填写手册（完整版）

面向 `ftllm server --kvmem '{...}'` 的使用者。设计语义部分引自 `docs/kvmem.md`，校验与选页逻辑引自 `src/kvmem.cpp`、`src/models/basellmkvmem.cpp`、`src/devices/cuda/fastllm-kvmem.cu`、`tools/fastllm_pytools/llm.py`（本文档位于 `docs/`，路径均相对仓库根）。

---

## 1. 一句话原理

上下文超过 GPU 常驻预算后，注意力不再看全部历史 KV，只看四个来源：

1. `sink_tokens`：序列最开头固定保留的一小段；
2. `prefix_tokens`：序列开头被"钉住"的一段，永远可见、不参与打分、不淘汰（用来保 chat template 的 system 段 / 工具 schema）；
3. `recent_tokens`：当前 query 往前的一段滑动窗口，必然可见；
4. `retrieval_tokens`：按 query 打分从旧页里挑出来的页（数量由它决定，不保证具体哪些）。

其余旧页被换出到主机内存，需要时再换回 GPU 槽位。

**关键认知**：没有被打分选中的页即便物理上还在 GPU 槽位里，也**不会**进入注意力视图，只是省一次换页开销。所以"精度"只由 `sink + prefix + recent + retrieval` 决定，`resident_tokens` 只决定开销与显存。

---

## 2. 三条硬规则（必背）

设页长 `P`（默认 **128**，可用 `--page_size` 改；multicuda 环境默认 16）。

| 规则 | 内容 | 违反时报错 |
| --- | --- | --- |
| R1 | 除 `max_tokens` 外，所有 token 预算必须是 `P` 的整数倍 | `KVMem: token budgets must be multiples of the KV page size` |
| R2 | `sink + prefix + recent + retrieval + prefill + P <= resident_tokens` | `KVMem: resident budget must cover sink + prefix + recent + retrieval + prefill + one boundary page` |
| R3 | `host_mib` 必须能装下 `max_tokens`（向上取整到整页）对应的全部完整注意力层 KV | `KVMem: host budget cannot back the configured logical context` / `KVMem host budget cannot back all full-attention layers at max_tokens` |

推导公式：

```
槽位数（页）      = resident_tokens / P
每轮可见旧页（页） = (sink + prefix + recent + retrieval) / P
每轮可见旧 token  = 上者 * P
resident 需满足   = 可见 token + prefill_tokens + P        （即 P 的一页边界余量）
```

对齐小工具：

```bash
python -c "P=128; v=lambda x: -(-x//P)*P; print('36000 ->', v(36000))"
```

---

## 3. 参数逐项说明

| 参数 | 类型 | 默认 | 单位 | 作用 | 调大 | 调小 |
| --- | --- | --- | --- | --- | --- | --- |
| `max_tokens` | int | 32768 | token | 逻辑上下文上限，**不必是 P 的倍数**；不得超过 `--tokens` / 模型原生上限 | 能承载更长请求，抬高 `host_mib` 需求 | 省主机内存 |
| `resident_tokens` | int | 8192 | token | 每层每卡的 GPU KV 槽位总数（页数 = 它 / P），池子在缓存创建时**一次性分配完** | 更多物理驻留、暂不改变可见页；抬高显存 | 省显存，但它只是 R2 的上限 |
| `sink_tokens` | int | 128 | token | 最开头固定保留，防"开头丢窗" | 一般别动 | 一般别动 |
| `prefix_tokens` | int | 0 | token | 开头钉住段。显式非 0 时**不做运行时推导**；为 0 时服务端按 chat template 推导 system 段长度（含 tools schema），再按 `resident-(sink+recent+retrieval+prefill+P)` 夹取并向上取整 | 指令遵守更稳，但吃 resident 且每轮要从前缀快照恢复，慢一点 | 把预算让给 `recent`/`retrieval` |
| `recent_tokens` | int | 2048 | token | 必留的滑动窗口，唯一无评分风险窗口，≥ P | 局部连贯、代码/引用一致性更好 | 长文能力下降最明显 |
| `retrieval_tokens` | int | 4096 | token | 每次刷新时按打分收纳的旧页额度（页数 = 它 / P） | 长程召回更强 | 记忆面变窄 |
| `prefill_tokens` | int | 512 | token | 单次追加/验证的最大 token 数，同时是 chunked prefill 的上限（`basellm.cpp:3766`） | 支持更大分块，吃 resident | 分块变小，速度略降；须 ≥ `draft_tokens + 1` |
| `host_mib` | int | 8192 | MiB | 该请求全部完整注意力层、全部设备的主机 KV 预算 | 支持更长逻辑上下文 | 可能 R3 报错 |
| `retrieval_interval` | int | 64 | 已提交 token | 推测解码（DFlash/MTP）检索评分复用间隔；1 = 逐轮刷新；**不必按页对齐** | 更省换页，但查询变了仍用旧评分，质量有风险 | 更准更慢 |
| `retrieval_norm_power` | float | 0.0 | - | 评分归一化指数：0 = 原始点积（高范数页占优：模板、重复文本）；1 = 纯余弦（只看方向对齐） | 内容页更难被高频模板挤掉 | 模板/重复文本更占优 |
| `enabled` | bool | true | - | 关掉等于不启用 | - | - |

---

## 4. 示例模型核算（Qwen3.5-MoE 35B-A3B，10 层完整注意力）

示例模型 `Qwen3_5MoeForConditionalGeneration`（35B-A3B，NVFP4 权重）的 `config.json`：40 层，`full_attention_interval=4` → 只有 **10 层完整注意力**参与分页；`num_key_value_heads=2`，`head_dim=256`。

每 token 每层 KV 字节 `= 4 * num_kv_heads * head_dim = 4 * 2 * 256 = 2048 B`（`qwen3_5.h:109`），单卡合计：

```
每 token 总 KV = 2048 B * 10 层 = 20480 B = 20 KiB
```

于是：

| 量 | 公式 | 例 |
| --- | --- | --- |
| GPU 分页池（单卡） | `resident_tokens * 20 KiB` | 40192 → 785 MiB |
| 检索索引（FP32 mean-K） | `ceil(max_tokens/P) * 2 heads * 256 * 4 B * 10 层`，按提交量懒增长 | 262144 → 上限约 40 MiB |
| 检索评分缓存 | `页数 * Q heads * 4 B * 10 层` | 262144 → 约 1.25 MiB |
| 主机 KV | `max_tokens * 20 KiB` | 262144 → **5120 MiB** |

常驻档位与显存对照（单卡，不含权重/工作区/GDN 状态）：

| resident_tokens | 槽位页数 | GPU KV 池 | 扣掉 prefill 4096 + 1 页后可用的可见 token |
| --- | --- | --- | --- |
| 29952 | 234 | 585 MiB | 25728 |
| 36096 | 282 | 705 MiB | 31872 |
| 40192 | 314 | 785 MiB | 35968 |
| 67200 | 525 | 1313 MiB | 62976 |

`max_tokens` 与 `host_mib` 对照：

| max_tokens | host_mib 下限 |
| --- | --- |
| 131072 | 2560 |
| 262144 | 5120 |
| 524288 | 10240 |
| 1000000 | 19531 |

---

## 5. 填参数的标准流程

1. 定 `max_tokens` = 你需要的最长上下文，且 ≤ `--tokens`。
2. 定 `host_mib` ≥ 上表下限，建议再留 20%。
3. 定"可见旧 token"目标 V（必须是 P 的倍数），页数 `V/P`。
4. `resident_tokens = V + prefill_tokens + P`（结果必须是 P 的倍数）。
5. 把 V 分给 `sink(128) / prefix / recent / retrieval`，四项都按 P 对齐，四 = V。
6. 核对 R1/R2 两式，再估 GPU（`resident * 20 KiB`）是否在显存余量内。
7. 复核 `prefill_tokens >= chunked_prefill_size`（否则实际分块被 `prefill_tokens` 压小）以及 `prefill_tokens >= draft_tokens + 1`（推测解码）。

### 分配经验

- `recent` 是唯一"必然可见"的窗口，通用质量最敏感，别低于 8192；能给的场景优先给满 16384。
- `prefix` 只在"system/工具 schema 很长且必须以原样常在"时才值钱；它按**开头**截取，给 9216 就只保 system 段的前 9216 token，尾部（往往正是 tools schema）仍会丢。
- `retrieval` 是最"廉价"的长程扩展，但没有保证；页数太少时长文问答容易答偏。
- 想要"system 段 30k 全保 + 长文能力"，只能把 resident 抬到能同时容下的档位（例：`prefix 30080 + recent 16384 + retrieval 16384 + prefill 4096 + sink 128 + 128 = 67200`）。

---

## 6. 现成配方（可直接复制）

### 6.1 通用长文 / 长对话（可见约 36k，prefix 交给运行时推导）

```json
{"max_tokens":262144,"resident_tokens":40192,"sink_tokens":128,"prefix_tokens":0,"recent_tokens":16384,"retrieval_tokens":16384,"prefill_tokens":4096,"host_mib":6144,"retrieval_interval":64}
```

运行时最多 pin 3072 token 的 system 段（自动夹取）。

### 6.2 agent / 工具调用（保前缀，可见约 36k）

```json
{"max_tokens":262144,"resident_tokens":40192,"sink_tokens":128,"prefix_tokens":9216,"recent_tokens":16384,"retrieval_tokens":10240,"prefill_tokens":4096,"host_mib":6144,"retrieval_interval":64}
```

### 6.3 显存紧张（可见约 25.7k）

```json
{"max_tokens":262144,"resident_tokens":29952,"sink_tokens":128,"prefix_tokens":0,"recent_tokens":16384,"retrieval_tokens":8192,"prefill_tokens":4096,"host_mib":6144,"retrieval_interval":64}
```

### 6.4 必须原样保留 30k system 段（可见约 60k）

```json
{"max_tokens":262144,"resident_tokens":67200,"sink_tokens":128,"prefix_tokens":30080,"recent_tokens":16384,"retrieval_tokens":16384,"prefill_tokens":4096,"host_mib":6144,"retrieval_interval":64}
```

GPU 分页池升到约 1313 MiB。

### 6.5 一百万 token 逻辑上下文

```json
{"max_tokens":1000000,"resident_tokens":40192,"sink_tokens":128,"prefix_tokens":0,"recent_tokens":16384,"retrieval_tokens":16384,"prefill_tokens":4096,"host_mib":20480,"retrieval_interval":64}
```

须同时加 `--max_context_length 1000000 --rope_scaling yarn`（Qwen3.5 原生 262144）。

---

## 7. 与其它命令行参数的关系

- `--page_size`：全局设置页长（`src/fastllm.cpp:713`）。cuda + cpu 组合默认 128；multicuda 默认 16。改它就要让所有预算跟着对齐。
- `--chunked_prefill_size`：JSON 未写 `prefill_tokens` 时，服务端用它当默认值（`util.py:1941`）；写进 JSON 以 JSON 为准。反过来，启用 KVMem 后实际分块上限 = `prefill_tokens`（`basellm.cpp:3766`）。
- `--kv_cache_limit`：与 KVMem 互斥，同时给会报错（`util.py:1939`）。
- `--cache_history`：KVMem 强制关闭历史缓存（`basellmkvmem.cpp:72`）。官方长上下文示例都显式写 `--cache_history false --prefix_cache false`，建议跟随以免歧义。
- `--max_batch`：KVMem 强制单请求（`basellmkvmem.cpp:42`、`llm.py:3502`）。**并发请求会撞到** `KVMem currently requires one request`，服务端要串行化调用方。
- 精度相关：`--atype` 与 `--kv_cache_dtype` 必须一致且为 FP16/BF16（`basellmkvmem.cpp:27`），所以不能配 FP8 KV（示例模型的 `kv_cache_scheme` 8bit 校准对 KVMem 无效）。
- `--moe_device cpu` 等 MoE/CPU 侧参数与 KVMem 无关（KV 全在 CUDA）。
- 需要模型走新 GPU 引擎并完成 warmup；`USE_OLD_ENGINE=1` 或跳 warmup 会被拒绝（`basellmkvmem.cpp:46-51`）。

---

## 8. 报错对照表

| 报错 | 原因 | 修法 |
| --- | --- | --- |
| `token budgets must be multiples of the KV page size` | 某个预算不是 128 的倍数 | 全部按 128 对齐 |
| `resident budget must cover sink + prefix + recent + retrieval + prefill + one boundary page` | 预算超订 | 抬 `resident_tokens` 或砍窗口 |
| `host budget cannot back the configured logical context` / `cannot back all full-attention layers at max_tokens` | `host_mib` 不够 | 加 `host_mib`（示例模型 ≥ `max_tokens * 20 KiB`）或减 `max_tokens` |
| `KVMem logical context exceeds the model's configured context` | `max_tokens` 超过 `--tokens`/模型上限 | 降 `max_tokens` 或配 `--max_context_length` |
| `KVMem requires matching FP16/BF16 compute and KV dtypes` | 用了 FP8 KV 或 atype/kv 不一致 | 两边都设 fp16 或 bf16 |
| `KVMem currently requires max_batch=1` / `KVMem currently requires one request` | 并发或 batch>1 | 单请求串行 |
| `KVMem: prefill_tokens must cover drafts + 1 verification tokens` | 推测解码窗口太小 | `prefill_tokens >= draft_tokens + 1` |
| `KVMem requires warmup` / `must be configured before warmup or inference` | 跳了 warmup / 配置时机不对 | 走正常 `--kvmem` 启动流程 |
| `KVMem does not support history caching` | 开了历史缓存 | `--cache_history false` |
| `KVMem: host budget exhausted`（运行时） | 冷页备份超出 `host_mib` | 加 `host_mib` |
| `KVMem currently supports text-only requests` | 多模态请求 | 只发文本 |
| `KVMem requires a CUDA device map it can page on` | 设备映射不是 `cuda` / 非 KV-head 切分的 TP | 用 `--device cuda` 或 KV-head 切分 TP |

---

## 9. 启动后怎么确认生效

- `[Fastllm] KVMem prefix pin: N tokens`：推导出的 system 段长度（只在显式 `prefix_tokens=0` 时打印）。
- `[Fastllm] KVMem prefix pin clamped to N tokens (requested M)`：推导值被 resident 上限截短，说明 `resident_tokens` 太小或 `recent/retrieval` 占太多。
- 没有 CLI 打印页命中率/换页统计；`KvMemStats`（`include/kvmem.h`）只在代码侧可读。
- 想验证某个 `prefix_tokens` 值是否够，可临时把它设 0 跑一次，读上面第一条打印的真实 system 长度。

---

## 10. 精度注意事项

- KVMem 是稀疏注意力近似，"KV 字节无损换页"不代表输出等于完整注意力（`docs/kvmem.md` 开头）。
- `retrieval_interval` 越大越省，但查询变化时会继续用旧评分，输出质量有风险；逐轮刷新设 1。
- 若发现模型反复咬住模板/重复文本而忽略正文，把 `retrieval_norm_power` 提到 0.3~0.5。
- 扩大 RoPE 范围（`--rope_scaling yarn`）与稀疏选页是两件独立的事，容量扩了不代表质量自动达标，需分别验证。
