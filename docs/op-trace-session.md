# Op Trace 全量会话归档与离线查看器规划

> 状态：P0-P4 已完成（导出、静态 HTML、会话管理、上传导入）。
> 目标读者：维护者。
> 最后更新：2026-11-19

## 0. 实施状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| P0 实时页 prefill 固定保留 | 已完成（2026-11-19） | native `pinned` 队列 + 页面分阶段裁剪 + `lost` 不清空；已通过 MSVC 单元冒烟测试与 CPU/GPU 构建 |
| P1 采集落盘 | 已完成（2026-11-19） | `ArchiveWriter` / chunk 编码 / `chunks.idx` / `meta.json` / 实时聚合；已通过 native 冒烟、强杀恢复检查、CPU/GPU 构建 |
| P2 读取器 + Viewer + 系统对话框 | 已完成（2026-11-19） | 目录/`.ftrace` 读取、恢复、分页、摘要、打包、FastAPI viewer、系统对话框、`ftllm op-trace-view` 快速路径 |
| P3 页面离线模式 + 自动打包 | 已完成（2026-11-19） | 数据源切换、分页、URL 恢复、summary 卡片、prefill/token 跳转、server 自动打包、启动补齐 |
| P4 导出、静态查看、会话管理与导入 | 已完成（2026-11-19） | Chrome Trace/Perfetto 导出、纯静态 `.ftrace` 查看器、会话管理/手动删除、浏览器上传导入 |

## 1. 背景

当前算子级 Trace（`/debug/op-trace`）的数据链路是：

1. native 侧 `fastllm::op_trace::Store` 用定长 `std::deque<Event>` 环形缓冲保存事件，
   容量默认 65536（`FASTLLM_OP_TRACE_CAPACITY`）。
2. Python server 通过 `op_trace_snapshot_json` 以 `after_seq` 增量拉取。
3. 浏览器把事件持续 append 到页面内存，上限同样是 65536 条
   （`op_trace_ui.py` 中 `MAX_EVENTS`）。

问题：

- decode 每 token 会产生数百到数千条事件（层数 × 算子数 + `[gap]`），
  prefill 事件位于队列头部，几乎在 decode 开始后几十个 token 内就被
  `pop_front` 覆盖（`op_trace.h` 的 `Store::Record`）。
- 页面侧也会按 `MAX_EVENTS` 从最旧事件裁剪；一旦检测到 `lost`
  还会清空全部已加载事件重新拉取。
- 结果：prefill 结束进入 decode 后，页面筛选 `phase=prefill` 没有数据，
  空表提示“更早的事件可能已被覆盖”。

## 2. 目标与非目标

### 2.1 目标

1. **全量采集**：从启用采集开始，到关闭/清空/进程退出为止，
   所有 `Record()` 产生的事件都落盘，不做抽样，不受内存环形队列影响。
2. **崩溃可恢复**：服务端是 `TerminateProcess` 硬退出
   （`server.py` 的 `_terminate_now`），不执行析构；归档必须容忍强杀，
   下次打开时自动恢复，最多丢失最后一个 flush 周期（约 200ms）。
3. **自动打包**：会话结束时自动生成单文件 `<sessionId>.ftrace`，
   便于拷贝、发送和用系统文件对话框直接选择；原目录默认保留。
4. **离线查看器**：`ftllm op-trace-view` 启动一个轻量网页服务，
   **不加载模型、不加载推理 native 库**；网页内通过系统文件对话框
   选择磁盘上的 session 目录或 `.ftrace` 文件即可浏览。
5. **复用现有页面**：实时页与离线页共用 `op_trace_ui.py` 的 `PAGE_HTML`，
   保留时间线、热点、Token 趋势、事件详情等交互。
6. **修复实时页 prefill 消失**：全 session 的 Prefill/Decode 卡、
   Token 趋势改为读取会话级聚合（live 用 native 聚合，离线用归档摘要），
   不再依赖页面缓冲区里是否还留着原始事件。

### 2.2 非目标

- 不做执行回放（trace 是记录，不是可重放的执行日志）。
- 不做跨进程实时流式聚合（每个服务进程各写自己的 session）。
- 纯静态单 HTML（浏览器内 JS 解析，无进程）已在 P4 实现。
- 不做 decode 抽样、会话自动轮转删除。全量是明确需求；磁盘不足时停止采集并告警，
  不静默删除历史。P4 提供会话管理里的手动删除，录制中的会话不可删除。
- 不做自动上传/云端分析；P4 只提供用户主动选择 `.ftrace` 后上传到本机
  viewer 归档根目录的导入操作。

## 3. 已确认决策

| 决策 | 结论 | 说明 |
|---|---|---|
| 采集范围 | 全量，不抽样 | `DECODE_EVERY=1`；容量上限 `0` 表示不限 |
| 会话结束 | 自动打包 | 封会话后后台生成 `.ftrace`，源目录默认保留 |
| 离线选择方式 | 系统文件对话框 | 对话框弹在 viewer 所在机器；无桌面时回退路径输入 |
| 查看器形态 | 独立轻量进程 | `ftllm op-trace-view`，只依赖 stdlib + fastapi/uvicorn |
| 单文件格式 | zip64 | 用 Python `zipfile` 打包（deflate），native 不引入 zlib |
| 纯静态 HTML | P4 已实现 | 单 HTML + 浏览器内解析 `.ftrace` |
| 保留策略 | 永久保留 | P4 会话管理支持手动删除；磁盘余量低于阈值停止采集并显式告警 |

## 4. 现有代码锚点

| 模块 | 位置 | 说明 |
|---|---|---|
| 事件模型 / 环形缓冲 | `include/utils/op_trace.h` | `Event`、`PhaseKind`、`PhaseScope`、`Store::Record`、`SnapshotJsonImpl` |
| 环形淘汰 | `include/utils/op_trace.h` `Store::Record` | `events.size() >= capacity` 时 `pop_front()` |
| 容量配置 | `include/utils/op_trace.h` `ParseEnvCapacity` | 默认 65536 |
| phase 埋点 | `src/models/basellm.cpp` 约 3186-3202、4129-4139 | `PhaseScope(kind, token, request)` |
| 算子记录 | `include/utils/op_trace_utils.h` | `RecordOp`、`LayerIndex`、`WeightName`、`TensorSummary` |
| C 导出 | `tools/src/pytools.cpp` 约 380-399 | `set_op_trace_enabled`、`clear_op_trace`、`op_trace_snapshot_json`、`free_op_trace_string` |
| Python 绑定 | `tools/fastllm_pytools/llm.py` 约 744-783 | `op_trace_supported/set_enabled/snapshot/clear` |
| HTTP 接口 | `tools/fastllm_pytools/server.py` 约 321-382 | `/debug/op-trace`、`/api/op-trace/{status,snapshot,enabled,clear}`，localhost/dev_mode 门禁 |
| 页面 | `tools/fastllm_pytools/op_trace_ui.py` | `PAGE_HTML`（单文件字符串）、轮询、筛选、渲染 |
| 页面上限 | `op_trace_ui.py` 约 699 | `MAX_EVENTS = 65536`、`KEEP_US = 1h` |
| 页面增量拉取 | `op_trace_ui.py` 约 981-1070 | `appendSnapshot`、`trimEvents`、`poll`，P3 对 `lost` 去重合并 |
| 阶段聚合 | `op_trace_ui.py` 约 1265-1360 | `buildPhaseTokenStats`、`renderPhaseTokens` |
| 阶段筛选 | `op_trace_ui.py` 约 2571 | `setPhaseFilter` |
| CLI 子命令 | `tools/fastllm_pytools/cli.py` 约 138-201 | `args_parser()` 与 `main()` 分发；`router` 快速路径在 142-146 |
| Python 拷贝 | `CMakeLists.txt` 约 3877 / 3908 | `copy_directory tools/fastllm_pytools`，新增 .py 会自动带上 |

构建与验证注意（见 `AGENTS.md`）：

- Python 侧改动要重新拷贝到运行目录：
  `scripts/windows/build_tools.bat gpu`（或手工拷贝到 `build-vs-gpu/tools/ftllm/`）。
- native 改动需要重新编译 `fastllm` / `fastllm_tools`；`build_gpu.bat` 只构建 `main`，
  不会触发 Python 拷贝。

## 5. 总体架构

```
                    ┌─────────────────────────────┐
模型线程 Record() ──┤ Store（内存 ring，实时页用） │
                    │   └─ ArchiveWriter（同锁追加）│
                    └──────────────┬──────────────┘
                                   │ 后台线程 flush
                                   ▼
                <trace_dir>/<sessionId>/（meta.json / chunks.idx / chunks/*.bin）
                                   │
                     封会话（disable/clear）后 Python 后台打包
                                   ▼
                         <sessionId>.ftrace（zip64）
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        ▼                                                     ▼
在线 server.py（有模型）                             ftllm op-trace-view（无模型）
  /api/op-trace/* 原有接口                             /api/op-trace/*（归档实现）
  /api/op-trace/summary（native 聚合）                 /api/op-trace/open|pick|page|summary
        └──────────────────── 共用 PAGE_HTML ─────────────────┘
```

原则：

- **写路径绝不阻塞模型**：`Record()` 只在现有 mutex 内把 `Event` 移入 chunk 缓冲；
  序列化、写盘、打包都在后台线程/进程完成。队列满时丢弃并计数，不影响推理。
- **读路径与 native 解耦**：归档读取完全由 Python 完成，因此 viewer 不依赖
  模型、CUDA、native 库版本；重启服务、换机器、单文件拷贝都能打开。
- **实时与离线共用页面**：页面通过 `/api/op-trace/status` 返回的
  `mode=live|viewer` 切换数据源，渲染逻辑复用。

## 6. 磁盘格式

### 6.1 目录布局

```
${FASTLLM_OP_TRACE_DIR:-<cwd>/.fastllm/op_trace}/
  <sessionId = yyyyMMdd-HHmmss-<pid>>/
    meta.json
    chunks.idx
    chunks/
      00001.bin
      00002.bin
      ...
  <sessionId>.ftrace            # 自动打包产物
  <sessionId>.ftrace.tmp        # 打包中的临时文件，可安全删除
```

- 每个服务进程独立 session 目录，无共享可变索引；查看器扫描目录即可列出全部会话。
- 目录名只含 ASCII，chunk 文件层级浅，规避 Windows 260 字符路径限制。

### 6.2 `meta.json`

```json
{
  "schema": 1,
  "session": "20261119-142341-12345",
  "pid": 12345,
  "host": "dev-pc",
  "created_at_ns": 1763543021123456789,
  "wall_base_ns": 1763543021123456789,
  "ended_at_ns": null,
  "sealed": false,
  "model": "Qwen3-...",
  "config": {
    "chunk_events": 4096,
    "flush_ms": 200,
    "io": true,
    "gaps": true,
    "gap_us": 20
  },
  "archive": {
    "committed_seq": 4096,
    "chunks": 1,
    "bytes": 81234,
    "dropped": 0,
    "state": "recording",
    "stopped_reason": "",
    "packed": false,
    "ftrace": null,
    "pack_error": ""
  }
}
```

- `committed_seq` 是**已封块**的最后一个 seq；读取端最多读到它。
- `state`: `recording | sealed | truncated | stopped`。
- `meta.json` 每 1s 原子更新（写 `meta.json.tmp` → `os.replace`）。
- 时间：`*_at_ns` 为 Unix 墙钟，仅用于会话列表/展示。**事件时间统一使用单调时钟**
  （与现有 snapshot 的 `t_us` 同一时钟，单位微秒）；`chunks.idx` 与分页接口里的
  `t0_us/t1_us/start_us/end_us` 都是同一单调时钟。如需显示绝对时间，用
  `wall_base_ns / 1000 + (t_us - t0_us)`（结果单位 us）换算；session 之间不要求可比。

### 6.3 chunk 负载（`chunks/NNNNN.bin`）

自包含，便于单块恢复与并行读取：

```
[ChunkHeader]
  magic       4B   "FOTR"
  version     u16  1
  flags       u16
  eventCount  u32
  firstSeq    u64
  baseTNs     u64   块内第一条事件的单调时钟起点
  stringsLen  u32
  payloadLen  u32
[StringTable]
  count       u16
  entries     [u16 len + UTF-8 bytes] × count   // op/weight/device/io 块内去重
[TidTable]
  count       u8
  values      u64 × count                        // 线程 id 块内字典
[Columns]（按列连续存放，便于按需解码）
  seq        : 0 起 delta varint（后续通常为 1）
  t_us       : 相对 baseTNs 的 us，zigzag delta varint（允许乱序事件为负）
  dur_ns     : varint
  tid        : TidTable 下标（u8，255 表示溢出）
  layer      : zigzag varint（-1 表示未标注）
  req        : zigzag varint
  token      : zigzag varint
  flags      : u8（bit0 = gap）
  phase      : u8（0 none / 1 prefill / 2 decode / 3 mixed）
  opId       : u16（StringTable 下标）
  devId      : u16（StringTable 下标）
  weightId   : u32（0xFFFFFFFF 表示空）
  ioId       : u32（0xFFFFFFFF 表示空）
  xfer_ns    : varint
  reshape_ns : varint
  run_ns     : varint
  sync_ns    : varint
```

- 字段语义与现有 `Event` 一致；`phase` 用 `PhaseName` 的字符串由 `phase` 枚举映射。
- `io` 只在 `FASTLLM_OP_TRACE_IO=1` 时记录；`TensorSummary` 的 16 个/800 字符上限保持不变。
- `payloadLen` 覆盖 **TidTable + Columns**；`stringsLen` 只覆盖 StringTable。
- 目标体积约 15-25B/事件（关闭 `io` 时更小）。
- **schema 版本**：`meta.json.schema` 与 chunk `version` 同时校验；读取器遇到未知版本
  明确报错，不做静默降级。

### 6.4 `chunks.idx`

追加式 JSONL，每行一个已封块（与 `meta.json` 的 `committed_seq` 同步）：

```json
{"file":"chunks/00001.bin","first_seq":1,"last_seq":4096,
 "t0_us":1234567890,"t1_us":1234644433,
 "phases":{"prefill":4096,"decode":0,"mixed":0,"none":0},
 "reqs":[{"req":0,"first_seq":1,"last_seq":4096}],
 "tokens":[{"req":0,"token":0,"first_seq":1,"last_seq":4096,
            "events":4096,"total_us":76543,"run_us":60000,"xfer_us":5000,
            "sync_us":3000,"gap_us":8543,"cuda_us":70000,"cpu_us":6543,
            "start_us":1234567890,"end_us":1234644433,"has_split":true}],
 "ops":[{"phase":"decode","op":"Linear","total_us":12345,"count":100,"max_us":300}],
 "bytes":81234}
```

> 示例数值仅表示单位为微秒的单调时钟，不是墙钟。`tokens[].start_us/end_us` 为
> 该 token 首/末事件的时间；`latency_us = end_us - start_us`。

- `tokens` 是摘要的来源；token 级 `first_seq/last_seq` 用于离线跳转。
- `ops` 每块可限制 top-K（例如 32），全 session 热点由各块合并后再取 top-N。
- 读取端只在打开会话时顺序读 `chunks.idx`（几十万行也仅几 MB），事件体按需解码。
- P2 追加了向后兼容字段：`phase_us`（各阶段耗时和）、`gaps`（空隙 top-K）、
  `devices`（设备事件/耗时）、token 的 `other_op_us`；旧索引缺少时读取器回退扫描
  块事件，新索引可直接合并摘要。

### 6.5 写入与提交协议

1. `Record()`（持 `Store::mutex`）：把 `Event` 移入当前 chunk 缓冲。
2. chunk 满（默认 4096 条）：把缓冲放入 `pending` 队列（容量 8），唤醒写线程。
3. 写线程：序列化 → 写 `chunks/NNNNN.bin.tmp` → `flush` → rename →
   追加一行到 `chunks.idx` 并 `flush` → 更新 `meta.json`（committed_seq/bytes/chunks）。
4. 每 `flush_ms`（默认 200ms）即使未满也封块，限制强杀丢失窗口。
5. `pending` 满：丢弃新块，`archive.dropped += n`；不阻塞模型。
6. 磁盘余量 < `FASTLLM_OP_TRACE_MIN_FREE_MB`（默认 2048）：
   `state=stopped`、`stopped_reason=low_disk`，停止归档但保留 ring 与告警日志。
7. 写文件名使用固定 5 位序号，避免依赖目录 mtime 排序。

### 6.6 会话生命周期

| 触发 | 行为 |
|---|---|
| `SetEnabled(true)` | 新建 session（目录 + meta），启动写线程；ring 继续工作 |
| `SetEnabled(false)` | flush 未满块 → 写 `sealed=true`、`ended_at_ns` → 后台打包 |
| `Clear()` | 若正在采集：封当前 session 并新建；若未采集：仅清 ring/epoch |
| 进程硬退出 | 无析构；已封块有效，未封的 chunk 缓冲丢失（≤200ms）；下次打开时恢复 |
| 服务启动 | 扫描 trace_dir：`sealed=false` 且属主 pid 不存活 → 恢复；`sealed=true` 未打包 → 自动打包 |
| 查看器打开 | 只读；`recording` 会话只读到 `committed_seq`，界面显示“录制中” |

**恢复算法**：

1. 读 `chunks.idx` 与 `chunks/` 列表，忽略 `.tmp` 和未入索引的块。
2. 按 `first_seq` 排序，校验：filename 连续、header magic/version、`eventCount`、
   `firstSeq` 与上一块 `last_seq + 1` 连续。
3. 遇到第一个不连续的块即截断（保留前面有效部分），重写 `chunks.idx`/`meta.json`。
4. 若 `sealed=false`：写 `ended_at_ns`、`sealed=true`，触发自动打包。

### 6.7 自动打包（`.ftrace`）

- 由 **Python** 完成（`zipfile.ZipFile(..., ZIP_DEFLATED, allowZip64=True)`），
  native 不引入 zlib 依赖。
- 触发点：
  - server 处理 `/api/op-trace/enabled`（关闭）或 `/api/op-trace/clear` 后，
    调用 native `op_trace_flush_and_seal()`，拿到 session 目录，投递后台打包任务；
  - 服务/查看器启动扫描时，补齐未打包的 `sealed` 会话；
  - 查看器打开未打包目录时，可选择“后台打包”，不阻塞浏览。
- 流程：写 `<sessionId>.ftrace.tmp` → `os.replace` 为 `<sessionId>.ftrace` →
  `meta.json.archive.packed=true/ftrace=<相对路径>`；失败写 `pack_error`，保留目录。
- 包内结构：

```
meta.json
chunks.idx
chunks/00001.bin
...
```

- 查看器通过 `zipfile.ZipFile` 随机读取，不落盘解压；支持 >4GB（zip64）。
- `FASTLLM_OP_TRACE_DELETE_DIR_AFTER_PACK=0`：默认打包后保留源目录；
  设为 1 时仅在打包成功后删除 `chunks/` 与索引（`meta.json` 保留）。

## 7. Native 实现

### 7.1 新文件

- `include/utils/op_trace_types.h`：`Event`/`Timing`/`PhaseKind`/`PhaseName`/`PhaseCode`、
  JSON 字符串与通用 env 解析；供 `op_trace.h` 与归档实现共用。
- `include/utils/op_trace_archive.h`：`ArchiveConfig` 与归档/聚合的自由函数声明
  （不包含 `<filesystem>`，避免污染 `op_trace.h` 的 CUDA 编译单元）。
- `src/utils/op_trace_archive.cpp`：归档写线程、chunk 编码、`chunks.idx`、`meta.json`、
  磁盘阈值与实时聚合；已加入 `FASTLLM_CXX_SOURCES`。
- 修改 `include/utils/op_trace.h`：`Store` 在 `Record` 中调用归档/聚合；`SetEnabled/Clear`
  生命周期；`SnapshotJson` 增加 session/archive 字段。
- 修改 `tools/src/pytools.cpp`：新增 C 导出。

### 7.2 关键接口（实际实现）

```cpp
namespace fastllm { namespace op_trace {

struct ArchiveConfig {
    bool enabled = true;
    std::string dir;
    size_t chunkEvents = 4096;
    size_t flushMs = 200;
    size_t pendingChunks = 8;
    uint64_t minFreeBytes = 2ull * 1024 * 1024 * 1024;
    uint64_t sessionMaxBytes = 0;
    bool io = true;
    bool gaps = true;
    uint64_t gapUs = 20;
};

ArchiveConfig ArchiveConfigFromEnv(bool io, bool gaps, uint64_t gapUs) noexcept;
void ArchiveConfigure(const ArchiveConfig &config) noexcept;
void ArchiveSetModel(const std::string &model) noexcept;
void ArchiveEnsureSession() noexcept;
void ArchiveAppend(const Event &event) noexcept;
bool ArchiveFlush(bool seal) noexcept;      // 内部 5s 超时；C 导出忽略返回值
void ArchiveRotateSession() noexcept;       // flush(seal) + 新 session
void ArchiveShutdown() noexcept;

std::string ArchiveSessionId() noexcept;
std::string ArchiveStatusJson() noexcept;   // snapshot 的 archive 对象
std::string ArchiveSessionJson() noexcept;  // 当前/最近 session 的 meta 摘要

void AggregateRecord(const Event &event) noexcept;
void AggregateClear() noexcept;
std::string ArchiveSummaryJson(size_t maxTokens) noexcept;

}}
```

- 归档状态是 `.cpp` 内单例；`Store` 不持有 `ArchiveWriter`，避免头文件拖入
  `<filesystem>/<fstream>`，也避免 `op_trace.h` 被 nvcc 编译时报错。
- 锁顺序：`Store::mutex → archive mutex`（`Record` 调用 `ArchiveAppend`）、
  `Store::mutex → aggregate mutex`（`Record` 调用 `AggregateRecord`）。
  归档/聚合实现都不回调 `Store`；`Flush/Rotate/EnsureSession` 由 `SetEnabled/Clear`
  在**释放** `Store::mutex` 后调用，等待写线程时不会阻塞推理线程持有 Store 锁。
- `ArchiveAppend` 只做 `current.push_back(event)`；满 `chunkEvents` 移入 `pending`
  （容量 8，满则丢弃新块并计数）。`current` 在 session 开始/换块后 `reserve`，
  避免频繁扩容复制 Event。
- 写线程使用 `std::condition_variable`，按 `flushMs` 超时封块（默认 200ms），
  chunk 文件写 `.tmp` → rename → 追加 `chunks.idx` → 更新 `meta.json`。
- `ArchiveFlush(true)` 由禁用/清空路径调用，带 5s 等待上限；超时记录
  `flush_timeout` 并返回 false，`Rotate` 在超时时不会启动新 session，避免写线程
  仍在写旧目录时切换。
- 强杀进程时最多丢失当前未封块（≤`flushMs`）；已 rename 的 chunk 与已追加的
  index 行保持完整，`.tmp` 和未入索引的文件由 P2 恢复逻辑忽略/重建。

### 7.3 Snapshot 扩展

在现有字段后追加（保持向后兼容）：

```json
{
  "schema": 1,
  "session_id": "20261119-142341-12345",
  "archive": {
    "enabled": true,
    "dir": "E:/traces/20261119-142341-12345",
    "committed_seq": 4096,
    "chunks": 1,
    "bytes": 81234,
    "dropped": 0,
    "state": "recording",
    "sealed": false,
    "packed": false
  }
}
```

### 7.4 C 导出（`pytools.cpp`）

```cpp
DLL_EXPORT void op_trace_flush(bool seal);
DLL_EXPORT char *op_trace_session_json();       // 当前/最近 session 的 meta 摘要
DLL_EXPORT char *op_trace_summary_json(int max_tokens); // 全 session 聚合（见 8.4）
DLL_EXPORT void op_trace_archive_shutdown();
DLL_EXPORT void set_op_trace_model(const char *model);
```

- `op_trace_summary_json` 的数据来自 **native 常驻聚合**（见 8.4），
  因此实时页在 ring 淘汰后仍能显示完整 prefill/token 统计。
- `max_tokens <= 0` 时取默认值；实现把单次输出限制在 8192 条 token，
  避免实时 HTTP 响应过大。
- `llm.py` 已绑定：`flush_op_trace`、`get_op_trace_session`、
  `get_op_trace_summary`、`shutdown_op_trace_archive`、`set_op_trace_model`。
- `server.py` 在启动时把 `model_name` 传给 native，供 `meta.json.model` 使用。
- 所有字符串返回值沿用 `string_to_chars` + `free_op_trace_string` 的约定。

### 7.5 实时聚合（修复 prefill 消失）

在 `src/utils/op_trace_archive.cpp` 内维护轻量聚合，`Record` 时随事件更新
（独立 mutex，O(1) 查找）：

```
per (req, token): events, total_us, run/xfer/sync/other_op, gap_event,
                  cuda/cpu/other, start_us, end_us, first_seq, last_seq, has_split
per phase:        events, us, tokens（tokens 只统计 decode）
per (phase, op):  total_us, count, max_us
per (phase, op) gap: total_us, count
```

- 内存量级：token 聚合上限 100000 条（超过按 first_seq 淘汰最早 token），
  输出时取最近 `max_tokens`（≤8192）条；phase/op 聚合不淘汰。
- `Clear()` 或新 session 时重置；`FASTLLM_OP_TRACE_ARCHIVE=0` 时仍会维护聚合
  （只省掉落盘，不省内存统计）。
- 实时页的 Prefill/Decode 卡、Token 趋势、Token 列表在 P3 已改为优先读该摘要（见 10.3）。

## 8. Python 读取器

### 8.1 新文件 `tools/fastllm_pytools/op_trace_archive.py`

纯标准库（`json`、`struct`、`zipfile`、`os`、`threading`），不 import `llm`/`server`。

```python
class TraceFormatError(Exception): ...

class Session:
    path: str
    meta: dict
    chunks: list[dict]       # 解析后的 chunks.idx

    @classmethod
    def open(cls, path: str, recover: bool = True, read_only: bool = False) -> "Session": ...
    def summary(self, max_tokens: int = 8192) -> dict: ...
    def page(self, from_seq: int = 0, limit: int = 8192,
             phase: str | None = None, req: int | None = None,
             token: int | None = None, op: str | None = None,
             min_dur_us: int = 0) -> dict: ...
    def find(self, phase: str | None = None, req: int | None = None,
             token: int | None = None) -> dict: ...
    def close(self) -> None: ...

def recover(path: str) -> dict: ...      # 截断索引、seal、返回 meta
def pack(path: str, output: str | None = None) -> str: ...  # 生成 .ftrace
def list_sessions(root: str) -> list[dict]: ...
```

### 8.2 打开与恢复

- 支持两种输入：
  - 目录：必须包含 `meta.json` 与 `chunks.idx`；缺失索引但有 `chunks/` 时尝试重建。
  - `.ftrace`：`zipfile` 打开，内部路径与目录一致。
- 校验 `schema`/chunk `version`；失败抛 `TraceFormatError`，HTTP 层返回 400。
- 目录模式且 `sealed=false`：
  - 属主 pid 存活（同机且 pid 存在）→ 只读，`committed_seq` 为界；
  - 否则调用 `recover()`：按 6.6 截断/修复并 mark sealed。
- 读取缓存：`chunks.idx`、`summary()`、已解码块 LRU（可选，默认缓存最近 8 块）。

### 8.3 分页

- 用 `chunks.idx` 二分定位起始块；顺序解码，行级过滤，凑满 `limit` 即返回。
- 过滤条件与页面现有语义一致：`phase/req/token/op/min_dur_us`；
  时间范围用 `from_us/to_us` 可选。
- 返回体与实时 snapshot 的 `events[]` **字段完全一致**，页面渲染代码无需改字段名。
- `next_seq = 最后一条 seq + 1`；`has_more` 由索引判断，不扫描全量。

### 8.4 摘要合并

- 全部来自 `chunks.idx` 的块摘要：
  - phase → events/us/tokens
  - (req, token) → 累加 events/us/run/xfer/sync/gap/start/end
  - (phase, op) → 累加 total/count/max，再取 top-N
- 输出结构（与 native `op_trace_summary_json` 对齐，页面共用）：

```json
{
  "schema": 1,
  "session": "...",
  "phases": {
    "prefill": {"events": 0, "us": 0, "tokens": 0},
    "decode":  {"events": 0, "us": 0, "tokens": 0},
    "mixed":   {"events": 0, "us": 0, "tokens": 0},
    "none":    {"events": 0, "us": 0, "tokens": 0}
  },
  "tokens": [
    {"req": 0, "token": 12, "first_seq": 1001, "last_seq": 5000,
     "events": 3999, "total_us": 12345, "latency_us": 15000,
     "run_us": 10000, "xfer_us": 500, "sync_us": 200, "gap_us": 1645,
     "cuda_us": 11000, "cpu_us": 1345, "start_us": 1234567890,
     "end_us": 1234582890, "has_split": true}
  ],
  "ops": [{"phase": "decode", "op": "Linear", "total_us": 123456, "count": 789, "max_us": 5000}],
  "devices": [{"dev": "cuda", "events": 100, "us": 1234}]
}
```

- `summary()` 结果按 `(mtime, committed_seq)` 缓存到进程内；
  录制中的会话每次增量合并新增块。
- 页面侧适配：把 summary 规范化为与 `buildPhaseTokenStats()` 相同字段名，
  或调整渲染函数直接读取该结构（见 10.3）。

### 8.5 打包

- `pack()` 使用 `zipfile.ZIP_DEFLATED`，zip64 默认开启；
  `chunks.idx`/`meta.json` 放包根，chunk 放 `chunks/`。
- 先写 `.tmp`，`os.replace` 原子改名；同名已存在且校验通过则跳过。
- 失败保留目录并写 `meta.archive.pack_error`。
- 打包任务串行执行（进程级 `threading.Lock`），避免同时压多个大 session。

## 9. 离线查看器

### 9.1 新文件 `tools/fastllm_pytools/op_trace_viewer.py`

- 依赖：标准库 + `fastapi`/`uvicorn`（项目已有）。
- **禁止** import `.llm`、`.server`、`fastllm_lib`；页面模板从 `op_trace_ui` 导入
  （`op_trace_ui.py` 只有字符串常量，无副作用）。
- 启动：

```bash
ftllm op-trace-view [--trace PATH] [--host 127.0.0.1] [--port 8765]
                    [--root DIR] [--no-browser] [--read-only]
```

- `cli.py` 在 `args_parser()` 之前拦截（仿照 `router` 的快速路径），
  避免导入模型相关模块；启动后 `webbrowser.open` 打开页面。
- 默认端口 8765；占用时自动 +1 重试（最多 10 次）并打印实际地址。

### 9.2 路由

```
GET  /                            -> PAGE_HTML
GET  /api/op-trace/status         -> mode/session/path/archive/packed/pack_error
GET  /api/op-trace/summary        -> Session.summary()
GET  /api/op-trace/export        -> Chrome Trace/Perfetto JSON 下载（format=chrome|perfetto）
GET  /api/op-trace/archive/page   -> Session.page(...)
GET  /api/op-trace/archive/find   -> Session.find(...)
POST /api/op-trace/open           -> 打开路径（目录/文件），返回 meta + 是否有摘要
POST /api/op-trace/close          -> 关闭当前会话（P3）
POST /api/op-trace/pick           -> 系统对话框，返回 {"path":...} 或 {"cancelled":true}
POST /api/op-trace/pack           -> 后台打包当前会话
GET  /api/op-trace/sessions       -> 扫描默认目录/--root 的会话列表
POST /api/op-trace/import?name=…  -> 上传 .ftrace 到 trace root 并校验
POST /api/op-trace/delete         -> 删除直属于 trace root 的会话目录或 .ftrace
```

- 只允许 localhost 访问（与现有 `_op_trace_allowed` 相同的判断）。
- `open` 校验：目录包含合法 `meta.json`；文件为 `.ftrace` 且包内含 `meta.json`。
- 同一时刻只允许一个已打开会话；切换时关闭旧句柄/zip 文件。
- `sessions` 返回 `kind`（`dir`/`ftrace`）、`active`、`size_bytes`、`mtime_ns` 等字段；
  已打包会话会同时出现目录和 `.ftrace` 两条记录，分别删除。
- `delete` 只接受 `viewer.root` 的直接子项：目录需有合法 `meta.json`，文件需通过
  `.ftrace` 校验；录制中的会话、`--read-only`、正在打包时拒绝。删除当前打开的
  会话会先关闭句柄。`viewer.root` 本身是单会话目录时不可删除该根目录。
- 页面 `PAGE_HTML` 无需服务端模板注入；页面启动时先请求 `/api/op-trace/status`
  判断 `mode`（11.1 接口示例）。

### 9.3 系统对话框 `tools/fastllm_pytools/op_trace_dialogs.py`

- 平台分派：
  - Windows：子进程 `powershell -NoProfile -STA -ExecutionPolicy Bypass -Command ...`，
    使用 `System.Windows.Forms.OpenFileDialog` / `FolderBrowserDialog`；
    设置 `[Console]::OutputEncoding=[Text.Encoding]::UTF8`，支持中文/空格路径。
  - macOS：`osascript` 的 `choose file` / `choose folder`，输出 POSIX path。
  - Linux：`zenity --file-selection [--directory]`，无 zenity 再试 `kdialog`。
- 失败/无桌面：返回 `unavailable`，页面显示“手动输入路径”输入框，功能不中断。
- 单飞：模块级 `threading.Lock`；子进程 300s 超时后 kill 并返回超时错误。
- 文件过滤器：`FastLLM Trace (*.ftrace)|*.ftrace|All files (*.*)|*.*`；
  目录模式不限制。
- 不把用户输入拼进 shell；路径由对话框进程输出，Python 侧只做存在性/格式校验。

### 9.4 “不加载模型”的验收

- 新增测试：启动 viewer 后断言 `"fastllm_pytools.llm" not in sys.modules`
  且 `"fastllm_pytools.server" not in sys.modules`。
- 启动耗时目标 < 2s；不分配 CUDA 上下文（`nvidia-smi` 无新增进程显存）。

### 9.5 Chrome Trace / Perfetto 导出

- `Session.export_chrome_trace(path)` 逐块流式写 JSON，不在内存中拼接全部事件；
  输出对象包含 `traceEvents`、`displayTimeUnit:"ms"` 和会话 `metadata`。
- 每个算子输出一条 `ph:"X"` 完整事件：`name=op`、`cat=phase`（gap 事件为
  `cat:"gap"`）、`ts=t_us - from_us`、`dur=dur_us`、`pid=1`、`tid=线程 id`；
  `args` 保留 seq/phase/layer/req/token/dev/weight/io/dur_ns/分项耗时。
- 进程名/线程名用 `ph:"M"` 元数据事件输出；时间归一化到会话最早事件。
- Viewer `GET /api/op-trace/export?format=chrome|perfetto` 生成临时文件后通过
  `FileResponse` 下载，响应结束清理临时目录；页面“导出 Trace”按钮仅 viewer 可用。
- 当前导出的是会话可见范围：已封会话为全量，录制中会话为 `committed_seq` 之前。

### 9.6 纯静态 HTML（`op_trace_static.py`）

- `ftllm op-trace-static --output op_trace_viewer.html` 生成单文件 HTML；
  不依赖 server、native 或模型，可直接用浏览器打开。
- 页面复用 `PAGE_HTML`，插入本地 API shim；`window.fetch` 只拦截
  `/api/op-trace/*`，状态/摘要/分页/find 全部由浏览器内 `StaticSession` 实现。
- 文件选择只接受 `.ftrace`：浏览器解析 ZIP central directory（含 ZIP64）、
  `DecompressionStream("deflate-raw")` 解压，再按 chunk 格式解码。
- 支持摘要卡片、时间线、事件表、阶段过滤、Token 列表与跳转、分页；
  数据始终留在本机，不上传。
- 浏览器内的 `tid`/纳秒级时间超过 JavaScript 安全整数时按 Number 近似；
  只影响线程显示名/亚微秒精度，不影响事件与分页。

当前限制：

- 只支持单个 `.ftrace` 文件，不支持目录、实时流或 URL 重建；
- 刷新页面后需重新选择文件；
- 不支持打包/导出/清理/导入等需要服务端的操作（按钮隐藏）；
- 首次摘要对旧索引会逐块解码，大文件打开时间随事件数增加。

### 9.7 会话管理与手动删除

- Viewer 顶栏新增“会话管理”，展开归档根目录下的会话列表；显示会话 ID、
  类型（目录/单文件）、实际占用、时间、录制/封包状态，支持打开和删除。
- 保留策略仍为永久保留：没有后台轮转/自动删除；磁盘不足只会停止采集。
  删除操作必须由用户确认，且只删除用户点击的那一条目录或 `.ftrace`。
- 目录和对应 `.ftrace` 是两条独立记录；页面提示两者分别删除，避免误删。
- 录制中的目录（`sealed=false` 且 owner 存活）禁用删除；后端再次校验。
- 服务端 `list_sessions()` 增加 `kind/active/size_bytes/mtime_ns`；
  文件大小按实际目录递归统计，不依赖 `meta.archive.bytes`。
- 删除当前打开的会话时，后端先关闭 `Session`/zip 句柄再删除，前端清空 viewer 状态。
- 纯静态 HTML 隐藏“会话管理”和删除入口，不提供文件系统写操作。

### 9.8 上传导入 `.ftrace`

- Viewer 顶栏“导入 `.ftrace`”打开浏览器文件选择；文件以
  `application/octet-stream` 原始 body POST 到 `/api/op-trace/import?name=…`，
  不经云端。
- 服务端流式写入 `viewer.root` 下的临时文件（不把整包读进内存），校验
  zip/chunk 索引/schema 后原子移动到根目录；成功返回 `path/session/size_bytes`，
  页面随即打开导入文件。
- 文件名只保留 basename，过滤路径分隔符和特殊字符；同名自动追加 `-1`、`-2`。
  目标已存在时使用独占创建/硬链接，绝不覆盖已有会话。
- `--read-only` 拒绝导入；空文件、非 `.ftrace`、损坏 zip/索引返回 400；
  临时文件在失败/中断时清理。
- 导入只写本机 `viewer.root`，不调用模型、不访问网络、不做自动上传。
- 纯静态 HTML 隐藏导入入口（静态页已有浏览器本地文件解析）。

## 10. 页面改造（`op_trace_ui.py`）

### 10.1 模式与状态

新增状态：

```js
state.source = "live";          // live | viewer
state.session = null;           // viewer: {id, path, meta, summary, sealed}
state.archive = null;           // {fromSeq, toSeq, nextSeq, hasMore, range}
state.summary = null;           // 全 session 聚合（两种模式都用）
```

启动流程：

1. `GET /api/op-trace/status`（viewer 新增）；失败则回退 live。
2. `mode === "viewer"` → 隐藏“启用采集/清空/暂停/跟随”，显示“打开数据”。
3. `mode === "live"` → 保持现有 `poll()`，但同时拉取 `/api/op-trace/summary`，
   并每 2s 刷新一次摘要卡片。

### 10.2 打开流程（viewer）

```
点击“选择文件”/“选择目录”
  → POST /api/op-trace/pick {mode}
  → 取消：无操作
  → 失败：显示手动路径输入
  → 成功：POST /api/op-trace/open {path}
  → 成功：切换到离线视图，写入 location.hash = "#trace=<encodeURIComponent(path)>"
  → 自动加载：summary + 第一页（优先定位到请求的 prefill 起点）
```

支持 URL 恢复：页面加载时解析 `#trace=`，viewer 模式下自动打开；
`#trace=...&phase=prefill&token=42&seq=...` 用于分享定位。

### 10.3 摘要驱动卡片（修复 prefill 消失）

- `renderPhaseTokens()`、`renderTokenList()`、`renderTimeSplit()`、
  Token 趋势改为：优先使用 `state.summary`，没有摘要再回退到
  `buildPhaseTokenStats()`（兼容旧数据/旧 server）。
- 摘要字段名统一为 8.4 的结构；渲染层做一次规范化，避免改动所有调用点。
- Prefill 卡显示全 session 累计与事件数；点击行为：
  - live：设置 `phase=prefill`，若有归档则跳归档；否则保持现有滚动定位。
  - viewer：`GET /archive/find?phase=prefill` 得到 `first_seq/last_seq`，
    加载该区间第一页，设置表格“上一页/下一页”的游标。

### 10.4 离线浏览与分页

- `state.events` 在离线下表示**当前页**（替换而非追加）；
  `computeView()` 使用当前页的 `start/end`，不自动 follow。
- 事件表底部增加分页条：`上一页 / 下一页 / 跳到 seq`，显示
  `第 x–y 条 / 共 N 条（会话）`；`limit` 默认 8192。
- Token 点击：加载该 token 的 `first_seq..last_seq`（通常 1-2 页）。
- 时间线顶部显示“内存 buffer 边界”标记（live）；离线显示“归档区间”。
- 空页提示区分：
  - “当前页没有匹配事件，试试上一页/下一页”；
  - “会话中不存在该筛选条件的事件”（由 summary/index 判断）。
- 由于 `state.events` 有界，原 `MAX_EVENTS/KEEP_US` 只保留 live 路径使用；
  离线路径不再裁剪。

### 10.5 实时模式细节

- `trimEvents()` 的 `MAX_EVENTS` 改为读取服务端 `capacity`（上限仍可配），
  避免“服务端还有、页面已经删了”。
- `poll()` 的 `lost` 分支改为：保留已有事件，从 `first_seq` 重新同步并去重合并，
  不再整体清空（归档模式下无影响，但实时页体验更好）。
- 采集关闭后显示 `.ftrace` 打包状态（`meta.archive.packed/pack_error`）。

## 11. HTTP 接口规格

### 11.1 `GET /api/op-trace/status`

```json
// live server
{"supported": true, "enabled": true, "mode": "live"}
// viewer
{"supported": true, "enabled": false, "mode": "viewer", "session": null}
```

### 11.2 `GET /api/op-trace/archive/page`

```
?from_seq=1001&limit=4096
&phase=prefill|decode|mixed|none
&req=0&token=42&op=Linear&min_dur_us=100
&from_us=&to_us=
```

```json
{
  "schema": 1,
  "session": "20261119-142341-12345",
  "range": {"from_us": 1234567890, "to_us": 1234644433},
  "events": [
    {"seq": 1001, "t_us": 1234567890, "dur_us": 123, "tid": 1,
     "layer": 0, "phase": "prefill", "req": 0, "token": -1,
     "op": "Linear", "dev": "cuda", "weight": "...", "io": "...",
     "xfer_us": 0, "reshape_us": 0, "run_us": 0, "sync_us": 0}
  ],
  "next_seq": 2000,
  "has_more": true,
  "total_events": 123456
}
```

`events` 字段与现有 snapshot 完全一致（`gap` 用 `"gap":1` 表示）。

### 11.3 `GET /api/op-trace/archive/find`

```
?phase=prefill&req=0            -> {"first_seq":1,"last_seq":4096,"t0_us":...,"t1_us":...}
?req=0&token=42                 -> {"first_seq":...,"last_seq":...,"t0_us":...,"t1_us":...}
```

### 11.4 `POST /api/op-trace/pick`

```json
{"mode": "file"}   // 或 "dir"
```

成功：`{"path": "E:\\traces\\x.ftrace"}`
取消：`{"cancelled": true}`
无桌面：HTTP 409 `{"error": "dialog_unavailable", "hint": "请手动输入路径"}`

### 11.5 `POST /api/op-trace/open`

```json
{"path": "E:\\traces\\x.ftrace"}
```

成功返回：`{"session": "...", "meta": {...}, "sealed": true, "packed": true}`
失败：HTTP 400 `{"error": "invalid_trace"}`。

## 12. 配置项

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `FASTLLM_OP_TRACE` | 空（关闭） | 采集总开关（沿用） |
| `FASTLLM_OP_TRACE_CAPACITY` | 65536 | 内存 ring 容量（沿用） |
| `FASTLLM_OP_TRACE_PREFILL_PIN` | 131072 | prefill 固定保留条数；0 关闭（P0 已实现） |
| `FASTLLM_OP_TRACE_GAPS` | 1 | 空隙事件（沿用） |
| `FASTLLM_OP_TRACE_GAP_US` | 20 | 空隙最小粒度（沿用） |
| `FASTLLM_OP_TRACE_IO` | 1 | 记录输入输出摘要（沿用） |
| `FASTLLM_OP_TRACE_DIR` | `<cwd>/.fastllm/op_trace` | 归档根目录（新增） |
| `FASTLLM_OP_TRACE_ARCHIVE` | 1 | 是否落盘归档（新增） |
| `FASTLLM_OP_TRACE_CHUNK_EVENTS` | 4096 | 每块事件数（新增） |
| `FASTLLM_OP_TRACE_FLUSH_MS` | 200 | 写线程 flush 周期（新增） |
| `FASTLLM_OP_TRACE_SESSION_MAX_MB` | 0 | 0=不限；>0 达到即停止归档并标记 truncated（新增） |
| `FASTLLM_OP_TRACE_MIN_FREE_MB` | 2048 | 磁盘余量阈值，低于则停止归档（新增） |
| `FASTLLM_OP_TRACE_PACKAGE_ON_SEAL` | 1 | 封会话后自动打包（新增） |
| `FASTLLM_OP_TRACE_DELETE_DIR_AFTER_PACK` | 0 | 打包后删源目录（新增） |

注意：`FASTLLM_OP_TRACE_*` 是 native 启动时读取；新增项在 `Store` 构造时解析一次，
运行期只通过 UI 开关控制采集。`FASTLLM_OP_TRACE_DIR` 由 Python 侧也需要读取
（server/viewer 打包、扫描），两边保持同一默认值算法并在启动日志打印实际路径。

P1 实际限制：`CHUNK_EVENTS` 钳制到 64..65536，`FLUSH_MS` 钳制到 50..60000，
`pendingChunks` 钳制到 1..64（默认 8）；实时 token 聚合上限 100000 条、单次 summary
最多返回 8192 条，这些上限目前不是环境变量。

## 13. 分阶段实施

### P0（可选快修，建议先做）——已完成

目标：不改架构先缓解“decode 后 prefill 消失”。

实现（2026-11-19）：

1. `Store` 新增 `pinned`（`std::deque<Event>`）固定保留队列。环形缓冲淘汰时，
   若队首是 `phase == "prefill"` 且 `pinned` 未满，则移入 `pinned` 而不是丢弃；
   容量由 `FASTLLM_OP_TRACE_PREFILL_PIN` 控制（默认 131072，0 表示关闭）。
2. `SnapshotJson` 合并输出 `pinned + ring`，事件仍按 seq 递增；新增字段
   `pinned`（当前条数）与 `pinned_capacity`。`lost` 改为扫描本次返回序列
   （含与 `after_seq` 的衔接）的 seq 缺口，适配 pinned 造成的稀疏前缀。
3. 页面 `trimEvents()` 改为分阶段裁剪：prefill 按 `pinned_capacity` 固定保留，
   其余事件按 `max(65536, 服务端 capacity)` 保留；`poll()` 遇到 `lost` 只计数，
   不再清空已加载事件。页头新增 `pinned` 统计，`dropped`/`loaded` 提示补充说明。
4. 验收：
   - MSVC 冒烟测试（临时用例，不在仓库内）：ring=64、pin=20、20 prefill + 100 decode
     后快照返回 84 条（20 pinned + 64 ring），分页跨 pinned/ring 正确，`lost` 判定正确。
   - `build_tools.bat cpu/gpu` 均编译通过；CPU 产物实测 snapshot 返回
     `pinned=0 / pinned_capacity=131072`。
   - 运行期（待人工确认，需带模型跑长 decode）：长 decode 后页面按
     `phase=prefill` 仍能看到事件；`pinned` 计数值随淘汰增长；内存增长约
     `pin 条数 × sizeof(Event)=240B + 字符串`（默认上限约 60-100MB）。

注意：P0 只解决“实时页仍能看到 prefill”，事件仍受内存上限约束；
“任意历史全量浏览”由 P1-P3 的磁盘归档解决。

### P1 采集落盘——已完成（2026-11-19）

实现：

1. `include/utils/op_trace_types.h` 抽出 `Event`/`Timing`/phase 映射/JSON/env 辅助；
   `include/utils/op_trace_archive.h` 只放声明；`src/utils/op_trace_archive.cpp`
   实现写线程、chunk 编码、`chunks.idx`、`meta.json`、磁盘/会话上限与 drop 计数，
   并已加入 `FASTLLM_CXX_SOURCES`。
2. `op_trace.h` 集成：`Store::Record` 在持有 Store 锁时调用 `ArchiveAppend` +
   `AggregateRecord`；`SetEnabled(true)` 确保 session、`SetEnabled(false)` flush/seal、
   `Clear()` 封旧 session 并新建；`SnapshotJson` 增加 `session_id/archive`。
3. `pytools.cpp` 导出 `op_trace_flush`、`op_trace_session_json`、
   `op_trace_summary_json(int)`、`op_trace_archive_shutdown`、`set_op_trace_model`；
   `llm.py` 绑定；`server.py` 启动时写入模型名。
4. 实时聚合同步实现（见 7.5）；`Clear`/新 session 重置，token 上限 100000。

验收（本机实测）：

- 全量采集：合成 20 prefill + 1000 decode（另有少量 gap 事件，多次运行共
  1028–1031 个 `Record`）→ `committed_seq = last_seq`、`archive.dropped = 0`、
  3 个 chunk、summary `tokens_total=1000`；Python 参考解码器逐列解析全部 chunk，
  seq 连续、字符串/列长度/索引行全部校验通过。小规模 20+100 的用例同样通过
  （122/123 事件）。
- 强杀进程（`Stop-Process -Force`，约 1.8s、233 事件）：29 个已 rename 的 chunk
  可完整解码，`chunks.idx` 行完整；`meta.json` 的 `committed_seq/chunks` 可能
  落后最多 1s（设计如此），P2 恢复时以 `chunks.idx` 为准重写。
- 会话上限：`SESSION_MAX_MB=1` 时停止于 959181 字节 / 9 块，状态 `truncated`，
  剩余 63136 事件计入 `archive.dropped`；低磁盘（阈值大于可用空间）时
  状态 `stopped`、`stopped_reason=low_disk`；两种情况 `SetEnabled(false)` 均写出
  `sealed=true`。
- `FASTLLM_OP_TRACE_ARCHIVE=0`：不创建目录/文件，snapshot 为
  `session_id=""`、`archive.enabled=false`、`state=idle`；实时聚合仍工作（见 7.5）。
- `Clear()`：两个 session 目录均 sealed；同秒同 pid 的目录名自动加 `-1` 后缀。
- 开销（200k 合成事件，本机）：归档关闭约 1.39–1.75 µs/事件，开启约
  2.36–2.55 µs/事件，增量约 0.6–0.8 µs/事件；真实模型单 token 对比尚未补测，
  属后续性能验证。
- `build_tools.bat cpu/gpu` 均通过；运行副本已刷新。

已知偏差/限制：

- chunk 的 `devId` 从规划中的 u8 改为 u16（StringTable 下标可能因大量唯一 io
  字符串超过 255）；chunk `version` 仍为 1（P1/P2/P3 未改变格式，读取器按版本校验）。
- `meta.json` 每 1s 更新；强杀后它可能滞后，`chunks.idx` 是恢复的权威来源。
- `FASTLLM_OP_TRACE_ARCHIVE=0` 仍会执行实时聚合，且 snapshot 多出
  `session_id/archive` 字段；页面忽略这些字段。

### P2 读取器 + Viewer + 系统对话框——已完成（2026-11-19）

实现：

1. 新增 `tools/fastllm_pytools/op_trace_archive.py`（纯标准库）：
   - 目录与 `.ftrace` 两种输入；`Session.open/summary/page/find/range/close`，
     `recover`、`pack`、`list_sessions`、`default_archive_root`。
   - 目录未 sealed 时按属主 pid 判断：存活则只读并限制到 `committed_seq`；
     属主不存活则走恢复。恢复以已提交 `chunks.idx` 为权威：忽略 `.tmp`/
     未入索引的块，按文件名与 seq 连续性校验 header/eventCount/first/last，
     遇到第一个断点即截断并重写索引/meta；索引缺失时才扫描 `chunks/` 重建。
   - 分页按索引定位，跨块解码，支持 `phase/req/token/op/min_dur_us/from_us/to_us`；
     事件字段与实时 snapshot 对齐（含 `dur_ns`，`gap` 仅在空隙事件出现）。
   - 摘要合并块索引；新索引包含 `phase_us`/`gaps`/`devices`/`other_op_us`
     时走快速路径，旧 P1 索引缺少这些字段时一次性解码块回退，结果一致。
   - 打包：`ZIP_DEFLATED` + `allowZip64`，先 `.tmp` 再 `os.replace`；同名有效
     `.ftrace` 跳过；打包后写 `archive.packed/ftrace/sha256/pack_error`。
2. 新增 `tools/fastllm_pytools/op_trace_viewer.py`：
   - FastAPI viewer，仅 localhost 门禁；`/`、`/api/op-trace/status|summary|snapshot`、
     `/api/op-trace/archive/page|find`、`/api/op-trace/open|pick|pack|sessions`。
   - 不 import `.llm`/`.server`/native；页面 shell 复用 `PAGE_HTML`；端口被占用时
     自动 +1（最多 10 次），默认 8765；`--no-browser/--read-only/--root/--trace`。
   - 同一时刻仅一个打开会话；pack 后台线程串行执行并回写状态。
3. 新增 `tools/fastllm_pytools/op_trace_dialogs.py`：
   - Windows PowerShell WinForms、macOS osascript、Linux zenity/kdialog；
     模块级单飞锁、300s 子进程超时、取消/无桌面/忙状态结构化返回；
     输出按 UTF-8 解码，支持中文/空格路径。
4. `cli.py` 在 `args_parser()` 前拦截 `op-trace-view`，不加载模型依赖；帮助中
   注册同名子命令；启动后定时打开浏览器。
5. `src/utils/op_trace_archive.cpp` 的块索引向后兼容扩展：`phase_us`、`gaps`、
   `devices`、token `other_op_us`；chunk 版本仍为 1，旧索引由读取器回退扫描。

验收（本机）：

- 新增 `test/api/test_op_trace_archive.py` 当前 29 项全部通过（P2 初次 21 项，P4 导出补 2 项、静态 HTML 补 2 项、会话删除补 2 项、上传导入补 2 项）：合成 3 chunk 归档
  （prefill/decode/mixed/none、gap、空 io、layer/token=-1、跨块 token）、恢复
  （截断块/删索引行/`.tmp`/索引缺失/未 sealed）、分页过滤、find、摘要与逐事件
  扫描对拍、目录/`.ftrace` 打包读取、viewer 路由/localhost/后台打包/沙箱 import、
  对话框超时/忙/无桌面/中文路径、CLI 快速路径。
- 1M 事件合成索引会话：首次 `/summary` 约 37ms，命中缓存约 0.1ms（目标 <200ms）。
- 8192 事件单页首次约 41ms（`Session.open` 预取前 2 块），满足 <100ms；
  未命中块缓存时单块解码约 150ms，因此大页/随机跳页会有一次性解码开销。
- 打开 P1 native 产物（1028 事件）的 `summary/page/find` 通过；对副本执行恢复后
  3 块/1028 seq 不变，摘要与恢复前一致；`.ftrace` 打包后仅凭单文件可读。
- `ftllm op-trace-view` 实测启动后 `/api/op-trace/status|open|summary|archive/page`
  返回正常；进程内无 `fastllm_pytools.llm`/`.server` 导入。
- Windows PowerShell UTF-8 输出中文/空格路径验证通过；实际 GUI 对话框需人工点击。
- `build_tools.bat cpu/gpu` 均通过，运行副本已刷新。

已知边界：

- P2 的离线 API/页面 shell 已在 P3 接入页面交互；P2 读取器本身不依赖 native。
- `.ftrace` 打包在 P2 由 API/`pack()` 触发；server 自动打包与启动补齐已在 P3 完成。
- 摘要快速路径的 `ops` 来自每块 top 32，跨块全局 top-N 在极端碎片化算子上是近似；
  token/phase/device 统计为精确合并。

### P3 页面离线模式 + 自动打包——已完成（2026-11-19）

实现：

1. `op_trace_ui.py` 页面按 `/api/op-trace/status` 的 `mode` 切换：
   - live 保留原实时轮询，同时每 2s 拉 `/api/op-trace/summary`； viewer 隐藏
     enable/clear/follow/pause，显示打开文件/目录、路径回退、打包、关闭与归档状态。
   - viewer 打开后加载 summary + 第一页；`#trace=...&phase=...&token=...&seq=...`
     恢复路径与定位；翻页/阶段/token 点击使用 `history.replaceState` 更新 hash。
   - summary 规范化为 `buildPhaseTokenStats()` 结构，卡片、Token 趋势与 Token 列表
     优先读全 session 摘要，旧 server 无摘要时回退当前事件聚合。
   - 事件表离线模式替换为当前页（不再 `slice(-300)`），底部分页条支持
     上一页/下一页/跳到 seq；`from_seq` 游标保证不丢不重，筛选页保留 base seq。
   - Prefill 卡点击调 `/archive/find?phase=prefill`，从 `first_seq` 加载；
     Token 点击用 summary 的 `first_seq/last_seq` 或 find 加载该 token 全部事件。
   - `.ftrace` 单文件与目录走同一套 API；只读打包按钮禁用并显示原因。
2. `server.py` 新增后台打包队列：
   - `/api/op-trace/enabled` 关闭时先由 native flush/seal，再取 `dir` 投递打包；
     `/api/op-trace/clear` 在 rotate 前记录旧目录，rotate 后投递旧会话。
   - uvicorn 启动后后台扫描默认 trace root：跳过仍存活的未封会话；对已封未打包
     或属主已失效的会话调用 `pack()`（其内部会先 recover）。
   - 打包串行单 worker，按目录去重；失败写 `meta.archive.pack_error`，不阻塞请求。
   - `/api/op-trace/status` 返回 live `mode/session/path/archive/packed/pack_error`，
     合并磁盘 `meta.json` 的打包状态；新增 `/api/op-trace/summary`。
3. `op_trace_viewer.py` 补齐 status 的 `archive` 字段与 `/api/op-trace/close`；
   viewer 打包后页面轮询能看到 `packed/pack_error`。
4. 实时 `appendSnapshot()` 对乱序/重复 seq 做 Map 合并并重排，`afterSeq` 取本批最后
   seq；`trimEvents()` 使用服务端 capacity；解决 lost 重同步重复行与页面容量错位。

验收（本机）：

- `test/api/test_op_trace_archive.py` 29 项、`test_op_trace_server.py` 5 项通过：
  viewer 归档 status/close、禁用接口投递打包、启动维护跳过存活属主、打包队列恢复死会话。
- Edge headless + CDP 实测 viewer：
  - 1028 事件会话打开后显示 1028 行；点击 Prefill 后 20 行、阶段 active、hash 更新；
    点击 token 后只剩该 token 的 1 条事件，token chip 正确。
  - 12000 事件会话第一页 `1–8192`，下一页 `8193–12000`，上一页回到 `1–8192`；
    分页按钮 disabled 状态与 hasMore 一致。
  - 10500 条 prefill 会话：Prefill 第一页 8192 条，下一页 2308 条，上一页返回。
  - 打包按钮从“未打包”变为“已打包”并禁用；仅复制 `.ftrace` 到空目录后仍显示
    1028 事件。
  - 空 viewer 显示“打开文件/目录”、未打开状态；live 假接口显示实时控件、摘要卡与事件表。
  - live 页面 `pageBar` 隐藏、viewer 控件隐藏；分页/打开控件互斥。
- 运行 `build_tools.bat cpu/gpu` 后刷新 `build-vs-*/tools/ftllm` 副本（最终验证
  见提交前构建记录）。

已知边界：

- live 页 Prefill 卡仍优先在实时 buffer/pinned 中定位；归档跳页只在 viewer 中进行。
- viewer 每页 8192 条；表格一次渲染当前页全部行，1M 级会话请用分页而非“全部”。
- summary 的 Token 列表默认最近 8192 条（`tokens_truncated` 标记）；更早 token 需
  通过归档分页/seq 定位。
- 自动打包的 server 端到端（真实模型）仍需按 14.3 手工验证；单测已用 mock native
  覆盖禁用接口与启动扫描。

### P4（导出、静态 HTML、会话管理、导入已完成）

已落地（2026-11-19）：

- Chrome Trace/Perfetto 导出：`Session.export_chrome_trace` 流式写 JSON；
  viewer `GET /api/op-trace/export?format=chrome|perfetto`；页面“导出 Trace”。
- 导出按块流式写出，支持目录和 `.ftrace`；当前只导出可见范围（录制中会话到
  `committed_seq`）。
- 本机 12k 事件会话导出 3,101,273 字节 JSON，事件/元数据计数一致；Edge headless
  点击页面“导出 Trace”得到正确文件名和成功状态。
- 纯静态单 HTML：`ftllm op-trace-static` 生成 `op_trace_viewer.html`；
  浏览器内 ZIP/ZIP64 + deflate-raw 解压、chunk 解码、summary/page/find，
  复用现有页面，不加载 server/native/model。
- 静态查看器实测：1028 事件 `.ftrace` 显示 1028 行；12k 会话分页
  `1–8192`/`8193–12000`；10500 prefill 分页 `1–8192`/`8193–10500`；
  Token 点击过滤到 1 条；全程 file://，无网络服务。
- 全量测试 35 项通过（归档 29、server 5、CLI 1）；CPU/GPU 构建及运行时副本
  `op_trace_static.py` 哈希一致。
- 会话管理：viewer 顶栏“会话管理”列出归档目录/`.ftrace`、实际大小和状态；
  支持打开、手动删除。录制中、只读模式、打包中拒绝删除；删除当前会话先关闭句柄。
- 删除安全边界：只允许 trace root 直接子项，目录校验 `meta.json`/schema，
  文件校验 `.ftrace`，拒绝越界路径和归档根目录本身。
- 页面实测：4 条目录/文件记录正确分列；打开目录后可从列表删除当前会话并清空
  viewer；录制中（owner 存活）删除按钮禁用；UI 删除 `.ftrace` 后列表刷新。
- 上传导入：viewer 顶栏“导入 `.ftrace`”选择浏览器文件，原始字节 POST 到
  `/api/op-trace/import`；服务端流式落临时文件、校验后原子移到 trace root，
  页面自动打开。文件名清理、同名加后缀，绝不覆盖已有文件。
- 导入实测：20KB `.ftrace` 从浏览器导入到空 root 后自动打开，归档列表出现
  单文件记录；只读模式、空文件、损坏文件均拒绝；静态 HTML 隐藏导入入口。

## 14. 测试计划

### 14.1 单元测试（`test/api/test_op_trace_archive.py`，P2/P3/P4 已实现，29 项通过）

- 合成归档：构造 3 个 chunk，覆盖 prefill/decode/mixed/none、gap、空 io、
  layer=-1、token=-1、跨块 token。
- 分页边界：`from_seq=0`、跨块、`limit` 大于剩余、过滤后无结果、时间范围。
- `find`：phase/prefill、req+token、token-only、缺失时返回空。
- 恢复：截断最后一个块、删除索引行、`.tmp` 残留、索引缺失重建、
  `sealed=false` 属主存活只读/属主失效封存。
- 打包：zip64 打包后目录/文件读取一致；`.tmp` 不进入包；sha256 回写。
- 摘要合并：新索引快速路径与逐事件扫描结果一致；1M 索引会话首算约 37ms。
- Chrome Trace/Perfetto 导出：13 事件/3 tid 的时间归一化、gap 分类、args/metadata、
  HTTP 下载头、format 校验与无会话错误。
- 纯静态 HTML：`build_static_html` 包含本地解析/shim，CLI 快速路径生成文件且不导入
  `llm`/`server`；浏览器手工验证 1028/12k/10500 事件、分页、阶段与 Token 跳转。
- Viewer/dialog/CLI：localhost 门禁、打开/分页/summary/pack、沙箱无
  `llm`/`server` 导入、中文/空格路径、取消/超时/忙/无桌面。
- P3 自动打包：`test/api/test_op_trace_server.py` 5 项，覆盖禁用/清空接口投递、
  启动维护跳过存活属主、队列恢复未封死会话、status/summary 路由。
- P3 页面：Edge headless/CDP 手工脚本验证打开、Prefill/Token 跳转、跨页、
  hash、打包按钮与 only-`.ftrace` 浏览；脚本不进入仓库。

### 14.2 集成测试（可选，需要 native 构建）

- 构造 `SetEnabled(true)` → 生成事件（可用测试钩子或最小模型）→ `Flush(true)`
  → Python 读取，事件数/字段与 native 计数一致。
- schema 版本不匹配时报错清晰。

### 14.3 手工验证

1. `scripts/windows/build_tools.bat gpu`。
2. 启动 server，打开 `/debug/op-trace`，启用采集。
3. 发一个长 prompt（触发 chunked prefill）+ 长输出（数百 token）。
4. 观察页头 `dropped`、`archive.state`；关闭采集，等待 `.ftrace` 出现。
5. `ftllm op-trace-view` → 系统对话框选文件 → 验证：
   prefill 全量、任意 token、跨页翻页、summary 卡、URL 恢复。
6. 拷贝 `.ftrace` 到另一台机器，仅启动 viewer 再验证。
7. `ftllm op-trace-static --output op_trace_viewer.html`，直接打开 HTML，
   选择同一 `.ftrace`，验证无服务端时 summary/分页/Token 跳转。
8. 打开“会话管理”，验证目录/`.ftrace` 大小与状态；录制中会话删除按钮禁用；
   删除当前会话后页面清空；删除 `.ftrace` 后文件消失。
9. 顶栏“导入 `.ftrace`”上传一个文件，确认写入 `--root`、自动打开且列表刷新；
   重复同名导入得到 `-1` 后缀，原文件不被覆盖。

### 14.4 性能与回归

- 开关归档对比：tokens/s、单 token 平均耗时、`events/s`。
- 磁盘：记录每小时增量，验证与事件率线性。
- 页面：1M 事件会话加载首屏与翻页耗时；无内存持续增长。
- 回归：`FASTLLM_OP_TRACE` 关闭时无文件创建、无额外线程。

### 14.5 Ubuntu + `ftllm` 验收

适用于 Ubuntu 上通过 `ftllm` 启动 server，且浏览器不一定运行在 server 本机。

1. **版本确认**：`which ftllm`、`ftllm --version`；执行
   `ftllm op-trace-view --help` 和 `ftllm op-trace-static --help`，确认不是旧安装包。
   源码运行时确认 `PYTHONPATH` 指向本次 Linux 构建产物。
2. **自动测试**（源码目录）：

```bash
cd /path/to/fastllm
export PYTHONPATH="$PWD/tools"
python -m unittest test.api.test_op_trace_archive \
  test.api.test_op_trace_server test.api.test_cli_webui
```

3. **采集**：

```bash
export FASTLLM_OP_TRACE=1
export FASTLLM_OP_TRACE_ARCHIVE=1
export FASTLLM_OP_TRACE_DIR=/data/fastllm-traces
mkdir -p "$FASTLLM_OP_TRACE_DIR"
ftllm server /data/models/your-model --auto --host 127.0.0.1 --port 8080
```

4. **远程浏览器**：server 绑定 `127.0.0.1`，通过 SSH 隧道访问，避免把
   `/debug/op-trace` 暴露到公网：

```bash
ssh -N -L 8080:127.0.0.1:8080 user@ubuntu-server
ssh -N -L 8765:127.0.0.1:8765 user@ubuntu-server
```

   本机打开 `http://127.0.0.1:8080/debug/op-trace`，启用采集，发长 prompt，
   输出数百 token，关闭采集后检查：

```bash
ls -lh /data/fastllm-traces
latest=$(ls -td /data/fastllm-traces/*/ | head -1)
ls -lh "$latest" "$latest.ftrace"
python -m json.tool "$latest/meta.json"
```

   通过标准：目录和 `.ftrace` 都存在，`sealed=true`、`archive.packed=true`、
   `pack_error` 为空；`kill -9` 只允许丢失最后一个 flush 周期。
5. **Viewer（无桌面）**：

```bash
ftllm op-trace-view --root /data/fastllm-traces \
  --no-browser --port 8765
```

   本机打开隧道后的 `http://127.0.0.1:8765/`。无 DISPLAY 时系统文件对话框不可用，
   这是预期行为；使用 `--trace /path/to/file.ftrace` 或页面路径输入/会话管理打开。
   验证 Prefill/Decode、Token 跳转、分页、URL 恢复、会话删除和导入。
6. **静态 HTML**：

```bash
ftllm op-trace-static --output /data/fastllm-traces/op_trace_viewer.html
```

   将 HTML 和 `.ftrace` 拷贝到本机浏览器验收；静态页只接受 `.ftrace`，
   不提供导入/删除/导出服务端操作。
7. **写权限与只读**：确认 server/viewer 用户对
   `FASTLLM_OP_TRACE_DIR` 有写权限；只读验收启动
   `ftllm op-trace-view --root ... --read-only --no-browser`，确认打包、
   导入、删除均不可用。
8. **磁盘**：`df -h` 确认 `FASTLLM_OP_TRACE_DIR` 所在分区余量；
   低于 `FASTLLM_OP_TRACE_MIN_FREE_MB` 时应收敛为 `state=stopped`，
   而不是静默删除历史。

## 15. 容量与性能预算

- 单个 decode token 事件数 ≈ 层数 × 每层算子数 + gap，典型 4000-8000。
- 事件体积（varint + 字符串字典）约 15-25B；`io` 关闭时取下限。
- 80k events/s → 约 1.2-2MB/s → 约 70-120MB/min → 4-7GB/h。
  **全量采集需要提前规划磁盘**；这是明确需求，因此默认不启用轮转删除。
- 写线程：4096 事件/块 ≈ 60-80KB；20 块/s；顺序写 + rename，磁盘压力可控。
- 热路径开销：`Record` 多一次 vector push（现有 mutex 内），无额外系统调用；
  序列化完全在写线程。
- 内存：ring（默认 65536 × ~250B ≈ 16MB）+ pending 队列（≈8MB）+ 摘要（几 MB）。

## 16. 风险与对策

| 风险 | 对策 |
|---|---|
| 强杀/断电丢尾部 | 200ms flush + `.tmp` 协议 + 启动恢复；文档写明最坏丢 200ms |
| 磁盘被写满 | `MIN_FREE_MB` 阈值停止归档并告警；不删历史；UI 显示 stopped |
| Windows 文件被占用/杀软扫描导致 rename 失败 | rename 重试 3 次 + 退避；失败保留 `.tmp` 并计数 |
| 中文/空格/长路径 | PowerShell UTF-8 输出；目录名 ASCII；层级浅；Python 侧使用 `os.fspath` |
| `.ftrace` > 4GB | `allowZip64=True`；打包前检查磁盘余量 |
| 摘要过大（10 万 token） | summary JSON 可 gzip 传输；DB 化留到后续；页面只取 top-N token |
| schema 演进 | `meta.schema` + chunk `version` 双校验；读取器显式拒绝未知版本 |
| 多进程写同一目录 | session 目录带 pid；无共享索引文件；列表只读扫描 |
| viewer 被远程访问 | 默认绑定 127.0.0.1 + 请求 host 校验；不提供 `--allow-remote` 之外的开关 |
| 对话框阻塞请求 | 单飞 + 300s 超时；失败回退手动路径 |
| 实时页仍可能被 ring 淘汰 | 卡片/趋势/Token 列表改读 summary；原始事件只是“最近窗口” |
| CUDA Graph replay 无逐算子事件 | 保持现状并在页面提示；不做伪事件 |

## 17. 变更文件清单

P0/P1 已落地：

| 类型 | 文件 |
|---|---|
| 新增 | `include/utils/op_trace_types.h` |
| 新增 | `include/utils/op_trace_archive.h` |
| 新增 | `src/utils/op_trace_archive.cpp` |
| 修改 | `include/utils/op_trace.h` |
| 修改 | `CMakeLists.txt`（`FASTLLM_CXX_SOURCES`） |
| 修改 | `tools/src/pytools.cpp` |
| 修改 | `tools/fastllm_pytools/llm.py` |
| 修改 | `tools/fastllm_pytools/server.py` |
| 修改 | `tools/fastllm_pytools/op_trace_ui.py` |
| 文档 | `docs/op-trace-session.md`（本文件） |

P2 已落地：

| 类型 | 文件 |
|---|---|
| 新增 | `tools/fastllm_pytools/op_trace_archive.py` |
| 新增 | `tools/fastllm_pytools/op_trace_viewer.py` |
| 新增 | `tools/fastllm_pytools/op_trace_dialogs.py` |
| 修改 | `tools/fastllm_pytools/cli.py`（`op-trace-view` 快速路径/帮助） |
| 修改 | `src/utils/op_trace_archive.cpp`（索引扩展 `phase_us/gaps/devices/other_op_us`） |
| 新增 | `test/api/test_op_trace_archive.py` |

P3 已落地：

| 类型 | 文件 |
|---|---|
| 修改 | `tools/fastllm_pytools/op_trace_ui.py`（离线模式/分页/summary 驱动/URL hash/lost 合并） |
| 修改 | `tools/fastllm_pytools/server.py`（disable/clear 自动打包、启动补齐） |
| 新增 | `test/api/test_op_trace_server.py`（自动打包/维护单测） |

P4 导出、静态 HTML、会话管理与导入已落地：

| 类型 | 文件 |
|---|---|
| 修改 | `tools/fastllm_pytools/op_trace_archive.py`（Chrome/Perfetto 流式导出、删除/大小统计、导入校验） |
| 修改 | `tools/fastllm_pytools/op_trace_viewer.py`（export/delete/import 路由、会话列表字段） |
| 修改 | `tools/fastllm_pytools/op_trace_ui.py`（导出按钮、会话管理面板/删除、上传入口） |
| 新增 | `tools/fastllm_pytools/op_trace_static.py`（单 HTML 生成 + 浏览器解析） |
| 修改 | `tools/fastllm_pytools/cli.py`（`op-trace-static` 快速路径） |
| 修改 | `test/api/test_op_trace_archive.py`（导出/HTTP/静态 HTML/删除/导入测试） |

## 18. 待实现时确认的细节

- session 目录默认放在 `<cwd>/.fastllm/op_trace`（P1 已按此实现）。若后续改到
  `%LOCALAPPDATA%`，需要同时修改 native 与 Python 两处默认值并更新文档。
- 实时页与离线页是否允许同时打开同一 session（只读允许多开；
  录制中的 session 只读到 `committed_seq`）。
- summary 的 `tokens` 默认返回最近 8192 条，并附 `tokens_total/tokens_truncated`；
  全量统计由 phase/ops 提供，页面按需分页。
