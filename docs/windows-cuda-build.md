# Windows CUDA 构建与使用

本文说明在 Windows 上用 MSVC + CUDA 编译 FastLLM，以及用本地构建出的 `fastllm_tools.dll` 跑推理服务。
Linux 构建见仓库根目录 `README.md`，本文只覆盖 Windows 专有的步骤与差异。

## 脚本位置

Windows 辅助脚本随仓库一起维护，位于 `scripts\windows\`，命令统一在**仓库根目录**下执行：

| 文件 | 用途 |
| --- | --- |
| `scripts\windows\build_tools.bat` | 编译 `fastllm_tools.dll`（`cpu` / `gpu` 两个后端） |
| `scripts\windows\run_ftllm.bat` | 建 venv 并运行 `ftllm`（`server` / `chat` / `webui` ...） |
| `scripts\windows\ftllm_deps.txt` | `run_ftllm.bat` 安装的运行期 Python 依赖 |

脚本用自身路径上溯两级定位仓库，所以在哪里调用都不影响行为，但下文示例均以仓库根目录为当前目录。

## 前置依赖

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Visual Studio 2022 | 17.x | 需要"使用 C++ 的桌面开发"工作负载，x64 工具链 |
| CUDA Toolkit | 13.4 | 默认路径 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4` |
| NCCL（Windows 移植） | — | CUDA 构建必需，官方不提供 Windows 版，需自行准备带 CMake 配置包的安装目录 |
| Python | 3.10+ | 仅运行 `ftllm` 时需要 |
| uv | 最新 | 仅运行 `ftllm` 时需要，用于建 `fastllm\.venv` |

CPU 构建（`build_tools.bat cpu`）只需要 VS2022。

## 第一步：让 CMake 找到 NCCL

`CMakeLists.txt` 在 Windows 上用 `find_package(NCCL CONFIG REQUIRED)` 定位 NCCL，需要提前给出安装目录：

```bat
set CMAKE_PREFIX_PATH=E:\Learn\nccl-windows\install
```

该目录下应有 `lib\cmake\NCCL\`。CMake 会把同名环境变量当作 `CMAKE_PREFIX_PATH` 的初值，
首次 configure 后路径记在 `build-vs-gpu\CMakeCache.txt` 的 `NCCL_DIR`，后续构建不必再设。
若 configure 报找不到 NCCL，就是这一步漏了或路径写错。

`build_tools.bat` 在 `CUDA_PATH` 未设置时会尝试使用 `...\CUDA\v13.4`；装在别处时先自己 `set CUDA_PATH=...`。

## 第二步：编译

```bat
scripts\windows\build_tools.bat gpu     :: USE_CUDA=ON
scripts\windows\build_tools.bat cpu     :: USE_CUDA=OFF
```

脚本行为：

- 生成器固定 `Visual Studio 17 2022` + `-A x64`，构建目录 `build-vs-gpu`（CPU 为 `build-vs-cpu`）。
- 只在 `build-vs-gpu\CMakeCache.txt` 不存在时才 configure，想换配置先删掉该文件。
- 构建目标只有 `fastllm_tools`，配置 `Release`。
- 并行度默认取 `%NUMBER_OF_PROCESSORS%`，可用环境变量 `FTLLM_BUILD_JOBS` 覆盖。
- 构建完成后检查 `build-vs-gpu\tools\ftllm\fastllm_tools.dll` 是否存在。

CUDA 构建的额外产物是 `tools\ftllm\nccl.dll`（CMake 的后置步骤从 `NCCL::nccl` 拷过来），
缺了它加载 DLL 会报 `找不到指定的模块 (or one of its dependencies)`。

只编译命令行的 `main` 示例时可以直接用 MSBuild：

```bat
cmake --build build-vs-gpu --config Release --target main -- /m
```

## 第三步：运行

```bat
set FTLLM_BUILD=gpu
scripts\windows\run_ftllm.bat server <模型路径> --port 8080 --device cuda
```

`FTLLM_BUILD` 决定用哪份产物（`cpu` / `gpu`，默认 `cpu`）。脚本首次运行会：

1. 用 `uv` 在仓库根目录建 `.venv`（已加入 `.gitignore`）；
2. 按 `scripts\windows\ftllm_deps.txt` 安装依赖；
3. 把 `build-vs-<后端>\tools` 加进 `PYTHONPATH`，并以 `python -m ftllm.cli` 启动。

需要重建 venv 时：`set FTLLM_SETUP=1 && scripts\windows\run_ftllm.bat --help`。

其它子命令同样是 `run_ftllm.bat <命令>`：

| 命令 | 用途 |
| --- | --- |
| `server` / `serve` | OpenAI 兼容 API 服务 |
| `chat` | 终端聊天 |
| `run` | 单次运行 |
| `webui` | 本地 Web UI |
| `benchmark` | 性能测试 |
| `download` | 下载模型 |

## 常用参数

设备与精度（`ftllm server` / `ftllm run` 共用）：

| 参数 | 说明 |
| --- | --- |
| `--device` | 主计算设备，如 `cuda`、`cuda:0`、`cpu`，也支持 `numa` |
| `--moe_device` | MoE 专家层设备，通常配 `--moe_device cpu` 做混合推理 |
| `--moe_device_layers` | 仅最后 N 层 MoE 用 `--moe_device`，`-1` 表示全部 |
| `--dtype` / `--kv_cache_dtype` | 权重与 KV cache 类型，如 `float16`、`fp8_e4m3` |
| `--max_batch` | 单次最大并发请求数 |
| `--moe_pinned_slots` / `--moe_pinned_slot_mb` | MoE 权重流式上传的固定内存中转环槽数与单槽宽(MB)，默认 16 × 4MB |

服务参数：`--host`（默认 `0.0.0.0`）、`--port`（默认 8080）、`--api_key`、`--model_name`。

内置的 C++ 示例 `main` 参数名略有不同：用 `--device` / `--moe_device` / `--moe_device_layers`，
KV cache 类型用 `--kv_dtype`，且 `--moe_device` 必须与 `--device` 同时给出，线程数不填时按 CPU 核数自动选择。

## 环境变量

| 变量 | 作用 |
| --- | --- |
| `FASTLLM_MOE_PINNED_UPLOAD=0` | 关闭 MoE 专家权重的固定内存中转，回退到直接拷贝 H2D |
| `FASTLLM_MOE_PINNED_SLOTS` / `FASTLLM_MOE_PINNED_SLOT_MB` | 中转环默认值（被同名命令行参数覆盖），总量上限 256MB |
| `FASTLLM_MOE_ASYNC_PREFETCH=1` | 单独启用"计算流按事件等拷贝"的预取排序 |
| `FASTLLM_PROFILE_MOE_STREAM=1` | 打印专家流式循环的 upload / compute / release / slotWait 耗时构成 |
| `FASTLLM_PRINT_PROFILE=1` | 打开慢算子、numas-linear、CPU 分配探针等归因日志 |
| `FASTLLM_PREFILL_MOE_GPU=1` | 原型：`--moe_device cpu` 时 prefill 把专家搬到显存用 GPU 算（默认关） |
| `FASTLLM_PREFILL_MOE_GPU_MIN_TOKENS` | 上述原型生效的最小 token 数，默认 1024 |

## 验证

- 启动日志里应出现 `[fastllm-pinned-staging] slots=16 slot=4 MB total=64 MB`，说明中转环按预期建立。
- 服务起来后请求 `http://127.0.0.1:8080/v1/chat/completions`，或直接
  `scripts\windows\run_ftllm.bat chat <模型路径> --device cuda`。
- 需要跑单测时单独配置一份构建目录：

```bat
cmake -S . -B build-vs-gpu-test -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON -DUNIT_TEST=ON
cmake --build build-vs-gpu-test --config Release --target nvfp4Block16GemmRegression
build-vs-gpu-test\Release\nvfp4Block16GemmRegression.exe --bench --gb 1
```

其中 `--linear` 走 `--moe_device cpu` 的 decode 路径（`RunLinearFloat16NVFP4`），不加它则是
`MergeMOE` 用的 `FastllmGemm` 路径。两者内核不同，对比时别混用。

## 已知限制

- **FlashInfer 内核在 MSVC 下关闭**：`CMakeLists.txt` 对 MSVC 强制 `FASTLLM_CUDA_LEGACY_ONLY`，
  attention 走 `fastllm-paged-attention-native.cu` 的原生实现（half / float32 两条路径）。
  因此 Windows 与 Linux 的结果、性能不会完全一致，做精度对比时以 CPU 参考为准。
- **NCCL 无官方 Windows 构建**：多卡通信依赖第三方移植，遇到加载或通信失败先确认
  `build-vs-gpu\tools\ftllm\nccl.dll` 的实际版本与 CUDA 运行时是否匹配。
- **nvcc + MSVC 的编译坑**：CUDA 13 的 CCCL 头文件要求 `/Zc:preprocessor`；
  部分泛型 lambda 实例化会让 nvcc 生成的 host stub 触发 cl.exe `C1001`，故补 `/permissive`。
  这两项已在 `CMakeLists.txt` 里处理。
- **`*.bat` / `*.cmd` 必须 CRLF**：LF 会让 cmd.exe 的标签/块解析错位，仓库已在 `.gitattributes` 中固定。
- **`example/apiserver` 已弃用**：该示例不再维护，服务端统一用 `ftllm server`。
