# Windows CUDA 构建与使用

本文说明在 Windows 上用 MSVC + CUDA 编译 FastLLM，以及用本地构建出的 `fastllm_tools.dll` 跑推理服务。
Linux 构建见仓库根目录 `README.md`，本文只覆盖 Windows 专有的步骤与差异。

## 脚本位置

Windows 辅助脚本随仓库一起维护，位于 `scripts\windows\`，命令统一在**仓库根目录**下执行：

| 文件 | 用途 |
| --- | --- |
| `scripts\windows\setup_build_env.bat` | 干净机器上一键用 winget 装齐构建依赖（加 `check` 只体检不安装） |
| `scripts\windows\build_tools.bat` | 编译 `fastllm_tools.dll`（`cpu` / `gpu` 两个后端） |
| `scripts\windows\build_nccl.bat` | 编译子模块里的 NCCL for Windows（`build_tools.bat gpu` 的前置依赖） |
| `scripts\windows\cuda_arch.ps1` | 探测本机 GPU 架构号，供 `build_nccl.bat` 使用 |
| `scripts\windows\run_ftllm.bat` | 建 venv 并运行 `ftllm`（`server` / `chat` / `webui` ...） |
| `scripts\windows\ftllm_deps.txt` | `run_ftllm.bat` 安装的运行期 Python 依赖 |

脚本用自身路径上溯两级定位仓库，所以在哪里调用都不影响行为，但下文示例均以仓库根目录为当前目录。

## 前置依赖

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Visual Studio 2022 | 17.x | 需要"使用 C++ 的桌面开发"工作负载，x64 工具链；VS IDE 或 VS Build Tools 都行（见"多架构（可移植）构建"） |
| CUDA Toolkit | 13.4 | 默认路径 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4` |
| NCCL（Windows 移植） | 2.29.x | CUDA 构建必需，官方不提供 Windows 版；仓库已把 [SystemPanic/nccl-windows](https://github.com/SystemPanic/nccl-windows) 挂成 `third_party/nccl-windows` 子模块并钉住 commit |
| Ninja | 1.10+ | 仅编译 NCCL 时需要 |
| Python | 3.10+ | 仅运行 `ftllm` 时需要 |
| uv | 最新 | 仅运行 `ftllm` 时需要，用于建 `fastllm\.venv` |

CPU 构建（`build_tools.bat cpu`）只需要 VS2022。

## 干净机器：先跑 setup_build_env.bat

只 clone 了源码、机器上什么都没有时，先让脚本把上面这些依赖用 winget 装齐：

```bat
scripts\windows\setup_build_env.bat check   :: 只体检，列出缺什么
scripts\windows\setup_build_env.bat         :: 装缺的那些
```

它会依次检查并安装（已有的跳过）：

| 依赖 | winget 包 | 说明 |
| --- | --- | --- |
| MSVC 工具链 | `Microsoft.VisualStudio.2022.BuildTools` | 带 `VCTools` 工作负载，装的是**命令行工具链不是 IDE** |
| CMake | `Kitware.CMake` | NCCL 和 fastllm 都靠它 |
| Ninja | `Ninja-build.Ninja` | 只有编 NCCL 用 |
| CUDA Toolkit | `Nvidia.CUDA` | 版本由 `FTLLM_CUDA_VERSION` 决定，默认 `13.4` |
| Python | `Python.Python.3.12` | 3.10+ 才能跑 ftllm，NCCL 的 configure 也要一个解释器 |
| uv | `astral-sh.uv` | `run_ftllm.bat` 用它建 venv |
| Git | `Git.Git` | 拉 nccl 子模块、更新代码 |

几个必然的限制，脚本不会替你解决：

- **要管理员权限**：winget 安装会弹 UAC，第一次装 VS Build Tools 约 3 GB，CUDA 约 4 GB，加上编译时间，整机从零到能跑通常要一小时以上。
- **装完要开新终端**：winget 改的是注册表里的 PATH，当前窗口看不到，所以脚本结尾会提示重开一个终端再执行 `build_nccl.bat`。
- **显卡驱动不装**：驱动是运行期依赖，脚本只检测 `nvidia-smi` 并提醒，需要自己去 NVIDIA 官网装（V100 用 580 分支）。
- **winget 本身缺失时**（部分 Windows Server / LTSC 镜像）脚本会直接失败并打印各组件的手动下载地址，先装 "App Installer" 再重试。
- `FTLLM_CUDA_VERSION=12.9` 会让脚本去找/装 12.9，那是给 V100 那份产物用的（见下文"V100（sm_70）"一节）。

## 第一步：准备 NCCL

CUDA 构建要链接 NCCL，官方没有 Windows 版，所以仓库把 Windows 移植挂成了子模块：

```bat
git submodule update --init third_party/nccl-windows
scripts\windows\build_nccl.bat
```

`build_nccl.bat` 会自动探测 GPU 架构（`cuda_arch.ps1` 读 `nvidia-smi`，取不到时交给 CMake 的 `native`）、
`call vcvarsall.bat x64` 进入 MSVC 环境，再以 Ninja 构建并安装到 `third_party\nccl-windows\install`。
产物 100MB 量级，首次约十几分钟。

可覆盖项：

| 变量 | 说明 |
| --- | --- |
| `NCCL_CUDA_ARCHS` | 目标架构号，如 `75`、`86;120`；默认按本机 GPU 探测 |
| `NCCL_JOBS` | 并行编译任务数，默认取 `%NUMBER_OF_PROCESSORS%` |
| `NCCL_VCVARS` | `vcvarsall.bat` 的绝对路径，默认由 `vswhere` 查找 |
| `NCCL_BUILD_DIR` / `NCCL_INSTALL_DIR` | 构建与安装目录，默认 `third_party\nccl-windows\build` 和 `...\install`；并列多套工具链时用它们分开 |

之后 `build_tools.bat gpu` 会自动把 `third_party\nccl-windows\install` 当作 `CMAKE_PREFIX_PATH`
（`CMakeLists.txt` 里是 `find_package(NCCL CONFIG REQUIRED)`）。要改用别的 NCCL，
先 `set NCCL_ROOT=<install 目录>` 或直接 `set CMAKE_PREFIX_PATH=<install 目录>`。

该路径只在首次 configure 时解析，结果记在 `build-vs-gpu\CMakeCache.txt` 的 `NCCL_DIR`；
换了 NCCL 位置要删掉 `CMakeCache.txt` 重新 configure。configure 报找不到 NCCL，就是这一步没做。

`build_tools.bat` 在 `CUDA_PATH` 未设置时会尝试使用 `...\CUDA\v13.4`；装在别处时先自己 `set CUDA_PATH=...`。

## 第二步：编译

```bat
scripts\windows\build_tools.bat gpu     :: USE_CUDA=ON
scripts\windows\build_tools.bat cpu     :: USE_CUDA=OFF
```

脚本行为：

- 生成器固定 `Visual Studio 17 2022` + `-A x64`，构建目录 `build-vs-gpu`（CPU 为 `build-vs-cpu`），可用 `FTLLM_BUILD_DIR` 换成别的目录名（相对仓库根）。
- CUDA 构建可用 `FTLLM_CUDA_ARCHS` 指定架构列表（见"多架构（可移植）构建"），不设则用 `native`。
- 只在 `build-vs-gpu\CMakeCache.txt` 不存在时才 configure，想换配置先删掉该文件。
- 构建目标只有 `fastllm_tools`，配置 `Release`。
- 并行度默认取 `%NUMBER_OF_PROCESSORS%`，可用环境变量 `FTLLM_BUILD_JOBS` 覆盖。
- 构建完成后检查 `build-vs-gpu\tools\ftllm\fastllm_tools.dll` 是否存在。

CUDA 构建的额外产物是 `tools\ftllm\nccl.dll`（CMake 的后置步骤从 `NCCL::nccl` 拷过来），
缺了它加载 DLL 会报 `找不到指定的模块 (or one of its dependencies)`。

两个容易踩的点：

- `gpu` 后端依赖第一步的 NCCL，没做第一步就直接 configure 会报找不到 NCCL。
- 构建前要停掉正在运行的 `ftllm` 服务。它加载着 `build-vs-gpu\tools\ftllm\fastllm_tools.dll`，
  后置的拷贝步骤会以 `Error copying file "...\Release\fastllm_tools.dll" to "...\tools\ftllm\.": Permission denied`
  报错退出（MSB3073），此时 DLL 已经编译出来但没同步到运行目录。

只编译命令行的 `main` 示例时可以直接用 MSBuild：

```bat
cmake --build build-vs-gpu --config Release --target main -- /m
```

## 多架构（可移植）构建

默认 `CUDA_ARCH` 是 `native`（`CMakeLists.txt` 顶部），只编本机算力、也不产 PTX。
把这种产物拷到算力不同的卡上，加载时只会得到
`CUDA error 209: no kernel image is available for execution on the device`，驱动连 JIT 都没有东西可用。

要出能拿到别的卡上跑的产物，configure 时把架构列表写全：

```bat
set FTLLM_CUDA_ARCHS=75;80;86;89;90;120
set NCCL_CUDA_ARCHS=75;80;86;89;90;120
scripts\windows\build_nccl.bat
scripts\windows\build_tools.bat gpu
```

- 两个架构变量都只在**首次 configure** 生效，改完要删掉 `build-vs-gpu\CMakeCache.txt`（NCCL 是 `third_party\nccl-windows\build\CMakeCache.txt`）重新 configure。
- NCCL 的内核也是按架构出的，必须跟 `FTLLM_CUDA_ARCHS` 用同一批，否则多卡通信在别的卡上照样 `no kernel image`。
- `120` 会被归一化成 `120f`（CUDA 12.9+ 的 family cubin），`121` 同理；列表里有 `90` 时会额外拉起 deep_gemm 的 `sm_90a` 目标，有 `120` 时会拉起 SM120 的专用目标。
- 本仓库的 `CUDA_ARCH` 只出 cubin，不出 `-virtual` 的 PTX，所以"未来的卡"也要显式加算力号。
- 体积：单架构的 `fastllm_tools.dll` 约 42 MB，每多一个架构大约多一份 CUDA 代码（构建已开 `-Xfatbin=-compress-all`）。

只装命令行工具链（不装 VS IDE）也够用，生成器和 `vswhere`/`vcvarsall.bat` 都在 Build Tools 里：

```bat
winget install Microsoft.VisualStudio.2022.BuildTools
:: 安装向导里勾上"使用 C++ 的桌面开发"（VCTools）工作负载
winget install Kitware.CMake Ninja-build.Ninja
```

Ninja 只有编 NCCL 那一步要，编 fastllm 本体不需要。

产物拷到**没装 CUDA/VS 的目标机**上跑时，还要补齐这些：

| 文件 | 来源 |
| --- | --- |
| `MSVCP140.dll`、`VCRUNTIME140.dll`、`VCRUNTIME140_1.dll` | VC++ 2015-2022 x64 运行库（`vc_redist.x64.exe`），或从 `...\VC\Redist\MSVC\<版本>\x64\Microsoft.VC143.CRT\` 拷 |
| `cublas64_13.dll`、`cublasLt64_13.dll` | CUDA Toolkit 的 `bin\x64`（合计约 550 MB，未随 DLL 一起出） |
| `nccl.dll` | 已在 `tools\ftllm\`，CMake 后置步骤从 `NCCL::nccl` 拷的 |
| `nvcuda.dll`、`nvcudart_hybrid64.dll` | NVIDIA 驱动自带，需要 r580+ 驱动（CUDA 13.x 的门槛） |

另外两点：主体按 `/arch:AVX2` 编译（少数文件 `/arch:AVX10.1`），目标机 CPU 至少要支持 AVX2；
`fastllm_tools.dll` 是 `ctypes.CDLL` 加载的纯 C 接口，跟 Python 版本无关，远端任意 Python 3 都能拉起来。

## V100（sm_70）要单独一套 CUDA 12.x 产物

CUDA 13.0 起 nvcc 移除了算力 < 7.5 的离线编译，本机 `nvcc --list-gpu-arch` 最小就是 `compute_75`，
所以 `FTLLM_CUDA_ARCHS` 里写 `70` 只会得到 unsupported gpu architecture，PTX 也救不了（PTX 只能往高算力 JIT）。
要让 V100 跑起来，必须用 **CUDA 12.x**（12.9 是最后一个还认 `sm_70` 的版本，已标 deprecated；
12.6 更稳，但只覆盖到 `sm_90`，没有 `120f` 归一化）。

工程本身有 Volta 路径，不用改代码：`src/devices/cuda/awq_sm70/`（TurboMind s884 子集，见 `CMakeLists.txt:561-572`）、
`include/devices/cuda/fastllm-fp8-sm70.cuh`（`__CUDA_ARCH__ == 700` 专用实现）、Marlin 在 `__CUDA_ARCH__ < 750` 下编成占位实现运行期回退。

两套产物并存的配方（各自独立构建目录 + 独立 NCCL install，互不覆盖）：

```bat
:: 1) V100：CUDA 12.9
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set NCCL_CUDA_ARCHS=70;75;86;89;90
set NCCL_BUILD_DIR=%CD%\third_party\nccl-windows\build-cu129
set NCCL_INSTALL_DIR=%CD%\third_party\nccl-windows\install-cu129
scripts\windows\build_nccl.bat

set NCCL_ROOT=%NCCL_INSTALL_DIR%
set FTLLM_CUDA_ARCHS=%NCCL_CUDA_ARCHS%
set FTLLM_BUILD_DIR=build-vs-gpu-cu129
scripts\windows\build_tools.bat gpu
:: 产物 build-vs-gpu-cu129\tools\ftllm\fastllm_tools.dll

:: 2) sm_75 及以上：照旧用 CUDA 13.4，构建目录仍是 build-vs-gpu，两边互不影响
```

几个要紧的点：

- NCCL 用哪套工具链编就配哪套产物，别共用 `install`：同进程里同时加载 `cudart64_12` 和 `cudart64_13` 官方不支持，
  出问题很难查。`build_nccl.bat` 的 `CUDA_PATH` 同样会走上面的 12.9。
- 12.9 下 `sm_120`/`120f` 的目标照样会打开（归一化门槛是 `>= 12.9`），所以这一份可以直接带 `120`。
- 打包时别改 DLL 文件名：`fastllm_pytools\llm.py` 里写死了 `fastllm_tools.dll`，按卡选目录（两套 `tools\ftllm`）分发。
- 验产物算力：`cuobjdump --list-elf <dll> | findstr sm_`，CUDA 12.9 那份应能看到 `sm_70`。
- V100 目标机侧：驱动要用 **580 分支**（最后一个支持 Volta 的驱动分支），cuBLAS 换成 CUDA 12 的
  `cublas64_12.dll`/`cublasLt64_12.dll`，其余运行库要求同上。
- 预期要压低：Volta 没有 bf16、没有 cp.async，FP8 走 `fastllm-fp8-sm70.cuh` 的模拟实现，
  paged attention 落在 `__CUDA_ARCH__ < 800` 的回退分支，Marlin int4 也是回退路径。

## 第三步：运行

```bat
scripts\windows\run_ftllm.bat server <模型路径> --port 8080 --device cuda
```

GPU 主计算 + 专家在主机内存（`--moe_device cpu`）的长输入场景，推荐参数见
[MoE 主机专家流式 prefill 调优](moe-prefill-tuning.md)。

`FTLLM_BUILD` 决定用哪份产物（`cpu` / `gpu`）。不设置时会自动挑：存在
`build-vs-gpu\tools\ftllm\fastllm_tools.dll` 就用 `gpu`，否则用 `cpu`。脚本首次运行会：

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
| `--chunked_prefill_size` | 分块 prefill 的切片大小，如 `4096`；前缀快照间隔会自动跟着它抬起来，不用再单独设 `--prefix_cache_snapshot_interval_pages` |
| `--moe_pinned_slots` / `--moe_pinned_slot_mb` | MoE 权重流式上传的固定内存中转环槽数与单槽宽(MB)，默认 16 × 4MB |
| `--moe_pin_host_weights` | 在主机侧钉住 MoE 专家权重的额度(MB)，默认 0 关闭；钉住后 DMA 直接从权重下发，省掉写中转槽的主机拷贝，但这块内存不可换出 |
| `--prefill_moe_gpu` | prefill 把主机上的路由专家搬上卡用 GPU 算，`1` 开 / `0` 关；默认 `-1` 按配置自动（专家在 cpu 内存里就开） |

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
| `FASTLLM_PREFILL_MOE_GPU` | 原型：`--moe_device cpu` 时 prefill 把专家搬到显存用 GPU 算；不设置时按配置自动（专家在 cpu 内存里就开），可用 `--prefill_moe_gpu 0/1` 显式覆盖 |
| `FASTLLM_PREFILL_MOE_GPU_MIN_TOKENS` | 上述原型生效的最小 token 数，默认 1024 |
| `FASTLLM_CUDA_INT4_GROUP_PREFILL_CUBLAS` | int4 group 权重在 prefill 大 M 上改走「反量化 + cuBLAS」而不是 Marlin（Marlin 是解码向内核）。默认关；`1` 打开。2060 SUPER 上实测与 Marlin 持平，属于可选的 A/B 开关，见 [MoE 主机专家流式 prefill 调优](moe-prefill-tuning.md) 第 6 节 |

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
