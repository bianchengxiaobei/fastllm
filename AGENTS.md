# 项目技能

## Skills

### Available skills

- git-commit-zh-split: 当用户要求提交代码、整理提交、准备 commit、拆分 commit、push，或指定提交与推送规范时使用。默认使用中文提交信息，将差异较大的改动拆分为多个提交；推送前先执行 fetch、stash、rebase、stash pop，再 push。 (file: .codex/skills/git-commit-zh-split/SKILL.md)
- fastllm-triton-ops: 当需要在 FastLLM CUDA 后端添加、扩展、调试、验证或测速 Triton 生成的算子时使用，覆盖 Python 编译服务、cubin 缓存、C++ fallback 接入和验证流程。 (file: .codex/skills/fastllm-triton-ops/SKILL.md)

### How to use skills

- Discovery: 上面列出的 skill 是当前仓库可用的项目级 skill，路径相对于本文件所在的仓库根目录。
- Trigger rules: 当用户直接提到 skill 名称，或请求明显匹配 skill 描述时，必须使用对应 skill。
- Missing/blocked: 如果 skill 文件缺失或无法读取，简短说明后继续采用最佳替代方案。
- Context hygiene: 先读 `SKILL.md`，只在需要时再加载更多上下文。

## 开发须知

### ftllm 的 Python 包是编译期拷贝出来的副本

`run_ftllm.bat` / `ftllm` 命令把 `PYTHONPATH` 指向编译产物目录，运行时导入的是
`tools/fastllm_pytools` 的**副本**，不是源码：

- 源码：`fastllm/tools/fastllm_pytools/*.py`
- 运行时实际加载：`fastllm/build-vs-<backend>/tools/ftllm/*.py`

副本由 `fastllm_tools` target 的 POST_BUILD `copy_directory` 生成
（见 `CMakeLists.txt` 中 `tools/fastllm_pytools` 那几条 `copy_directory`）。

踩坑点：只改 Python 源码不重新拷贝，改动完全不生效，而且没有任何报错，容易误判成
"打印/开关没实现"。

- 改完 Python 源码必须跑 `scripts/windows/build_tools.bat [cpu|gpu]` 重新拷贝，
  或手工 `copy /Y <源码文件> build-vs-<backend>/tools/ftllm/`。
- 只跑 `build_gpu.bat`（target 是 `main`）**不会**触发那次拷贝。
- Python 模块只在进程启动时 import 一次，验证前必须重启 server。

排查手法（先确认代码到底加载的是哪份）：

- `fc /n <源码文件> <副本文件>` 看两者差异。
- 在副本里 `grep` 新增的符号，例如 `grep DecodeTextPrinter build-vs-gpu/tools/ftllm/llm.py`。

同理，定位"服务端实际走哪条生成路径"要顺链路查，别只看函数名：

- `ftllm server` → `openai_server/fastllm_completion.py` → `model.launch_stream_response`
  + `stream_response_handle_async`（常驻调度线程路径，走 `basellm::LaunchResponseTokens` /
  `FetchResponseTokens`），**不经过** `basellm::Response` / `ResponseBatch`；
  后者只服务 CLI `main` 和 `batch_response`。
