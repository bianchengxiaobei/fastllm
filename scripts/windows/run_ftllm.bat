@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO=%%~fI"
rem 没显式指定就挑已经编好的产物: 有 gpu 物料就用 gpu, 否则退回 cpu,
rem 省得每次开窗口都要先 set FTLLM_BUILD
if "%FTLLM_BUILD%"=="" (
    if exist "%REPO%\build-vs-gpu\tools\ftllm\fastllm_tools.dll" (
        set "FTLLM_BUILD=gpu"
    ) else (
        set "FTLLM_BUILD=cpu"
    )
)

set "TOOLS=%REPO%\build-vs-%FTLLM_BUILD%\tools"
set "VENV=%REPO%\.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "DEPS=%SCRIPT_DIR%ftllm_deps.txt"

if not exist "%TOOLS%\ftllm\fastllm_tools.dll" (
    echo [ftllm] cannot find %TOOLS%\ftllm\fastllm_tools.dll
    echo [ftllm] run build_tools.bat %FTLLM_BUILD% first
    exit /b 1
)

if not exist "%VPY%" set "FTLLM_SETUP=1"

if defined FTLLM_SETUP (
    where uv >nul 2>nul
    if errorlevel 1 (
        echo [ftllm] uv not found on PATH, see https://docs.astral.sh/uv/
        exit /b 1
    )
    echo [ftllm] create venv %VENV%
    if defined FTLLM_PYTHON (
        uv venv "%VENV%" --python %FTLLM_PYTHON%
    ) else (
        uv venv "%VENV%"
    )
    if errorlevel 1 exit /b 1
    echo [ftllm] install deps from ftllm_deps.txt
    uv pip install --python "%VPY%" -r "%DEPS%"
    if errorlevel 1 exit /b 1
)

if defined PYTHONPATH (
    set "PYTHONPATH=%TOOLS%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%TOOLS%"
)

"%VPY%" -m ftllm.cli %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo [ftllm] exit=%RC%
    echo [ftllm] rebuild venv: set FTLLM_SETUP=1 ^&^& run_ftllm.bat --help
)
exit /b %RC%
