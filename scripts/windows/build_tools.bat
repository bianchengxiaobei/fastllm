@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO=%%~fI"
set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=cpu"

rem no goto/labels here: cmd desyncs when seeking a label in this repo's LF-only bat files
set "BACKEND_OK="
if /I "%BACKEND%"=="cpu" set "BACKEND_OK=1"
if /I "%BACKEND%"=="gpu" set "BACKEND_OK=1"
if not defined BACKEND_OK (
    echo [build] usage: build_tools.bat [cpu^|gpu]
    exit /b 1
)

set "SRC=%REPO%"
set "BUILD=%SRC%\build-vs-%BACKEND%"
set "GENERATOR=Visual Studio 17 2022"

if /I "%BACKEND%"=="gpu" (
    if not defined CUDA_PATH (
        if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4" (
            set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
        )
    )
)

if /I "%BACKEND%"=="gpu" echo [build] CUDA_PATH=%CUDA_PATH%

if not exist "%BUILD%\CMakeCache.txt" (
    echo [build] configure %BUILD%
    if /I "%BACKEND%"=="gpu" (
        cmake -S "%SRC%" -B "%BUILD%" -G "%GENERATOR%" -A x64 -DUSE_CUDA=ON
    ) else (
        cmake -S "%SRC%" -B "%BUILD%" -G "%GENERATOR%" -A x64 -DUSE_CUDA=OFF
    )
    if errorlevel 1 (
        echo [build] FAILED
        exit /b 1
    )
)

rem without /m MSBuild compiles serially and uses a single core on this 16-core box
if not defined FTLLM_BUILD_JOBS set "FTLLM_BUILD_JOBS=%NUMBER_OF_PROCESSORS%"
if not defined FTLLM_BUILD_JOBS set "FTLLM_BUILD_JOBS=4"
echo [build] build target fastllm_tools (jobs=%FTLLM_BUILD_JOBS%)
cmake --build "%BUILD%" --config Release --target fastllm_tools -- /m:%FTLLM_BUILD_JOBS% /p:UseMultiToolTask=true /p:EnforceProcessCountAcrossBuilds=true
if errorlevel 1 (
    echo [build] FAILED
    exit /b 1
)

set "TOOLS=%BUILD%\tools"
if not exist "%TOOLS%\ftllm\fastllm_tools.dll" (
    echo [build] target built but %TOOLS%\ftllm\fastllm_tools.dll is missing
    exit /b 1
)

echo [build] ok: %TOOLS%\ftllm\fastllm_tools.dll
echo [build] next: run_ftllm.bat server ^<model path^> --port 8080
exit /b 0
