@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO=%%~fI"
set "SRC=%REPO%\third_party\nccl-windows"
set "BUILD=%SRC%\build"
set "INSTALL=%SRC%\install"

rem no goto/labels here: cmd desyncs when seeking a label in this repo's LF-only bat files
if not exist "%SRC%\CMakeLists.txt" (
    echo [nccl] %SRC% has no source
    echo [nccl] run: git submodule update --init third_party/nccl-windows
    exit /b 1
)

where ninja >nul 2>nul
if errorlevel 1 (
    echo [nccl] ninja not found on PATH
    exit /b 1
)

rem 架构影响 NCCL 内核的编译产物，务必与本机 GPU 一致，可用 NCCL_CUDA_ARCHS 覆盖
if not defined NCCL_CUDA_ARCHS (
    for /f "usebackq delims=" %%A in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%cuda_arch.ps1"`) do set "NCCL_CUDA_ARCHS=%%A"
)
if not defined NCCL_CUDA_ARCHS set "NCCL_CUDA_ARCHS=native"

if not defined CUDA_PATH (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4" (
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
    )
)

rem Ninja 构建必须自己先进入 MSVC x64 环境
if not defined NCCL_VCVARS (
    for /f "usebackq tokens=*" %%A in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set "NCCL_VCVARS=%%A\VC\Auxiliary\Build\vcvarsall.bat"
)
if not exist "%NCCL_VCVARS%" (
    echo [nccl] cannot find vcvarsall.bat, set NCCL_VCVARS manually
    exit /b 1
)
call "%NCCL_VCVARS%" x64 > nul
if errorlevel 1 (
    echo [nccl] vcvarsall.bat x64 failed
    exit /b 1
)

if not defined NCCL_JOBS set "NCCL_JOBS=%NUMBER_OF_PROCESSORS%"
if not defined NCCL_JOBS set "NCCL_JOBS=4"

echo [nccl] archs=%NCCL_CUDA_ARCHS% jobs=%NCCL_JOBS%
echo [nccl] install %INSTALL%
cmake -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=%NCCL_CUDA_ARCHS% -DCMAKE_INSTALL_PREFIX="%INSTALL%"
if errorlevel 1 (
    echo [nccl] FAILED
    exit /b 1
)

cmake --build "%BUILD%" --parallel %NCCL_JOBS% --target install
if errorlevel 1 (
    echo [nccl] FAILED
    exit /b 1
)

if not exist "%INSTALL%\lib\cmake\NCCL\NCCLConfig.cmake" (
    echo [nccl] install finished but %INSTALL%\lib\cmake\NCCL\NCCLConfig.cmake is missing
    exit /b 1
)

echo [nccl] ok: %INSTALL%\lib\cmake\NCCL\NCCLConfig.cmake
echo [nccl] build_tools.bat gpu picks this install up automatically
exit /b 0
