@echo off
setlocal
rem Install the Windows build prerequisites of FastLLM from a bare machine.
rem   setup_build_env.bat          install whatever is missing
rem   setup_build_env.bat check    report only, install nothing
rem FTLLM_CUDA_VERSION picks the toolkit (13.4 default, 12.9 for the V100 artifact).
rem All comments in this file stay ASCII: UTF-8 Chinese inside rem breaks cmd parsing.

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO=%%~fI"

set "MODE=%~1"
set "CHECK_ONLY="
if /I "%MODE%"=="check" set "CHECK_ONLY=1"
set "CUDAVER_EXPLICIT="
if defined FTLLM_CUDA_VERSION set "CUDAVER_EXPLICIT=1"
if not defined FTLLM_CUDA_VERSION set "FTLLM_CUDA_VERSION=13.4"
set "WG=winget install -e --accept-package-agreements --accept-source-agreements --disable-interactivity"
set "MISSING="

echo [setup] mode=%MODE% cuda=%FTLLM_CUDA_VERSION% repo=%REPO%

where winget >nul 2>nul
if errorlevel 1 (
    echo [setup] FAIL: winget is missing
    echo [setup] on Windows 10/11 install "App Installer" from the Microsoft Store, or install manually:
    echo [setup]   VS Build Tools https://aka.ms/vs/17/release/vs_buildtools.exe
    echo [setup]   CMake  https://cmake.org/download/    Ninja https://github.com/ninja-build/ninja/releases
    echo [setup]   CUDA   https://developer.nvidia.com/cuda-toolkit-archive
    exit /b 1
)

rem --- git: only needed for the nccl submodule and for pulling updates ---
where git >nul 2>nul
if errorlevel 1 (
    set "MISSING=%MISSING% git"
    if not defined CHECK_ONLY (
        echo [setup] installing Git
        %WG% --id Git.Git
    ) else (
        echo [setup] missing: git
    )
)

rem --- MSVC toolchain: VS Build Tools is enough, no IDE needed ---
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VCPATH="
if exist "%VSWHERE%" for /f "usebackq delims=" %%A in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCPATH=%%A"
rem quote every path echoed from inside an if ( ) body: a ) inside the path (e.g. "Program Files (x86)")
rem closes the block while cmd is still reading it and the script dies with "was unexpected at this time"
if defined VCPATH (
    echo [setup] ok: MSVC toolchain at "%VCPATH%"
) else (
    set "MISSING=%MISSING% vs-buildtools"
    if not defined CHECK_ONLY (
        echo [setup] installing Visual Studio 2022 Build Tools with the C++ workload ^(about 3 GB^)
        %WG% --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --quiet --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    ) else (
        echo [setup] missing: MSVC toolchain ^(Visual Studio 2022 Build Tools, VCTools workload^)
    )
)

rem --- CMake drives both nccl and fastllm ---
where cmake >nul 2>nul
if errorlevel 1 (
    set "MISSING=%MISSING% cmake"
    if not defined CHECK_ONLY (
        echo [setup] installing CMake
        %WG% --id Kitware.CMake
    ) else (
        echo [setup] missing: cmake
    )
)

rem --- Ninja is only needed by the NCCL step, install it anyway ---
where ninja >nul 2>nul
if errorlevel 1 (
    set "MISSING=%MISSING% ninja"
    if not defined CHECK_ONLY (
        echo [setup] installing Ninja
        %WG% --id Ninja-build.Ninja
    ) else (
        echo [setup] missing: ninja
    )
)

rem --- CUDA toolkit: a pinned version must exist as-is; otherwise honour CUDA_PATH ---
set "CUDAROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%FTLLM_CUDA_VERSION%"
if not exist "%CUDAROOT%\bin\nvcc.exe" if not defined CUDAVER_EXPLICIT if defined CUDA_PATH set "CUDAROOT=%CUDA_PATH%"
if exist "%CUDAROOT%\bin\nvcc.exe" (
    echo [setup] ok: CUDA toolkit at "%CUDAROOT%"
) else (
    set "MISSING=%MISSING% cuda"
    if not defined CHECK_ONLY (
        echo [setup] installing CUDA %FTLLM_CUDA_VERSION% ^(several GB, takes a while^)
        %WG% --id Nvidia.CUDA --version %FTLLM_CUDA_VERSION%
    ) else (
        echo [setup] missing: CUDA %FTLLM_CUDA_VERSION% ^(expected C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%FTLLM_CUDA_VERSION%^)
    )
)

rem --- Python 3.10+ for ftllm itself; the NCCL configure also wants an interpreter ---
set "PY365="
for /f "usebackq delims=" %%A in (`py -3 -c "import sys;print(sys.version_info>=(3,10))" 2^>nul`) do set "PY365=%%A"
set "PY3VER="
for /f "usebackq delims=" %%A in (`py -3 --version 2^>nul`) do set "PY3VER=%%A"
if "%PY365%"=="True" (
    echo [setup] ok: %PY3VER%
) else (
    set "MISSING=%MISSING% python3"
    if not defined CHECK_ONLY (
        echo [setup] installing Python 3.12
        %WG% --id Python.Python.3.12
    ) else (
        echo [setup] missing: Python 3.10+ ^(found "%PY3VER%"^)
    )
)

rem --- uv is what run_ftllm.bat uses to build its venv ---
where uv >nul 2>nul
if errorlevel 1 (
    set "MISSING=%MISSING% uv"
    if not defined CHECK_ONLY (
        echo [setup] installing uv
        %WG% --id astral-sh.uv
    ) else (
        echo [setup] missing: uv
    )
)

rem --- the display driver is not installable here; it is required to run, not to build ---
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [setup] WARN: nvidia-smi not found, install the NVIDIA driver before running a model
)

echo.
if defined MISSING (
    echo [setup] still missing:%MISSING%
    if defined CHECK_ONLY echo [setup] rerun without the check argument to install them
    exit /b 1
)

echo [setup] all prerequisites present
echo [setup] open a NEW terminal before building: PATH changes are not visible in this one
echo [setup] then:
echo [setup]   git submodule update --init third_party/nccl-windows
echo [setup]   scripts\windows\build_nccl.bat
echo [setup]   scripts\windows\build_tools.bat gpu
echo [setup]   scripts\windows\run_ftllm.bat server ^<model path^>
exit /b 0
