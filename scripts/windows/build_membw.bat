@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "SRC=%SCRIPT_DIR%membw.cpp"
set "EXE=%SCRIPT_DIR%membw.exe"

rem Builds the standalone DRAM read-bandwidth probe next to this bat.
rem No Visual Studio IDE needed: MSVC build tools are enough.
rem No goto/labels here: cmd desyncs when seeking a label in this repo's LF-only bat files.

set "VSPATH="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%I"

where cl >nul 2>nul
if errorlevel 1 if defined VSPATH call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul

where cl >nul 2>nul
if errorlevel 1 (
    echo [membw] FAIL: cl.exe not found.
    echo [membw] install the build tools first: scripts\windows\setup_build_env.bat
    exit /b 1
)

if not exist "%SRC%" (
    echo [membw] FAIL: missing %SRC%
    exit /b 1
)

echo [membw] compile %SRC%
pushd "%SCRIPT_DIR%"
cl /nologo /O2 /utf-8 /std:c++17 /EHsc /Fe:"%EXE%" "%SRC%"
set "CL_RC=%errorlevel%"
if exist "%SCRIPT_DIR%membw.obj" del /q "%SCRIPT_DIR%membw.obj" >nul 2>nul
popd
if not "%CL_RC%"=="0" (
    echo [membw] FAILED
    exit /b 1
)
if not exist "%EXE%" (
    echo [membw] FAILED: %EXE% was not produced
    exit /b 1
)

echo [membw] ok: %EXE%
echo [membw] stop the model service first, then run: membw.exe
if /I "%~1"=="run" "%EXE%"
exit /b 0
