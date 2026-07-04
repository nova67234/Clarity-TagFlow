@echo off
rem Build the GPU-accelerated (Vulkan) release of Clarity TagFlow.
rem
rem Why this script exists (Windows only):
rem  - llama.cpp's nested vulkan-shaders-gen project breaks under the Visual
rem    Studio cmake generator (MSBuild runs its configure/build/install steps
rem    out of order), so the build must use the Ninja generator from a VS dev
rem    environment.
rem  - The nested build paths exceed MAX_PATH under the project's target\
rem    directory, so the build uses a short CARGO_TARGET_DIR (C:\ctf).
rem
rem Requirements: VS 2022 (C++ workload), Vulkan SDK (VULKAN_SDK set or in
rem C:\VulkanSDK). The first build compiles llama.cpp + all Vulkan shaders and
rem takes several minutes; afterwards it is incremental.
rem
rem Usage:  scripts\build-vulkan.cmd        (build only)
rem         scripts\build-vulkan.cmd run    (build, then launch the app)

setlocal enabledelayedexpansion
cd /d "%~dp0.."

rem --- Locate Visual Studio via vswhere ---
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo error: vswhere.exe not found - is Visual Studio installed?
    exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
    echo error: no Visual Studio with the C++ workload found.
    exit /b 1
)

rem --- Locate the Vulkan SDK (env var, else newest under C:\VulkanSDK) ---
if not defined VULKAN_SDK (
    for /f "delims=" %%d in ('dir /b /ad /o-n "C:\VulkanSDK" 2^>nul') do (
        if not defined VULKAN_SDK set "VULKAN_SDK=C:\VulkanSDK\%%d"
    )
)
if not defined VULKAN_SDK (
    echo error: Vulkan SDK not found. Install it from https://vulkan.lunarg.com/
    exit /b 1
)
echo Using VS:         %VSDIR%
echo Using Vulkan SDK: %VULKAN_SDK%

rem --- Dev env + Ninja + short target dir ---
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CMAKE_GENERATOR=Ninja"
set "CARGO_TARGET_DIR=C:\ctf"
set "PATH=%PATH%;%VULKAN_SDK%\Bin;%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"

cargo build --release --features llm-vulkan
if errorlevel 1 exit /b 1

echo.
echo Built: C:\ctf\release\Clarity_TagFlow.exe
if /i "%~1"=="run" start "" "C:\ctf\release\Clarity_TagFlow.exe"
