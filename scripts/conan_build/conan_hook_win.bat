@echo off
setlocal enabledelayedexpansion

:: Set output directory from input parameter, use default if not provided
set "output_dir=%1"
if "!output_dir!"=="" (
    :: Default output directory: project_root/build/generators
    set "output_dir=%~dp0..\..\build\generators"
)

:: Create output directory (suppress error if directory already exists)
mkdir "!output_dir!" >nul 2>&1
if not exist "!output_dir!" (
    echo ERROR: Failed to create output directory: !output_dir!
    exit /b 1
)

:: Calculate project root directory based on script location
set "SCRIPT_DIR=%~dp0"  :: Get directory of current batch script
set "PROJECT_SOURCE_DIR=!SCRIPT_DIR!..\.."  :: Navigate up two levels to project root
pushd "!PROJECT_SOURCE_DIR!" >nul  :: Convert to absolute path
set "PROJECT_SOURCE_DIR=!cd!"
popd >nul

:: Define path to Conan profile (x86_64 version)
set "PROFILE=!SCRIPT_DIR!conan_profile_win.x86_64"

:: Check if Conan profile exists
if not exist "!PROFILE!" (
    echo ERROR: Conan profile not found at: !PROFILE!
    exit /b 1
)

:: Run Conan install with explicit profiles (CRITICAL: No comments in command lines)
echo [Conan] Installing dependencies...
echo [Conan] Project root: !PROJECT_SOURCE_DIR!
echo [Conan] Output directory: !output_dir!
echo [Conan] Using profile: !PROFILE!

:: Conan command with proper line continuation (^ must be the last character of each line)
conan install "!PROJECT_SOURCE_DIR!" ^
    --output-folder "!output_dir!" ^
    --build=missing ^
    -pr:b "!PROFILE!" ^
    -pr:h "!PROFILE!" ^
    -v

:: Check toolchain file in both possible locations
set "TOOLCHAIN_FILE=!output_dir!\generators\conan_toolchain.cmake"
if not exist "!TOOLCHAIN_FILE!" (
    set "TOOLCHAIN_FILE=!output_dir!\conan_toolchain.cmake"
    if not exist "!TOOLCHAIN_FILE!" (
        echo ERROR: Conan failed to generate toolchain file: !TOOLCHAIN_FILE!
        exit /b 1
    )
)

echo [Conan] Toolchain file generated successfully.
endlocal