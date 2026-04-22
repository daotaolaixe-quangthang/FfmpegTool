@echo off
title FfmpegTool Web UI
color 0A

echo.
echo  ============================================================
echo   FfmpegTool -- Video Frame Extractor ^& Smart Filter
echo   Web UI Server
echo  ============================================================
echo.

:: Move to the tool directory (where app.py lives)
cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Please install Python 3.10+
    echo.
    pause
    exit /b 1
)

:: Check Flask
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo  [INSTALL] Flask not found. Installing...
    pip install flask -q
    echo  [OK] Flask installed.
    echo.
)

:: Add FFmpeg to PATH (winget default install location)
set "PATH=%PATH%;C:\Program Files\FFmpeg\bin;%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1-full_build\bin"

echo  [INFO] Starting server at http://localhost:5000
echo  [INFO] Browser will open automatically...
echo  [INFO] Press Ctrl+C to stop the server.
echo.
echo  ============================================================
echo.

:: Run Flask app — logs print live to this console window
python app.py

:: If it exits (e.g. Ctrl+C), pause so window doesn't close instantly
echo.
echo  [STOPPED] Server has exited.
pause
