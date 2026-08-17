@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo 缺少虚拟环境，请先双击运行「一键修复环境.bat」
    pause
    exit /b 1
)
".venv\Scripts\python.exe" gui_app.py
pause
