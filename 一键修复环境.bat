@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   漫画翻译助手 - 一键修复环境
echo ============================================
echo.
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 正在创建虚拟环境 ...
    python -m venv .venv
    if errorlevel 1 (
        echo 创建失败：请确认已安装 Python 3.10 或更高版本
        pause
        exit /b 1
    )
) else (
    echo [1/3] 虚拟环境已存在
)
echo [2/3] 正在安装依赖（首次需要联网，约几百MB，请耐心等待）...
".venv\Scripts\python.exe" -m pip install -r requirements.txt python-dotenv
if errorlevel 1 (
    echo 安装失败，请检查网络后重试
    pause
    exit /b 1
)
echo [3/3] 完成！现在可以双击 run_gui.bat 启动程序。
pause
