@echo off
cd /d "%~dp0"
python -m src.run_all %*
pause
