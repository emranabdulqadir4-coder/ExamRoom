@echo off
REM One-click start for Windows: creates the virtual environment and installs the requirements the first
REM time, then starts ExamGuard (restarting it if it stops).   start.bat --https --host 0.0.0.0
cd /d "%~dp0"
if not exist .venv (
  echo First run: creating the virtual environment ...
  python -m venv .venv
  if errorlevel 1 (
    echo Could not create it. Install Python 3.10-3.12 from python.org and tick "Add Python to PATH".
    pause
    exit /b 1
  )
)
call .venv\Scripts\activate.bat
if not exist .venv\.requirements-installed (
  echo Installing requirements ^(needs internet, a few minutes^) ...
  python -m pip install --upgrade pip >nul
  pip install -r requirements.txt
  if errorlevel 1 (
    echo Installing failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
  echo done> .venv\.requirements-installed
)
python start.py %*
pause
