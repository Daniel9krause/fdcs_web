@echo off
setlocal
cd /d "%~dp0"
rem ---- find the Python that has FDCS installed (your env / venv folder) ----
set "PY="
for %%P in ("%~dp0..\env\Scripts\python.exe" "%~dp0env\Scripts\python.exe" "%~dp0..\venv\Scripts\python.exe" "%~dp0venv\Scripts\python.exe" "%~dp0..\.venv\Scripts\python.exe" "%~dp0.venv\Scripts\python.exe") do (
  if not defined PY if exist %%P set "PY=%%~P"
)
if not defined PY (
  echo No virtual environment found next to this folder - creating one in "%~dp0env" ...
  python -m venv "%~dp0env" || (echo Python is not installed. Get Python 3.11 from python.org and tick "Add to PATH". & pause & exit /b 1)
  set "PY=%~dp0env\Scripts\python.exe"
)
rem ---- install packages the first time ----
"%PY%" -c "import flask, waitress, qrcode" 2>nul || (
  echo Installing packages - first time only, this can take 5-10 minutes...
  "%PY%" -m pip install -r requirements.txt || (echo Package install failed - see the message above. & pause & exit /b 1)
)
title FDCS - public website
echo.
echo  FDCS - making your website public (any phone, anywhere, with HTTPS)
echo  -------------------------------------------------------------------
"%PY%" launcher.py public
echo.
pause
