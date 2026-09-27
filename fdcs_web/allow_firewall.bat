@echo off
rem Lets phones on the same Wi-Fi / hotspot reach FDCS on port 8000.
rem Windows blocks this by default, especially on hotspots marked as "Public" networks.
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo Asking Windows for administrator permission...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
netsh advfirewall firewall delete rule name="FDCS web app (port 8000)" >nul 2>&1
netsh advfirewall firewall add rule name="FDCS web app (port 8000)" dir=in action=allow protocol=TCP localport=8000 profile=private,public,domain
if %errorlevel% equ 0 (
  echo.
  echo Done. Phones on the same Wi-Fi or hotspot can now open FDCS on port 8000.
  echo To undo later:  netsh advfirewall firewall delete rule name="FDCS web app (port 8000)"
) else (
  echo Could not add the firewall rule.
)
echo.
pause
