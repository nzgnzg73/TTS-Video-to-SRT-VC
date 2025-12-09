@echo off

REM --- Activate Virtual Environment ---
call venv\Scripts\activate

REM --- Start Server Accessible on LAN ---
start /B python app v4.py

REM --- Wait 3 seconds so server can start ---
timeout /t 3 >nul

REM --- Open Localhost URL ---
start http://127.0.0.1:8007/

REM --- ALSO open LAN IP URL for mobile use ---
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do set LOCALIP=%%a
start http://%LOCALIP%:8007/

pause