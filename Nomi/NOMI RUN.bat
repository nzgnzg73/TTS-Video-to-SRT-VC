@echo off
title Nomi - System Manager
color 0A
cd /d "%~dp0"

echo.
echo  ====================================
echo       NOMI - System Manager
echo  ====================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed!
    echo Please install Python from python.org
    pause
    exit
)

:: Run Nomi
echo Starting Nomi...
echo.
python Nomi.py

echo.
echo Server has been stopped.
pause