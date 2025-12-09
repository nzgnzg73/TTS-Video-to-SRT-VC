@echo off
TITLE TTS Server



ECHO TTS Server




ECHO Starting the TTS Server...



echo The link I have given has to be opened in the browser. 

echo Link:-    http://127.0.0.1:8004
echo Link:-    http://localhost:8004



ECHO Your File is Running ((server.py)).



REM Activate the virtual environment. The 'CALL' command is important.
CALL .\venv\Scripts\activate

ECHO Virtual environment activated. Starting server...
ECHO To stop the server, press CTRL+C in this window.
ECHO.

REM Run the Python server script
python server.py

ECHO Server has been stopped.
PAUSE