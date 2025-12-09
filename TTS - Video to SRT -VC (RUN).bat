@echo off
TITLE TTS - Video to SRT -VC



ECHO TTS - Video to SRT -VC





ECHO Starting the TTS - Video to SRT - VC......




echo The link I have given has to be opened in the browser. 

echo Link:-    http://127.0.0.1:8004
echo Link:-    http://localhost:8004



ECHO Your File is Running ((server_vc.py))
ECHO ---------------------------------------------------------------------------------------------------------------------------------
echo Contact Email: nzgnzg73@gmail.com
echo nzg73.blogspot.com
echo YouTube Channel: @NZG73
echo https://youtube.com/@nzg73
ECHO ---------------------------------------------------------------------------------------------------------------------------------
REM Activate the virtual environment. The 'CALL' command is important.
CALL .\venv\Scripts\activate

ECHO Virtual environment activated. Starting server...
ECHO To stop the server, press CTRL+C in this window.
ECHO.

REM Run the Python server script
python server_vc.py

ECHO Server has been stopped.
PAUSE