.
<div align="center">

# 🎙️ NZG73 Ultimate AI Audio Studio
### TTS • Voice Cloning • Transcriber • Audio Cleaner • Converter

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey?style=for-the-badge)](https://github.com/nzgnzg73)

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbmV5Y3AyZnB5Y3AyZnB5Y3AyZnB5Y3AyZnB5Y3AyZnB5Y3AyZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L2r39d73503nZc713w/giphy.gif" width="600" alt="Audio Waveform Animation">
</p>

**An all-in-one offline AI Studio for content creators.** Generate lifelike speech, clone voices, transcribe videos to SRT, remove noise from audio, and convert formats—all running locally on your machine.

[✨ Features](#-features) • [🚀 Installation](#-installation) • [📖 Usage](#-usage) • [⚙️ Configuration](#%EF%B8%8F-configuration) • [👤 Contact](#-contact)

</div>

---

## ✨ Features

This tool is packed with 5 powerful modules accessible from a single interface:

### 🗣️ 1. Advanced Text-to-Speech (TTS) & Voice Cloning
* **Multilingual Support:** Generate speech in **24+ languages** (English, Urdu, Hindi, Arabic, Chinese, French, etc.).
* **Voice Cloning:** Clone any voice using a short reference audio file (`.wav`/`.mp3`).
* **Long-Form Generation:** Automatic text splitting/chunking for audiobooks.
* **Customization:** Control Speed, Temperature, Pitch, and Exaggeration.
* **Background Mode:** Optimized for mobile; keeps generating even when the screen is off.

### 🔤 2. Audio/Video Transcriber (Whisper)
* **Video to SRT:** Convert video files directly to subtitles (`.srt`) and text.
* **High Accuracy:** Powered by OpenAI Whisper models (Tiny to Large-v3).
* **Translation:** Translate foreign audio into English subtitles automatically.
* **Hardware:** Supports **GPU (CUDA)** for blazing speed or CPU processing.

### 🔄 3. Voice-to-Voice Converter
* **Timbre Transfer:** Change the input voice to match a target speaker while keeping the emotion and intonation.
* **Real-time Logic:** Setup for AI voice changing workflows.

### 🎛️ 4. Audio Cleaner Pro (Offline)
* **Noise Reduction:** Remove background hiss, rumble, and static.
* **Silence Removal:** Auto-trim silent parts from recordings.
* **Enhancement:** Adjust pitch (Deep/Alien/Kid) and speed.

### 🛠️ 5. Audio Master Studio
* **Video to Audio:** Extract high-quality audio (MP3/WAV) from video files.
* **Format Converter:** Convert between MP3, WAV, AAC, OGG.
* **Recorder:** Built-in microphone recorder with visualizer.

---

## 🚀 Installation

Follow these steps to set up the studio on your local machine.

### Prerequisites
* **OS:** Windows 10/11 (64-bit) or Linux.
* **Python:** Version 3.10 or higher.
* **FFmpeg:** Required for audio processing. [Download Here](https://ffmpeg.org/download.html).
* **GPU (Optional):** NVIDIA GPU with CUDA recommended for faster processing.

### Step-by-Step Guide

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/nzgnzg73/your-repo-name.git](https://github.com/nzgnzg73/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you install the correct PyTorch version for your hardware from [pytorch.org](https://pytorch.org/))*

4.  **Install FFmpeg**
    * Download FFmpeg and add it to your System PATH.
    * Verify by running `ffmpeg -version` in CMD.

---

## 📖 Usage

### 1. Run the Server
Start the application by running the main Python script:

```bash
python app.py

Or if you used a specific filename: python main.py or uvicorn app:app --reload
2. Open the Interface
Once the server starts, open your web browser and navigate to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

(Or the URL displayed in your terminal)
3. How to Use Modules
| Module | Instructions |
|---|---|
| TTS | Enter text, select language/voice, click Generate. Use "Voice Cloning" to upload a reference audio. |
| Transcriber | Upload a Video/Audio file. Select "Task" (Transcribe/Translate). Click Start Processing. |
| Cleaner | Upload noisy audio. Adjust "Noise Reduction" slider. Click Process. |
| Converter | Go to "Audio Master Studio", upload video, select output format (MP3/WAV), and convert. |
📂 Project Structure
├── app.py                 # Main backend server (FastAPI/Flask)
├── templates/
│   └── index.html         # Main User Interface (The UI code)
├── static/
│   ├── styles.css         # Styling
│   └── script.js          # Frontend Logic
├── models/                # Folder for AI models (Whisper/XTTS)
├── output/                # Generated audio/SRT files saved here
├── requirements.txt       # Python dependencies
└── README.md              # Documentation

⚙️ Configuration & Tips
 * Mobile Usage: This tool is optimized for mobile.
   * Tip: On Android Chrome, go to Settings > Apps > Chrome > Battery and set to Unrestricted to allow background audio generation.
 * Models: On the first run, the app may download AI models (Whisper/Coqui). This requires an internet connection and may take a few minutes.
 * Zoom Lock: Use the 🔓 button in the bottom left to lock the screen zoom for a better app-like experience on mobile.
👤 Contact & Support
Created by NZG73.
 * 📺 YouTube: @NZG73
 * 🌐 Website: nzg73.blogspot.com
 * 📧 Email: nzgnzg73@gmail.com
<div align="center">
If you like this project, please give it a ⭐ Star on GitHub!
</div>

