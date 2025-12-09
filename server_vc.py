# File: server-vc.py
# Main FastAPI application for the TTS Server.
# Handles API requests for text-to-speech generation, UI serving,
# configuration management, and file uploads.
# NOW INTEGRATED WITH: Voice-to-Voice Converter and Audio Transcriber features.
# - Voice-to-Voice Converter: Runs as additional routes/endpoints within this FastAPI app.
# - Audio Transcriber: Runs as additional routes/endpoints within this FastAPI app.
# - On/Off toggles for Voice-to-Voice and Transcriber: These load/unload models in memory (e.g., GPU/CPU) without restarting the server.
# - Restart Button: Fully functional, restarts the entire app (including all integrated features).
# - Enable Multilingual: Fully functional, loads multilingual model for TTS without affecting other features.
# - All original TTS features (voice cloning, generation, etc.) remain intact.
# - Ports: TTS on configured port (e.g., 8004 from config), but since it's one app, no separate ports for converters/transcriber. Access via sub-paths like /vc/ and /transcribe/.
# - Models for VC and Transcriber: Loaded only when toggled ON, unloaded when OFF to free GPU/CPU.
# - No 404 errors: All routes properly defined.
# - Detailed Implementation: Ensured compatibility by converting Flask routes to FastAPI equivalents, using global states for model management, and handling dependencies carefully.

import os
import io
import logging
import logging.handlers  # For RotatingFileHandler
import shutil
import time
import uuid
import yaml  # For loading presets
import numpy as np
import librosa  # For potential direct use if needed, though utils.py handles most
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Literal
import webbrowser  # For automatic browser opening
import threading  # For automatic browser opening
import sys
import datetime
import traceback
import site
import torch
import soundfile as sf
import importlib
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel  # For Audio Transcriber
from werkzeug.utils import secure_filename  # یہ نئی لائن شامل کی ہے (Voice-to-Voice کے لیے)

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    File,
    UploadFile,
    Form,
    BackgroundTasks,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
    FileResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# --- Internal Project Imports ---
from config import (
    config_manager,
    get_host,
    get_port,
    get_log_file_path,
    get_output_path,
    get_reference_audio_path,
    get_predefined_voices_path,
    get_ui_title,
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
    get_gen_default_speed_factor,
    get_gen_default_language,
    get_audio_sample_rate,
    get_full_config_for_template,
    get_audio_output_format,
)

import engine  # TTS Engine interface
from models import (  # Pydantic models
    CustomTTSRequest,
    ErrorResponse,
    UpdateStatusResponse,
)
import utils  # Utility functions

from pydantic import BaseModel, Field


class OpenAISpeechRequest(BaseModel):
    model: str
    input_: str = Field(..., alias="input")
    voice: str
    response_format: Literal["wav", "opus", "mp3"] = "wav"  # Add "mp3"
    speed: float = 1.0
    seed: Optional[int] = None


# --- Logging Configuration ---
log_file_path_obj = get_log_file_path()
log_file_max_size_mb = config_manager.get_int("server.log_file_max_size_mb", 10)
log_backup_count = config_manager.get_int("server.log_file_backup_count", 5)

log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.handlers.RotatingFileHandler(
            str(log_file_path_obj),
            maxBytes=log_file_max_size_mb * 1024 * 1024,
            backupCount=log_backup_count,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Global Variables & Application Setup ---
startup_complete_event = threading.Event()  # For coordinating browser opening

# --- NEW: Global States for Integrated Features (Voice-to-Voice and Transcriber) ---
# These manage ON/OFF states without restarting the server.
vc_model_instance = None  # Voice-to-Voice model
vc_current_device = None
vc_is_enabled = False  # Toggle flag for Voice-to-Voice
tts_is_enabled = True   # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
transcriber_current_model = None  # Audio Transcriber model
transcriber_current_device = None
transcriber_current_model_name = None
transcriber_is_enabled = False  # Toggle flag for Transcriber
tts_is_enabled = True   # TTS آن/آف کے لیے
# --- Voice-to-Voice Converter Specific Config ---
HF_REPO_ID = "Nzgnzg73/chatterbox"
script_dir = os.path.dirname(os.path.abspath(__file__))
chatterbox_src_path = os.path.join(script_dir, 'src')
models_root_dir = os.path.join(script_dir, 'models')
target_model_dir = os.path.join(models_root_dir, HF_REPO_ID)

if chatterbox_src_path not in sys.path:
    sys.path.insert(0, chatterbox_src_path)

# Import Chatterbox Modules for Voice-to-Voice
try:
    import chatterbox.vc
    import chatterbox.models.s3gen
    import chatterbox.tts
    
    importlib.reload(chatterbox.models.s3gen)
    importlib.reload(chatterbox.vc)
    
    from chatterbox.vc import ChatterboxVC
except ImportError as e:
    logger.error(f"Error importing Voice-to-Voice modules: {e}")
    sys.exit(1)

# --- Audio Transcriber Specific Config ---
if os.name == 'nt':
    try:
        paths = site.getsitepackages()
        for p in paths:
            cudnn_bin = os.path.join(p, 'nvidia', 'cudnn', 'bin')
            if os.path.isdir(cudnn_bin):
                os.add_dll_directory(cudnn_bin)
                break
            cudnn_bin_v12 = os.path.join(p, 'nvidia', 'cudnn_cu12', 'bin')
            if os.path.isdir(cudnn_bin_v12):
                os.add_dll_directory(cudnn_bin_v12)
                break
    except Exception as e:
        logger.warning(f"⚠️ Warning: {e}")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_DATA_PATH = os.path.join(BASE_DIR, "Models", "Fast-Whisper-Small-Webui")
TRANSCRIBER_MODELS_DIR = MAIN_DATA_PATH
TRANSCRIBER_UPLOAD_FOLDER = os.path.join(MAIN_DATA_PATH, "Uploads")
TRANSCRIBER_OUTPUT_FOLDER = os.path.join(MAIN_DATA_PATH, "Outputs")
os.makedirs(TRANSCRIBER_MODELS_DIR, exist_ok=True)
os.makedirs(TRANSCRIBER_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIBER_OUTPUT_FOLDER, exist_ok=True)

# --- Helper Functions for Voice-to-Voice ---
def ensure_vc_model_exists():
    if not os.path.exists(target_model_dir) or not os.listdir(target_model_dir):
        logger.info(f"Voice-to-Voice model files not found. Starting download from: {HF_REPO_ID}...")
        try:
            os.makedirs(target_model_dir, exist_ok=True)
            snapshot_download(
                repo_id=HF_REPO_ID, 
                local_dir=target_model_dir,
                allow_patterns=["*.safetensors", "*.pt", "*.json", "conds/*"],
                ignore_patterns=["t3_*", "*.bin", "*.h5", "*adapter*"]
            )
            logger.info("Voice-to-Voice Model Download Complete!")
        except Exception as e:
            logger.error(f"Voice-to-Voice Download Failed: {e}")
            if os.path.exists(target_model_dir) and not os.listdir(target_model_dir):
                os.rmdir(target_model_dir)
            raise RuntimeError("Voice-to-Voice Model download failed. Check internet connection.")
    else:
        pass  # Model already exists

def get_vc_model(device_choice):
    global vc_model_instance, vc_current_device
    
    ensure_vc_model_exists()
    
    if vc_model_instance is None or vc_current_device != device_choice:
        logger.info(f"Loading Voice-to-Voice model on {device_choice.upper()}...")
        
        if device_choice == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("GPU selected but CUDA not available!")
            
        if vc_model_instance is not None:
            del vc_model_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        try:
            vc_model_instance = ChatterboxVC.from_local(target_model_dir, device=device_choice)
            vc_current_device = device_choice
            logger.info("Voice-to-Voice Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Voice-to-Voice model: {e}")
            raise e
        
    return vc_model_instance

# --- Helper Functions for Audio Transcriber ---
def format_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def get_transcriber_model(device_choice, model_name):
    global transcriber_current_model, transcriber_current_device, transcriber_current_model_name
    
    if (transcriber_current_model is not None and 
        transcriber_current_device == device_choice and 
        transcriber_current_model_name == model_name):
        return transcriber_current_model

    logger.info(f"🔄 Loading Transcriber Model: {model_name.upper()} on {device_choice.upper()}...")
    logger.info(f"📥 Downloading/Loading from: {TRANSCRIBER_MODELS_DIR}")
    
    if device_choice == "cuda":
        try:
            transcriber_model = WhisperModel(model_name, device="cuda", compute_type="int8", download_root=TRANSCRIBER_MODELS_DIR)
            logger.info("✅ GPU Test Passed!")
        except Exception as e:
            logger.error(f"❌ GPU Failed: {e}")
            raise RuntimeError(f"GPU Crash: {str(e)}")
    else:
        transcriber_model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=TRANSCRIBER_MODELS_DIR)
        
    transcriber_current_model = transcriber_model
    transcriber_current_device = device_choice
    transcriber_current_model_name = model_name
    
    logger.info(f"✅ Transcriber Model ({model_name}) Loaded Successfully!")
    return transcriber_current_model

# --- Delayed Browser Open (Original) ---
def _delayed_browser_open(host: str, port: int):
    try:
        startup_complete_event.wait(timeout=30)
        if not startup_complete_event.is_set():
            logger.warning("Server startup did not signal completion within timeout. Browser will not be opened automatically.")
            return

        time.sleep(1.5)
        display_host = "localhost" if host == "0.0.0.0" else host
        browser_url = f"http://{display_host}:{port}/"
        logger.info(f"Attempting to open web browser to: {browser_url}")
        webbrowser.open(browser_url)
    except Exception as e:
        logger.error(f"Failed to open browser automatically: {e}", exc_info=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    logger.info("TTS Server: Initializing application...")
    try:
        logger.info(f"Configuration loaded. Log file at: {get_log_file_path()}")

        paths_to_ensure = [
            get_output_path(),
            get_reference_audio_path(),
            get_predefined_voices_path(),
            Path("ui"),
            config_manager.get_path(
                "paths.model_cache", "./model_cache", ensure_absolute=True
            ),
        ]
        for p in paths_to_ensure:
            p.mkdir(parents=True, exist_ok=True)

        if not engine.load_model():
            logger.critical(
                "CRITICAL: TTS Model failed to load on startup. Server might not function correctly."
            )
        else:
            logger.info("TTS Model loaded successfully via engine.")
            
            # --- AUTO BROWSER OPENING DISABLED ---
            # host_address = get_host()
            # server_port = get_port()
            # browser_thread = threading.Thread(
            #     target=lambda: _delayed_browser_open(host_address, server_port),
            #     daemon=True,
            # )
            # browser_thread.start()
            # -------------------------------------

        logger.info("Application startup sequence complete.")
        startup_complete_event.set()
        yield
    except Exception as e_startup:
        logger.error(
            f"FATAL ERROR during application startup: {e_startup}", exc_info=True
        )
        startup_complete_event.set()
        yield
    finally:
        logger.info("TTS Server: Application shutdown sequence initiated...")
        # Unload integrated models on shutdown
        global vc_model_instance, transcriber_current_model
        if vc_model_instance is not None:
            del vc_model_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if transcriber_current_model is not None:
            del transcriber_current_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("Integrated models unloaded.")
        logger.info("TTS Server: Application shutdown complete.")


# --- FastAPI Application Instance ---
app = FastAPI(
    title=get_ui_title(),
    description="Text-to-Speech server with advanced UI and API capabilities, integrated with Voice-to-Voice Converter and Audio Transcriber.",
    version="2.0.2",  # Version Bump
    lifespan=lifespan,
)


## Restart Server Button 👇👇👇👇👇👇👇👇👇

@app.post("/api/restart", tags=["System"])
async def restart_engine_endpoint():
    """
    Restart the entire TTS - Video to SRT - VC safely, including all integrated features (Voice-to-Voice and Transcriber).
    """
    import asyncio

    logger.warning("⚠ Restart request received via /api/restart endpoint.")

    async def _delayed_restart():
        # Wait a bit so response can be sent, then restart the process
        await asyncio.sleep(2)
        python = sys.executable
        os.execl(python, python, *sys.argv)

    # Start restart in background
    asyncio.create_task(_delayed_restart())

    return JSONResponse({"status": "restarting", "message": "Server restarting..."})

## Restart Server Button End 👆👆👆👆👆👆👆👆👆


# --- NEW: Endpoints for Toggling Features ---
@app.post("/api/toggle_vc", tags=["System"])
async def toggle_vc_endpoint(enabled: bool = Form(...)):
    """
    Toggle Voice-to-Voice Converter ON/OFF.
    - ON: Loads model into GPU/CPU.
    - OFF: Unloads model from memory.
    """
    global vc_is_enabled, vc_model_instance
    if enabled:
        if not vc_is_enabled:
            try:
                # Load with default device (can be configured later)
                get_vc_model('cuda' if torch.cuda.is_available() else 'cpu')
                vc_is_enabled = True
                return JSONResponse({"status": "enabled", "message": "Voice-to-Voice Converter enabled and model loaded."})
            except Exception as e:
                logger.error(f"Failed to enable Voice-to-Voice: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load Voice-to-Voice model: {str(e)}")
    else:
        if vc_is_enabled:
            if vc_model_instance is not None:
                del vc_model_instance
                vc_model_instance = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            vc_is_enabled = False
            return JSONResponse({"status": "disabled", "message": "Voice-to-Voice Converter disabled and model unloaded."})
    return JSONResponse({"status": "no_change", "message": "Voice-to-Voice state unchanged."})

@app.post("/api/toggle_transcriber", tags=["System"])
async def toggle_transcriber_endpoint(enabled: bool = Form(...)):
    """
    Toggle Audio Transcriber ON/OFF.
    - ON: Loads model into GPU/CPU.
    - OFF: Unloads model from memory.
    """
    global transcriber_is_enabled, transcriber_current_model
    if enabled:
        if not transcriber_is_enabled:
            try:
                # Load with default device and model (can be configured)
                get_transcriber_model('cuda' if torch.cuda.is_available() else 'cpu', 'small')
                transcriber_is_enabled = True
                return JSONResponse({"status": "enabled", "message": "Audio Transcriber enabled and model loaded."})
            except Exception as e:
                logger.error(f"Failed to enable Transcriber: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load Transcriber model: {str(e)}")
    else:
        if transcriber_is_enabled:
            if transcriber_current_model is not None:
                del transcriber_current_model
                transcriber_current_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            transcriber_is_enabled = False
            return JSONResponse({"status": "disabled", "message": "Audio Transcriber disabled and model unloaded."})
    return JSONResponse({"status": "no_change", "message": "Transcriber state unchanged."})

# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
# یہ نیا فنکشن یہاں پیسٹ کرو
@app.post("/api/toggle_tts", tags=["System"])
async def toggle_tts_endpoint(enabled: bool = Form(...)):
    global tts_is_enabled
    if enabled and not tts_is_enabled:
        tts_is_enabled = True
        return JSONResponse({"status": "enabled", "message": "TTS فیچر آن ہو گیا"})
    elif not enabled and tts_is_enabled:
        tts_is_enabled = False
        return JSONResponse({"status": "disabled", "message": "TTS فیچر آف ہو گیا"})
    else:
        return JSONResponse({"status": "no_change", "message": "کوئی تبدیلی نہیں"})
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Static Files and HTML Templates ---
ui_static_path = Path(__file__).parent / "ui"
if ui_static_path.is_dir():
    app.mount("/ui", StaticFiles(directory=ui_static_path), name="ui_static_assets")
else:
    logger.warning(
        f"UI static assets directory not found at '{ui_static_path}'. UI may not load correctly."
    )

# This will serve files from 'ui_static_path/vendor' when requests come to '/vendor/*'
if (ui_static_path / "vendor").is_dir():
    app.mount(
        "/vendor", StaticFiles(directory=ui_static_path / "vendor"), name="vendor_files"
    )
else:
    logger.warning(
        f"Vendor directory not found at '{ui_static_path}' /vendor. Wavesurfer might not load."
    )

@app.get("/styles.css", include_in_schema=False)
async def get_main_styles():
    styles_file = ui_static_path / "styles.css"
    if styles_file.is_file():
        return FileResponse(styles_file)
    raise HTTPException(status_code=404, detail="styles.css not found")

@app.get("/script.js", include_in_schema=False)
async def get_main_script():
    script_file = ui_static_path / "script.js"
    if script_file.is_file():
        return FileResponse(script_file)
    raise HTTPException(status_code=404, detail="script.js not found")

outputs_static_path = get_output_path(ensure_absolute=True)
try:
    app.mount(
        "/outputs",
        StaticFiles(directory=str(outputs_static_path)),
        name="generated_outputs",
    )
except RuntimeError as e_mount_outputs:
    logger.error(
        f"Failed to mount /outputs directory '{outputs_static_path}': {e_mount_outputs}. "
        "Output files may not be accessible via URL."
    )

templates = Jinja2Templates(directory=str(ui_static_path))

# --- NEW: Mount Static Folders for Integrated Features ---
# For Voice-to-Voice UI (assuming 'ui' folder exists for it)
vc_ui_path = Path(__file__).parent / "ui_vc"  # Assume you have a separate 'ui_vc' folder for Voice-to-Voice HTML
if vc_ui_path.is_dir():
    app.mount("/vc/ui", StaticFiles(directory=vc_ui_path), name="vc_ui")
else:
    logger.warning("Voice-to-Voice UI directory not found. UI may not load.")

# For Transcriber UI (template_folder='ui' in original, but integrate into main UI)
transcriber_ui_path = Path(__file__).parent / "ui_transcriber"  # Assume separate folder if needed
if transcriber_ui_path.is_dir():
    app.mount("/transcribe/ui", StaticFiles(directory=transcriber_ui_path), name="transcriber_ui")
else:
    logger.warning("Transcriber UI directory not found. UI may not load.")

# --- API Endpoints ---

# --- Main UI Route (Original TTS UI) ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_web_ui(request: Request):
    """Serves the main web interface (index.html)."""
    logger.info("Request received for main UI page ('/').")
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e_render:
        logger.error(f"Error rendering main UI page: {e_render}", exc_info=True)
        return HTMLResponse(
            "<html><body><h1>Internal Server Error</h1><p>Could not load the TTS interface. "
            "Please check server logs for more details.</p></body></html>",
            status_code=500,
        )

# --- NEW: Voice-to-Voice UI Route ---
@app.get("/vc", response_class=HTMLResponse, include_in_schema=False)
async def vc_index(request: Request):
    if not vc_is_enabled:
        raise HTTPException(status_code=503, detail="Voice-to-Voice Converter is disabled.")
    try:
        return templates.TemplateResponse("vc_index.html", {"request": request})  # Assume vc_index.html in ui folder
    except:
        return HTMLResponse("<h1>Voice-to-Voice UI</h1><p>Load your UI here.</p>")

# --- NEW: Transcriber UI Route ---
@app.get("/transcribe", response_class=HTMLResponse, include_in_schema=False)
async def transcribe_index(request: Request):
    if not transcriber_is_enabled:
        raise HTTPException(status_code=503, detail="Audio Transcriber is disabled.")
    try:
        return templates.TemplateResponse("transcribe_index.html", {"request": request})  # Assume transcribe_index.html
    except:
        return HTMLResponse("<h1>Audio Transcriber UI</h1><p>Load your UI here.</p>")

# --- API Endpoint for Initial UI Data (Original) ---
@app.get("/api/ui/initial-data", tags=["UI Helpers"])
async def get_ui_initial_data():
    logger.info("Request received for /api/ui/initial-data.")
    try:
        full_config = get_full_config_for_template()
        reference_files = utils.get_valid_reference_files()
        predefined_voices = utils.get_predefined_voices()
        loaded_presets = []
        presets_file = ui_static_path / "presets.yaml"
        if presets_file.exists():
            with open(presets_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
                if isinstance(yaml_content, list):
                    loaded_presets = yaml_content
                else:
                    logger.warning(
                        f"Invalid format in {presets_file}. Expected a list, got {type(yaml_content)}."
                    )
        else:
            logger.info(
                f"Presets file not found: {presets_file}. No presets will be loaded for initial data."
            )

        initial_gen_result_placeholder = {
            "outputUrl": None,
            "filename": None,
            "genTime": None,
            "submittedVoiceMode": None,
            "submittedPredefinedVoice": None,
            "submittedCloneFile": None,
        }

        return {
            "config": full_config,
            "reference_files": reference_files,
            "predefined_voices": predefined_voices,
            "presets": loaded_presets,
            "initial_gen_result": initial_gen_result_placeholder,
        }
    except Exception as e:
        logger.error(f"Error preparing initial UI data for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to load initial data for UI."
        )

@app.post("/load_multilingual_model")
async def load_multilingual_model_endpoint():
    """Load the multilingual TTS model."""
    from engine import load_multilingual_model
    
    try:
        success = load_multilingual_model()
        if success:
            return {"status": "success", "message": "Multilingual model loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load multilingual model")
    except Exception as e:
        logger.error(f"Error in load_multilingual_model_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load multilingual model: {str(e)}")

# --- Configuration Management API Endpoints (Original) ---
@app.post("/save_settings", response_model=UpdateStatusResponse, tags=["Configuration"])
async def save_settings_endpoint(request: Request):
    logger.info("Request received for /save_settings.")
    try:
        partial_update = await request.json()
        if not isinstance(partial_update, dict):
            raise ValueError("Request body must be a JSON object for /save_settings.")
        logger.debug(f"Received partial config data to save: {partial_update}")

        if config_manager.update_and_save(partial_update):
            restart_needed = any(
                key in partial_update
                for key in ["server", "tts_engine", "paths", "model"]
            )
            message = "Settings saved successfully."
            if restart_needed:
                message += " A server restart may be required for some changes to take full effect."
            return UpdateStatusResponse(message=message, restart_needed=restart_needed)
        else:
            logger.error(
                "Failed to save configuration via config_manager.update_and_save."
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to save configuration file due to an internal error.",
            )
    except ValueError as ve:
        logger.error(f"Invalid data format for /save_settings: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(ve)}")
    except Exception as e:
        logger.error(f"Error processing /save_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings save: {str(e)}",
        )

@app.post(
    "/reset_settings", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def reset_settings_endpoint():
    """Resets the configuration in config.yaml back to hardcoded defaults."""
    logger.warning("Request received to reset all configurations to default values.")
    try:
        if config_manager.reset_and_save():
            logger.info("Configuration successfully reset to defaults and saved.")
            return UpdateStatusResponse(
                message="Configuration reset to defaults. Please reload the page. A server restart may be beneficial.",
                restart_needed=True,
            )
        else:
            logger.error("Failed to reset and save configuration via config_manager.")
            raise HTTPException(
                status_code=500, detail="Failed to reset and save configuration file."
            )
    except Exception as e:
        logger.error(f"Error processing /reset_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings reset: {str(e)}",
        )

@app.post(
    "/restart_server", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def restart_server_endpoint():
    """Attempts to trigger a server restart."""
    logger.info("Request received for /restart_server.")
    message = (
        "Server restart initiated. If running locally without a process manager, "
        "you may need to restart manually. For managed environments (Docker, systemd), "
        "the manager should handle the restart."
    )
    logger.warning(message)
    return UpdateStatusResponse(message=message, restart_needed=True)

# --- UI Helper API Endpoints (Original) ---
@app.get("/get_reference_files", response_model=List[str], tags=["UI Helpers"])
async def get_reference_files_api():
    """Returns a list of valid reference audio filenames (.wav, .mp3)."""
    logger.debug("Request for /get_reference_files.")
    try:
        return utils.get_valid_reference_files()
    except Exception as e:
        logger.error(f"Error getting reference files for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve reference audio files."
        )

@app.get(
    "/get_predefined_voices", response_model=List[Dict[str, str]], tags=["UI Helpers"]
)
async def get_predefined_voices_api():
    """Returns a list of predefined voices with display names and filenames."""
    logger.debug("Request for /get_predefined_voices.")
    try:
        return utils.get_predefined_voices()
    except Exception as e:
        logger.error(f"Error getting predefined voices for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve predefined voices list."
        )

# --- File Upload Endpoints (Original) ---
@app.post("/upload_reference", tags=["File Management"])
async def upload_reference_audio_endpoint(files: List[UploadFile] = File(...)):
    logger.info(f"Request to /upload_reference with {len(files)} file(s).")
    ref_path = get_reference_audio_path(ensure_absolute=True)
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append(
                {"filename": "Unknown", "error": "File received with no filename."}
            )
            logger.warning("Upload attempt with no filename.")
            continue

        safe_filename = utils.sanitize_filename(file.filename)
        destination_path = ref_path / safe_filename

        try:
            if not (
                safe_filename.lower().endswith(".wav")
                or safe_filename.lower().endswith(".mp3")
            ):
                raise ValueError("Invalid file type. Only .wav and .mp3 are allowed.")

            if destination_path.exists():
                logger.info(
                    f"Reference file '{safe_filename}' already exists. Skipping duplicate upload."
                )
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(
                f"Successfully saved uploaded reference file to: {destination_path}"
            )

            max_duration = config_manager.get_int(
                "audio_output.max_reference_duration_sec", 30
            )
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration
            )
            if not is_valid:
                logger.warning(
                    f"Uploaded file '{safe_filename}' failed validation: {validation_msg}. Deleting."
                )
                destination_path.unlink(missing_ok=True)
                upload_errors.append(
                    {"filename": safe_filename, "error": validation_msg}
                )
            else:
                uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    all_current_reference_files = utils.get_valid_reference_files()
    response_data = {
        "message": f"Processed {len(files)} file(s).",
        "uploaded_files": uploaded_filenames_successfully,
        "all_reference_files": all_current_reference_files,
        "errors": upload_errors,
    }
    status_code = (
        200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    )
    if upload_errors:
        logger.warning(
            f"Upload to /upload_reference completed with {len(upload_errors)} error(s)."
        )
    return JSONResponse(content=response_data, status_code=status_code)


@app.post("/upload_predefined_voice", tags=["File Management"])
async def upload_predefined_voice_endpoint(files: List[UploadFile] = File(...)):
    logger.info(f"Request to /upload_predefined_voice with {len(files)} file(s).")
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append(
                {"filename": "Unknown", "error": "File received with no filename."}
            )
            logger.warning("Upload attempt for predefined voice with no filename.")
            continue

        safe_filename = utils.sanitize_filename(file.filename)
        destination_path = predefined_voices_path / safe_filename

        try:
            if not (
                safe_filename.lower().endswith(".wav")
                or safe_filename.lower().endswith(".mp3")
            ):
                raise ValueError(
                    "Invalid file type. Only .wav and .mp3 are allowed for predefined voices."
                )

            if destination_path.exists():
                logger.info(
                    f"Predefined voice file '{safe_filename}' already exists. Skipping duplicate upload."
                )
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(
                f"Successfully saved uploaded predefined voice file to: {destination_path}"
            )
            # Basic validation (can be extended if predefined voices have specific requirements)
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration_sec=None
            )  # No duration limit for predefined
            if not is_valid:
                logger.warning(
                    f"Uploaded predefined voice '{safe_filename}' failed basic validation: {validation_msg}. Deleting."
                )
                destination_path.unlink(missing_ok=True)
                upload_errors.append(
                    {"filename": safe_filename, "error": validation_msg}
                )
            else:
                uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing predefined voice file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    all_current_predefined_voices = (
        utils.get_predefined_voices()
    )  # Fetches formatted list
    response_data = {
        "message": f"Processed {len(files)} predefined voice file(s).",
        "uploaded_files": uploaded_filenames_successfully,  # List of raw filenames uploaded
        "all_predefined_voices": all_current_predefined_voices,  # Formatted list for UI
        "errors": upload_errors,
    }
    status_code = (
        200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    )
    if upload_errors:
        logger.warning(
            f"Upload to /upload_predefined_voice completed with {len(upload_errors)} error(s)."
        )
    return JSONResponse(content=response_data, status_code=status_code)


# --- TTS Generation Endpoint (Original) ---


@app.post(
    "/tts",
    tags=["TTS Generation"],
    summary="Generate speech with custom parameters",
    responses={
        200: {
            "content": {"audio/wav": {}, "audio/opus": {}},
            "description": "Successful audio generation.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or input.",
        },
        404: {
            "model": ErrorResponse,
            "description": "Required resource not found (e.g., voice file).",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during generation.",
        },
        503: {
            "model": ErrorResponse,
            "description": "TTS engine not available or model not loaded.",
        },
    },
)
async def custom_tts_endpoint(
    request: CustomTTSRequest, background_tasks: BackgroundTasks
):
    if not tts_is_enabled:
        raise HTTPException(status_code=503, detail="TTS فیچر آف ہے۔ پہلے آن کریں۔")
    
    perf_monitor = utils.PerformanceMonitor(
        enabled=config_manager.get_bool("server.enable_performance_monitor", False)
    )
    perf_monitor.record("TTS request received")

    if not engine.MODEL_LOADED:
        logger.error("TTS request failed: Model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    logger.info(
        f"Received /tts request: mode='{request.voice_mode}', format='{request.output_format}'"
    )
    logger.debug(
        f"TTS params: seed={request.seed}, split={request.split_text}, chunk_size={request.chunk_size}"
    )
    logger.debug(f"Input text (first 100 chars): '{request.text[:100]}...'")

    audio_prompt_path_for_engine: Optional[Path] = None
    if request.voice_mode == "predefined":
        if not request.predefined_voice_id:
            raise HTTPException(
                status_code=400,
                detail="Missing 'predefined_voice_id' for 'predefined' voice mode.",
            )
        voices_dir = get_predefined_voices_path(ensure_absolute=True)
        potential_path = voices_dir / request.predefined_voice_id
        if not potential_path.is_file():
            logger.error(f"Predefined voice file not found: {potential_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Predefined voice file '{request.predefined_voice_id}' not found.",
            )
        audio_prompt_path_for_engine = potential_path
        logger.info(f"Using predefined voice: {request.predefined_voice_id}")

    elif request.voice_mode == "clone":
        if not request.reference_audio_filename:
            raise HTTPException(
                status_code=400,
                detail="Missing 'reference_audio_filename' for 'clone' voice mode.",
            )
        ref_dir = get_reference_audio_path(ensure_absolute=True)
        potential_path = ref_dir / request.reference_audio_filename
        if not potential_path.is_file():
            logger.error(
                f"Reference audio file for cloning not found: {potential_path}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Reference audio file '{request.reference_audio_filename}' not found.",
            )
        max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
        is_valid, msg = utils.validate_reference_audio(potential_path, max_dur)
        if not is_valid:
            raise HTTPException(
                status_code=400, detail=f"Invalid reference audio: {msg}"
            )
        audio_prompt_path_for_engine = potential_path
        logger.info(
            f"Using reference audio for cloning: {request.reference_audio_filename}"
        )

    perf_monitor.record("Parameters and voice path resolved")

    all_audio_segments_np: List[np.ndarray] = []
    final_output_sample_rate = (
        get_audio_sample_rate()
    )  # Target SR for the final output file
    engine_output_sample_rate: Optional[int] = (
        None  # SR from the TTS engine (e.g., 24000 Hz)
    )

    if request.split_text and len(request.text) > (
        request.chunk_size * 1.5 if request.chunk_size else 120 * 1.5
    ):
        chunk_size_to_use = (
            request.chunk_size if request.chunk_size is not None else 120
        )
        logger.info(f"Splitting text into chunks of size ~{chunk_size_to_use}.")
        text_chunks = utils.chunk_text_by_sentences(request.text, chunk_size_to_use)
        perf_monitor.record(f"Text split into {len(text_chunks)} chunks")
    else:
        text_chunks = [request.text]
        logger.info(
            "Processing text as a single chunk (splitting not enabled or text too short)."
        )

    if not text_chunks:
        raise HTTPException(
            status_code=400, detail="Text processing resulted in no usable chunks."
        )

    for i, chunk in enumerate(text_chunks):
        logger.info(f"Synthesizing chunk {i+1}/{len(text_chunks)}...")
        try:
            chunk_audio_tensor, chunk_sr_from_engine = engine.synthesize(
                text=chunk,
                audio_prompt_path=(
                    str(audio_prompt_path_for_engine)
                    if audio_prompt_path_for_engine
                    else None
                ),
                temperature=(
                    request.temperature
                    if request.temperature is not None
                    else get_gen_default_temperature()
                ),
                exaggeration=(
                    request.exaggeration
                    if request.exaggeration is not None
                    else get_gen_default_exaggeration()
                ),
                cfg_weight=(
                    request.cfg_weight
                    if request.cfg_weight is not None
                    else get_gen_default_cfg_weight()
                ),
                seed=(
                    request.seed if request.seed is not None else get_gen_default_seed()
                ),
                language=(
                    request.language
                    if request.language is not None
                    else get_gen_default_language()
                ),
            )

            perf_monitor.record(f"Engine synthesized chunk {i+1}")

            if chunk_audio_tensor is None or chunk_sr_from_engine is None:
                error_detail = f"TTS engine failed to synthesize audio for chunk {i+1}."
                logger.error(error_detail)
                raise HTTPException(status_code=500, detail=error_detail)

            if engine_output_sample_rate is None:
                engine_output_sample_rate = chunk_sr_from_engine
            elif engine_output_sample_rate != chunk_sr_from_engine:
                logger.warning(
                    f"Inconsistent sample rate from engine: chunk {i+1} ({chunk_sr_from_engine}Hz) "
                    f"differs from previous ({engine_output_sample_rate}Hz). Using first chunk's SR."
                )

            current_processed_audio_tensor = chunk_audio_tensor

            speed_factor_to_use = (
                request.speed_factor
                if request.speed_factor is not None
                else get_gen_default_speed_factor()
            )
            if speed_factor_to_use != 1.0:
                current_processed_audio_tensor, _ = utils.apply_speed_factor(
                    current_processed_audio_tensor,
                    chunk_sr_from_engine,
                    speed_factor_to_use,
                )
                perf_monitor.record(f"Speed factor applied to chunk {i+1}")

            # ### MODIFICATION ###
            # All other processing is REMOVED from the loop.
            # We will process the final concatenated audio clip.
            processed_audio_np = current_processed_audio_tensor.cpu().numpy().squeeze()
            all_audio_segments_np.append(processed_audio_np)

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e_chunk:
            error_detail = f"Error processing audio chunk {i+1}: {str(e_chunk)}"
            logger.error(error_detail, exc_info=True)
            raise HTTPException(status_code=500, detail=error_detail)

    if not all_audio_segments_np:
        logger.error("No audio segments were successfully generated.")
        raise HTTPException(
            status_code=500, detail="Audio generation resulted in no output."
        )

    if engine_output_sample_rate is None:
        logger.error("Engine output sample rate could not be determined.")
        raise HTTPException(
            status_code=500, detail="Failed to determine engine sample rate."
        )

    try:
        # ### MODIFICATION START ###
        # First, concatenate all raw chunks into a single audio clip.
        final_audio_np = (
            np.concatenate(all_audio_segments_np)
            if len(all_audio_segments_np) > 1
            else all_audio_segments_np[0]
        )
        perf_monitor.record("All audio chunks processed and concatenated")

        # Now, apply all audio processing to the COMPLETE audio clip.
        if config_manager.get_bool("audio_processing.enable_silence_trimming", False):
            final_audio_np = utils.trim_lead_trail_silence(
                final_audio_np, engine_output_sample_rate
            )
            perf_monitor.record(f"Global silence trim applied")

        if config_manager.get_bool(
            "audio_processing.enable_internal_silence_fix", False
        ):
            final_audio_np = utils.fix_internal_silence(
                final_audio_np, engine_output_sample_rate
            )
            perf_monitor.record(f"Global internal silence fix applied")

        if (
            config_manager.get_bool("audio_processing.enable_unvoiced_removal", False)
            and utils.PARSELMOUTH_AVAILABLE
        ):
            final_audio_np = utils.remove_long_unvoiced_segments(
                final_audio_np, engine_output_sample_rate
            )
            perf_monitor.record(f"Global unvoiced removal applied")
        # ### MODIFICATION END ###

    except ValueError as e_concat:
        logger.error(f"Audio concatenation failed: {e_concat}", exc_info=True)
        for idx, seg in enumerate(all_audio_segments_np):
            logger.error(f"Segment {idx} shape: {seg.shape}, dtype: {seg.dtype}")
        raise HTTPException(
            status_code=500, detail=f"Audio concatenation error: {e_concat}"
        )

    output_format_str = (
        request.output_format if request.output_format else get_audio_output_format()
    )

    encoded_audio_bytes = utils.encode_audio(
        audio_array=final_audio_np,
        sample_rate=engine_output_sample_rate,
        output_format=output_format_str,
        target_sample_rate=final_output_sample_rate,
    )
    perf_monitor.record(
        f"Final audio encoded to {output_format_str} (target SR: {final_output_sample_rate}Hz from engine SR: {engine_output_sample_rate}Hz)"
    )

    if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
        logger.error(
            f"Failed to encode final audio to format: {output_format_str} or output is too small ({len(encoded_audio_bytes or b'')} bytes)."
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode audio to {output_format_str} or generated invalid audio.",
        )

    media_type = f"audio/{output_format_str}"
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    suggested_filename_base = f"tts_output_{timestamp_str}"
    download_filename = utils.sanitize_filename(
        f"{suggested_filename_base}.{output_format_str}"
    )
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}

    logger.info(
        f"Successfully generated audio: {download_filename}, {len(encoded_audio_bytes)} bytes, type {media_type}."
    )
    logger.debug(perf_monitor.report())

    # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
    # اب دونوں ماڈلز کے لیے فوراً آؤٹ پٹ + صحیح .wav فائل
    if hasattr(engine, 'current_model_type') and engine.current_model_type and ('mtl' in str(engine.current_model_type).lower() or 'multi' in str(engine.current_model_type).lower()):
        output_filename = f"tts_multi_{int(time.time())}.wav"
    else:
        output_filename = f"tts_output_{int(time.time())}.wav"

    output_path = get_output_path() / output_filename
    with open(output_path, "wb") as f:
        f.write(encoded_audio_bytes)

    # اہم: Content-Type کو واپس audio/wav کر دو تاکہ براؤزر اسے WAV سمجھے
    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=output_filename
    )
    # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech_endpoint(request: OpenAISpeechRequest):
    # Determine the audio prompt path based on the voice parameter
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    reference_audio_path = get_reference_audio_path(ensure_absolute=True)
    voice_path_predefined = predefined_voices_path / request.voice
    voice_path_reference = reference_audio_path / request.voice

    if voice_path_predefined.is_file():
        audio_prompt_path = voice_path_predefined
    elif voice_path_reference.is_file():
        audio_prompt_path = voice_path_reference
    else:
        raise HTTPException(
            status_code=404, detail=f"Voice file '{request.voice}' not found."
        )

    # Check if the TTS model is loaded
    if not engine.MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    try:
        # Use the provided seed or the default
        seed_to_use = (
            request.seed if request.seed is not None else get_gen_default_seed()
        )

        # Synthesize the audio
        audio_tensor, sr = engine.synthesize(
            text=request.input_,
            audio_prompt_path=str(audio_prompt_path),
            temperature=get_gen_default_temperature(),
            exaggeration=get_gen_default_exaggeration(),
            cfg_weight=get_gen_default_cfg_weight(),
            seed=seed_to_use,
        )

        if audio_tensor is None or sr is None:
            raise HTTPException(
                status_code=500, detail="TTS engine failed to synthesize audio."
            )

        # Apply speed factor if not 1.0
        if request.speed != 1.0:
            audio_tensor, _ = utils.apply_speed_factor(audio_tensor, sr, request.speed)

        # Convert tensor to numpy array
        audio_np = audio_tensor.cpu().numpy()

        # Ensure it's 1D
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze()

        # Encode the audio to the requested format
        encoded_audio = utils.encode_audio(
            audio_array=audio_np,
            sample_rate=sr,
            output_format=request.response_format,
            target_sample_rate=get_audio_sample_rate(),
        )

        if encoded_audio is None:
            raise HTTPException(status_code=500, detail="Failed to encode audio.")

        # Determine the media type
        media_type = f"audio/{request.response_format}"

        # Return the streaming response
        return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

    except Exception as e:
        logger.error(f"Error in openai_speech_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW: Voice-to-Voice Converter Endpoint (Converted from Flask to FastAPI) ---

# ---  Voice-to-Voice Converter ---
@app.post("/process_vc", tags=["Voice-to-Voice"])
async def process_voice_conversion(
    source_audio: UploadFile = File(...),
    target_audio: UploadFile = File(...),
    device: str = Form("cpu")
):
    if not vc_is_enabled:
        raise HTTPException(status_code=503, detail="Voice-to-Voice Converter is disabled. Enable it first.")
    
    try:
        # فولڈر بناؤ
        os.makedirs("temp_uploads", exist_ok=True)
        source_path = os.path.join("temp_uploads", secure_filename(source_audio.filename))
        target_path = os.path.join("temp_uploads", secure_filename(target_audio.filename))

        # فائلیں سیو کرو
        source_content = await source_audio.read()
        target_content = await target_audio.read()
        with open(source_path, "wb") as f:
            f.write(source_content)
        with open(target_path, "wb") as f:
            f.write(target_content)

        # ماڈل لوڈ کرو — اگر نہیں ملا تو خود ڈاؤن لوڈ کرے گا
        model = get_vc_model(device)

        # بالکل محفوظ طریقہ — کوئی پیرامیٹر نہیں، بس دو فائلیں
        wav = model.generate(source_path, target_voice_path=target_path)

        # WAV فائل خود سیو کرو (soundfile سے)
        os.makedirs("outputs", exist_ok=True)
        output_filename = f"vc_output_{int(time.time())}.wav"
        output_path = os.path.join("outputs", output_filename)

        # wav tensor ہو یا numpy، دونوں کے لیے کام کرے گا
        if hasattr(wav, 'cpu'):
            wav_np = wav.cpu().numpy().squeeze()
        else:
            wav_np = np.array(wav).squeeze()

        import soundfile as sf
        sf.write(output_path, wav_np, 24000)

        # صفائی
        try:
            os.remove(source_path)
            os.remove(target_path)
        except:
            pass

        return FileResponse(output_path, media_type="audio/wav", filename=output_filename)

    except Exception as e:
        logger.error(f"Voice-to-Voice Error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice Conversion Failed: {str(e)}")
# --- END Voice-to-Voice Converter ---
# --- NEW: Audio Transcriber Endpoint (Converted from Flask to FastAPI) ---
# --- Audio Video to SRT Transcriber --- 
@app.post("/transcribe", tags=["Audio Transcriber"])
async def transcribe(
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    device: str = Form("cpu"),
    model: str = Form("small"),
    language: str = Form("ur"),
    task: str = Form("transcribe"),
    target_language: Optional[str] = Form(None)
):
    if not transcriber_is_enabled:
        raise HTTPException(status_code=503, detail="Audio Transcriber is disabled. Enable it first.")

    # اگر YouTube URL دیا ہو تو اسے ڈاؤن لوڈ کرو (yt-dlp استعمال کرو)
    if youtube_url and youtube_url.strip():
        import yt_dlp
        import tempfile
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': 'temp_yt_audio.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'quiet': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                filepath = "temp_yt_audio.wav"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"YouTube download failed: {str(e)}")
    else:
        if not file or file.filename == '':
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # فائل سیو کرو
        content = await file.read()
        filepath = os.path.join(TRANSCRIBER_UPLOAD_FOLDER, file.filename)
        with open(filepath, "wb") as f:
            f.write(content)

    try:
        transcriber_model = get_transcriber_model(device, model)
        
        # Task کے مطابق فیصلہ کرو
        if task == "translate":
            segments, info = transcriber_model.transcribe(
                filepath, 
                beam_size=5, 
                language=language if language != "auto" else None,
                task="translate",
                target_language=target_language
            )
        else:
            segments, info = transcriber_model.transcribe(
                filepath, 
                beam_size=5, 
                language=language if language != "auto" else None,
                task="transcribe"
            )

        srt_content = ""
        text_content = ""
        segment_id = 1
        for segment in segments:
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            srt_content += f"{segment_id}\n{start} --> {end}\n{text}\n\n"
            text_content += text + " "
            segment_id += 1

        # SRT فائل سیو کرو
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        srt_path = os.path.join(TRANSCRIBER_OUTPUT_FOLDER, f"{base_name}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        # صفائی
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists("temp_yt_audio.wav"):
                os.remove("temp_yt_audio.wav")
        except:
            pass

        return JSONResponse({
            "message": "Success",
            "srt_content": srt_content,
            "text_content": text_content.strip()
        })

    except Exception as e:
        logger.error(f"Transcriber Error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing Failed: {str(e)}")
# --- END Audio Video to SRT Transcriber ---    
# --- Main Execution ---
if __name__ == "__main__":
    server_host = get_host()
    server_port = get_port()

    logger.info(f"Starting Integrated TTS Server on http://{server_host}:{server_port}")
    logger.info(
        f"API documentation at http://{server_host}:{server_port}/docs"
    )
    logger.info(f"Web UI at http://{server_host}:{server_port}/")
    logger.info(f"Voice-to-Voice at http://{server_host}:{server_port}/vc (after enabling)")
    logger.info(f"Audio Transcriber at http://{server_host}:{server_port}/transcribe (after enabling)")

    import uvicorn

    uvicorn.run(
        "server_vc:app",  # Note: File name is server-vc.py, so module is server_vc
        host=server_host,
        port=server_port,
        log_level="info",
        workers=1,
        reload=False,
    )
