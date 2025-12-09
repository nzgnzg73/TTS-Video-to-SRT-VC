# local.py (Download ONLY on Button Click)
import os
import sys
import torch
import soundfile as sf
import importlib
from flask import Flask, request, send_file, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import time

# --- Auto Download Library ---
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: 'huggingface_hub' library is missing.")
    print("Please run: pip install huggingface_hub")
    sys.exit(1)

# --- CONFIGURATION ---
HF_REPO_ID = "Nzgnzg73/chatterbox"

# --- Setup Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
chatterbox_src_path = os.path.join(script_dir, 'src')
models_root_dir = os.path.join(script_dir, 'models')
target_model_dir = os.path.join(models_root_dir, HF_REPO_ID)

if chatterbox_src_path not in sys.path:
    sys.path.insert(0, chatterbox_src_path)

# --- Import Chatterbox Modules ---
try:
    import chatterbox.vc
    import chatterbox.models.s3gen
    import chatterbox.tts
    
    importlib.reload(chatterbox.models.s3gen)
    importlib.reload(chatterbox.vc)
    
    from chatterbox.vc import ChatterboxVC
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- Flask App ---
app = Flask(__name__, static_folder='ui')

# --- Global Vars ---
model_instance = None
current_device = None

def ensure_model_exists():
    """ 
    یہ فنکشن صرف تب چلے گا جب بٹن دبایا جائے گا۔
    """
    if not os.path.exists(target_model_dir) or not os.listdir(target_model_dir):
        print(f"Model files not found. Starting download from: {HF_REPO_ID}...")
        print("Please wait, this will run only once...")
        
        try:
            os.makedirs(target_model_dir, exist_ok=True)
            snapshot_download(
                repo_id=HF_REPO_ID, 
                local_dir=target_model_dir,
                allow_patterns=["*.safetensors", "*.pt", "*.json", "conds/*"],
                ignore_patterns=["t3_*", "*.bin", "*.h5", "*adapter*"]
            )
            print("Download Complete!")
        except Exception as e:
            print(f"Download Failed: {e}")
            if os.path.exists(target_model_dir) and not os.listdir(target_model_dir):
                os.rmdir(target_model_dir)
            raise RuntimeError("Model download failed. Check internet connection.")
    else:
        # اگر ماڈل پہلے سے ہے تو چپ چاپ آگے بڑھ جائے گا
        pass

def get_vc_model(device_choice):
    global model_instance, current_device
    
    # 1. یہاں چیک کرے گا (صرف پروسیسنگ کے وقت)
    ensure_model_exists()

    # 2. ماڈل لوڈ
    if model_instance is None or current_device != device_choice:
        print(f"Loading model on {device_choice.upper()}...")
        
        if device_choice == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("GPU selected but CUDA not available!")
            
        if model_instance is not None:
            del model_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        try:
            model_instance = ChatterboxVC.from_local(target_model_dir, device=device_choice)
            current_device = device_choice
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e
        
    return model_instance

@app.route('/')
def index():
    return send_from_directory('ui', 'index.html')

@app.route('/process_vc', methods=['POST'])
def process_voice_conversion():
    try:
        if 'source_audio' not in request.files or 'target_audio' not in request.files:
            return "Both source and target audio files are required.", 400
            
        source_file = request.files['source_audio']
        target_file = request.files['target_audio']
        
        device = request.form.get('device', 'cpu')
        cfg_rate = float(request.form.get('cfg_rate', 0.5))
        sigma_min = float(request.form.get('sigma_min', 1e-06))

        os.makedirs("temp_uploads", exist_ok=True)
        source_path = os.path.join("temp_uploads", secure_filename(source_file.filename))
        target_path = os.path.join("temp_uploads", secure_filename(target_file.filename))
        source_file.save(source_path)
        target_file.save(target_path)

        # جب یہ فنکشن کال ہوگا، تب ہی ڈاؤنلوڈ چیک ہوگا
        model = get_vc_model(device)

        wav = model.generate(
            source_path,
            target_voice_path=target_path,
            inference_cfg_rate=cfg_rate,
            sigma_min=sigma_min
        )

        os.makedirs("outputs", exist_ok=True)
        output_filename = f"vc_output_{int(time.time())}.wav"
        output_path = os.path.join("outputs", output_filename)
        model.save_wav(wav, output_path)

        try:
            os.remove(source_path)
            os.remove(target_path)
        except:
            pass

        return send_file(output_path, mimetype="audio/wav", as_attachment=True, download_name=output_filename)

    except RuntimeError as e:
        return str(e), 500
    except Exception as e:
        print(f"Error: {e}")
        return f"Processing Error: {str(e)}", 500

if __name__ == '__main__':
    print("Starting server at http://127.0.0.1:5004")
    # یہاں سے وہ ڈاؤنلوڈ والی لائن ہٹا دی گئی ہے
    app.run(host='0.0.0.0', port=5004, debug=True)
