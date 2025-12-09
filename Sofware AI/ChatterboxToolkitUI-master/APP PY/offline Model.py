# local.py (Fixed for Safetensors/Folder Models)
import os
import sys
import torch
import soundfile as sf
import importlib
from flask import Flask, request, send_file, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import time

# --- Setup Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
chatterbox_src_path = os.path.join(script_dir, 'src')
models_dir = os.path.join(script_dir, 'models')

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

def get_vc_model(device_choice):
    global model_instance, current_device
    
    # 1. Check if 'models' directory exists
    if not os.path.exists(models_dir):
        raise RuntimeError(f"The 'models' folder is missing at: {models_dir}")

    # 2. Logic to find the correct path inside 'models'
    # اگر ماڈل ایک فولڈر کے اندر ہے (جیسا کہ آپ کے پاس conds.pt فولڈر لگ رہا ہے)
    # یا اگر ماڈل ڈائریکٹ models فولڈر میں ہے۔
    
    target_model_path = models_dir 
    
    # اگر models کے اندر کوئی سب فولڈر ہے تو اسے ٹارگٹ بنائیں
    subfolders = [f.path for f in os.scandir(models_dir) if f.is_dir()]
    if subfolders:
        # پہلا فولڈر اٹھا لیں (مثال: models/chatterbox-model/)
        target_model_path = subfolders[0]
        print(f"Detected model folder: {target_model_path}")

    # 3. Model Loading
    if model_instance is None or current_device != device_choice:
        print(f"Loading model from: {target_model_path} on {device_choice.upper()}...")
        
        if device_choice == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("GPU selected but CUDA not available!")
            
        if model_instance is not None:
            del model_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        try:
            # یہاں ہم 'from_pretrained' یا 'from_local' کے بجائے
            # سیدھا فولڈر پاتھ دے رہے ہیں کیونکہ safetensors کو فولڈر چاہیے ہوتا ہے۔
            model_instance = ChatterboxVC.from_local(target_model_path, device=device_choice)
            current_device = device_choice
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            # ایک اور کوشش: اگر models_dir ہی روٹ ہے
            try:
                print("Retrying with root models dir...")
                model_instance = ChatterboxVC.from_local(models_dir, device=device_choice)
                current_device = device_choice
                print("Model loaded successfully (Retry).")
            except Exception as e2:
                raise RuntimeError(f"Could not load model. Error 1: {e}, Error 2: {e2}")
        
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
    app.run(host='0.0.0.0', port=5004, debug=True)
