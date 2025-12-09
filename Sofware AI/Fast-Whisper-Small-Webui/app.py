# 🌟 Updated app.py (Portable Models & Custom Folder Structure)
# .............................................................................................
# 🇬🇧 Location: This file is located in the ROOT folder.
# 🇵🇰 Urdu: یہ فائل مین (Root) فولڈر میں موجود ہے۔

import os
import sys
import datetime
import traceback
import site
from flask import Flask, request, jsonify, render_template
from faster_whisper import WhisperModel

# 🌟 1. UNIVERSAL GPU FIX
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
        print(f"⚠️ Warning: {e}")

# 🌟 2. PORTABLE FOLDER SETUP (اہم تبدیلی) 🌟
# .............................................................................................
# یہ کوڈ اس بات کو یقینی بناتا ہے کہ ماڈلز C ڈرائیو میں نہیں، بلکہ یہیں ڈاؤنلوڈ ہوں۔
# یہ خود بخود "Models/Fast-Whisper-Small-Webui" کا فولڈر بنا دے گا۔

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# مین فولڈر کا راستہ
MAIN_DATA_PATH = os.path.join(BASE_DIR, "Models", "Fast-Whisper-Small-Webui")

# سب-فولڈرز (Uploads اور Outputs کو بھی اسی کے اندر رکھا ہے)
MODELS_DIR = MAIN_DATA_PATH  # ماڈلز یہاں سیو ہوں گے
UPLOAD_FOLDER = os.path.join(MAIN_DATA_PATH, "Uploads")
OUTPUT_FOLDER = os.path.join(MAIN_DATA_PATH, "Outputs")

# فولڈرز بنائیں (اگر نہیں ہیں)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"📂 Data Directory: {MAIN_DATA_PATH}")

# 🌟 Setup Flask
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = Flask(__name__, template_folder='ui')

# 🌟 Global Variables
current_model = None
current_device = None
current_model_name = None

# --- Function to Switch/Load Model ---
def get_model(device_choice, model_name):
    global current_model, current_device, current_model_name
    
    if (current_model is not None and 
        current_device == device_choice and 
        current_model_name == model_name):
        return current_model

    print(f"🔄 Loading Model: {model_name.upper()} on {device_choice.upper()}...")
    print(f"📥 Downloading/Loading from: {MODELS_DIR}")
    
    if device_choice == "cuda":
        # 🌟 GPU Mode
        try:
            # download_root=MODELS_DIR وہ جادو ہے جو ماڈل کو آپ کے فولڈر میں رکھتا ہے
            model = WhisperModel(model_name, device="cuda", compute_type="int8", download_root=MODELS_DIR)
            print("✅ GPU Test Passed!")
        except Exception as e:
            print(f"❌ GPU Failed: {e}")
            raise RuntimeError(f"GPU Crash: {str(e)}")
    else:
        # 🌟 CPU Mode
        model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=MODELS_DIR)
        
    current_model = model
    current_device = device_choice
    current_model_name = model_name
    
    print(f"✅ Model ({model_name}) Loaded Successfully!")
    return current_model

# --- Helper Functions ---
def format_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    device_choice = request.form.get('device', 'cpu') 
    model_choice = request.form.get('model', 'small')
    lang_choice = request.form.get('language', 'ur')

    if lang_choice == "auto":
        lang_choice = None

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        # فائل کو نئے Uploads فولڈر میں سیو کریں
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            model = get_model(device_choice, model_choice)
            segments, info = model.transcribe(filepath, beam_size=5, language=lang_choice)

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

            # فائل کو نئے Outputs فولڈر میں سیو کریں
            base_name = os.path.splitext(file.filename)[0]
            with open(os.path.join(OUTPUT_FOLDER, f"{base_name}.srt"), "w", encoding="utf-8") as f:
                f.write(srt_content)

            # صفائی (صرف اپلوڈ کی ہوئی فائل ڈیلیٹ کریں، ماڈل یا رزلٹ نہیں)
            try:
                os.remove(filepath)
            except:
                pass

            return jsonify({
                "message": "Success",
                "srt_content": srt_content,
                "text_content": text_content.strip()
            })

        except RuntimeError as re:
            return jsonify({"error": str(re) + " -> Please Switch to CPU."}), 500
            
        except Exception as e:
            print("Server Crash Error Details:")
            traceback.print_exc()
            error_msg = str(e)
            if "cudnn" in error_msg.lower() or "dll" in error_msg.lower():
                 return jsonify({"error": "GPU Error: Missing Libraries. Please run: pip install nvidia-cudnn-cu12"}), 500
            
            return jsonify({"error": f"Processing Failed: {error_msg}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2188, debug=True, threaded=False)
