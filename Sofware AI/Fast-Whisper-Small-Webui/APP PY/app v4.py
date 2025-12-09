#app
import time
import logging
import os
import gradio as gr
from faster_whisper import WhisperModel
from languages import get_language_names, get_language_from_name
from subtitle_manager import Subtitle
from pathlib import Path
import psutil
import pynvml
from whisper_models import whisper_models
import re
import io
import datetime

logging.basicConfig(level=logging.INFO)
last_model = None
model = None

description = "faster-whisper is a reimplementation of OpenAI's Whisper model using CTranslate2, which is a fast inference engine for Transformer models."
article = "Read the [documentation here](https://github.com/SYSTRAN/faster-whisper)."
compute_types = [
    "auto", "default", "int8", "int8_float32",
    "int8_float16", "int8_bfloat16", "int16",
    "float16", "float32", "bfloat16"
]

# Output folder setup - نیا ایڈیشن
OUTPUT_FOLDER = "output_files"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def get_free_gpu_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return meminfo.free

def get_workers_count():
    try:
        memory = get_free_gpu_memory()
        logging.info("CUDA memory")
    except Exception:
        memory = psutil.virtual_memory().available
        logging.info("RAM memory")
        
    logging.info(f"memory:{memory/ 1_000_000_000} GB")
    workers = int(memory / 2_000_000_000)
    logging.info(f"workers:{workers}")
    return workers

# نیا فنکشن: فائل کو output فولڈر میں save کریں
def save_to_output_folder(content, filename, subfolder=""):
    """فائل کو output فولڈر میں save کریں"""
    try:
        if subfolder:
            folder_path = os.path.join(OUTPUT_FOLDER, subfolder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        else:
            folder_path = OUTPUT_FOLDER
        
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return file_path
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        return None

def extract_text_from_srt(srt_file):
    """SRT فائل سے صرف متن نکالیں (ٹائمر ہٹا کر)"""
    try:
        with open(srt_file.name, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # SRT فارمیٹ کو پراسس کریں
        lines = content.split('\n')
        plain_text = ""
        
        for line in lines:
            # ٹائمر لائنز کو چھوڑ دیں (جو numbers اور --> والی لائنز)
            if '-->' in line or line.strip().isdigit() or not line.strip():
                continue
            # صرف متن والی لائنز شامل کریں
            plain_text += line.strip() + " "
        
        return plain_text.strip()
    except Exception as e:
        return f"Error reading SRT file: {str(e)}"

def read_pdf_file(pdf_file):
    """PDF فائل کا متن پڑھیں"""
    try:
        # PyPDF2 استعمال کریں اگر انسٹال ہے
        try:
            import PyPDF2
            with open(pdf_file.name, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except ImportError:
            return "PDF پڑھنے کے لیے PyPDF2 انسٹال کریں: pip install PyPDF2"
    except Exception as e:
        return f"PDF فائل پڑھنے میں خرابی: {str(e)}"

def read_file_content(file):
    """فائل کا مواد پڑھیں - تمام فارمیٹس کے لیے"""
    try:
        file_extension = file.name.lower()
        
        # TXT فائلز
        if file_extension.endswith(('.txt', '.log', '.ini', '.cfg', '.sql', '.bat', '.cmd', '.sh')):
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()
        
        # SRT فائل - اصل مواد واپس کریں (ٹائمر نہ ہٹائیں)
        elif file_extension.endswith('.srt'):
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()
        
        # PDF فائل
        elif file_extension.endswith('.pdf'):
            return read_pdf_file(file)
        
        # HTML/XML فائلز
        elif file_extension.endswith(('.html', '.htm', '.xml')):
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()
        
        # JSON/YAML فائلز
        elif file_extension.endswith(('.json', '.yaml', '.yml')):
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()
        
        # CSV فائل
        elif file_extension.endswith('.csv'):
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()
        
        # Programming فائلز
        elif file_extension.endswith(('.php', '.js', '.css', '.c', '.cpp', '.java', '.py', '.rb', '.go', '.ts', '.kt', '.swift')):
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()
        
        # دیگر فائلز کے لیے
        else:
            try:
                # binary فائلز کے لیے
                with open(file.name, 'rb') as f:
                    content = f.read()
                    # اگر text ہے تو واپس کریں، ورنہ message
                    try:
                        return content.decode('utf-8')
                    except:
                        return f"Binary file: {file.name}\nSize: {len(content)} bytes\n(Text content cannot be displayed)"
            except Exception as e:
                return f"File reading error: {str(e)}"
                
    except Exception as e:
        return f"File processing error: {str(e)}"

def clean_text_special(text):
    """ٹیکسٹ سے special characters ہٹائیں"""
    if not text:
        return ""
    
    # Special characters جو ہٹانے ہیں
    special_chars = [',', '.', ':', "'", '"', ';', '-', '_', '>', '<', '\\', '•', '|', '`',
                    '~', '°', '^', '<', '>', '~', '•', ';', ':', "'", '"', '_']
    
    # ہر special character کو ہٹائیں
    for char in special_chars:
        text = text.replace(char, '')
    
    return text

def remove_extra_lines(text):
    """اضافی خالی لائنیں ہٹائیں"""
    if not text:
        return ""
    
    # 3 یا زیادہ newlines کو 2 newlines میں تبدیل کریں
    text = re.sub(r'\n{3,}', '\n\n', text)
    # شروع اور آخر کی سپیس ہٹائیں
    text = text.strip()
    
    return text

def text_to_file(text):
    """ٹیکسٹ کو فائل میں تبدیل کریں"""
    if not text:
        return None
    
    buffer = io.BytesIO()
    buffer.write(text.encode('utf-8'))
    buffer.seek(0)
    return buffer

def remove_timers_from_text(timed_text):
    """ٹائمر والے ٹیکسٹ سے ٹائمر ہٹائیں"""
    if not timed_text:
        return "", "0 words, 0 characters", "0 words, 0 characters"
    
    # ان پٹ کی گنتی
    input_words = len(timed_text.split())
    input_chars = len(timed_text.replace('\n', '').replace(' ', ''))
    input_count = f"📥 Input: {input_words} words, {input_chars} characters"
    
    # مختلف قسم کے ٹائمر پیٹرن - تمام ممکنہ فارمیٹس
    timer_patterns = [
        r'\d+\s*',  # نمبر (1, 2, 3, etc.)
        r'\d+:\d+:\d+,\d+\s*-->\s*\d+:\d+:\d+,\d+\s*',  # 00:00:00,000 --> 00:00:02,799
        r'\d+:\d+:\d+\.\d+\s*-->\s*\d+:\d+:\d+\.\d+\s*',  # 00:00:00.000 --> 00:00:02.799
        r'0x[0-9A-Fa-f]+\s*',  # 0x00, 0x02, 0x0A, etc.
        r'\d+:\d+\s*',  # 0:00, 0:02, 1:30, etc.
        r'\[\d+\.\d+s\s*-\s*\d+\.\d+s\]:?\s*',  # [12.34s - 56.78s]:
        r'\[\d+\s*\.\s*\d+\s*-\s*\d+\s*\.\s*\d+\]:?\s*',  # [12.34 - 56.78]:
        r'\b\d+\s*\.\s*\d+\s*',  # 12.34, 56.78
        r'\[\d+\s*\]\s*',  # [123]
        r'TIME:\s*\d+\.\d+\s*',  # TIME: 12.34
        r'\b\d+\.\d+\s*seconds?\s*',  # 12.34 seconds
        r'\b\d+\s*seconds?\s*',  # 12 seconds
        r'^:\s*',  # لائن کے شروع میں کولون
        r'^\s*:\s*',  # لائن کے شروع میں سپیس کے ساتھ کولون
    ]
    
    clean_text = timed_text
    
    # ہر پیٹرن کو ہٹائیں
    for pattern in timer_patterns:
        clean_text = re.sub(pattern, '', clean_text)
    
    # لائن کے شروع میں کولون ہٹائیں
    lines = clean_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # لائن کے شروع میں کولون ہٹائیں
        if line.startswith(':'):
            line = line[1:].strip()
        if line:  # صرف غیر خالی لائنز شامل کریں
            cleaned_lines.append(line)
    
    final_text = '\n'.join(cleaned_lines)
    
    # آؤٹ پٹ کی گنتی
    output_words = len(final_text.split())
    output_chars = len(final_text.replace('\n', '').replace(' ', ''))
    output_count = f"📤 Output: {output_words} words, {output_chars} characters"
    
    return final_text, input_count, output_count

def convert_to_format(text, format_type):
    """ٹیکسٹ کو مختلف فارمیٹس میں تبدیل کریں"""
    if not text:
        return None, "Please enter text first"
    
    try:
        buffer = io.BytesIO()
        
        if format_type == "TXT":
            buffer.write(text.encode('utf-8'))
            filename = "converted_text.txt"
        
        elif format_type == "HTML":
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Converted Text</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
        .content {{ white-space: pre-wrap; }}
    </style>
</head>
<body>
    <div class="content">{text}</div>
</body>
</html>"""
            buffer.write(html_content.encode('utf-8'))
            filename = "converted_text.html"
        
        elif format_type == "JSON":
            json_content = {"text": text, "word_count": len(text.split()), "char_count": len(text)}
            import json
            buffer.write(json.dumps(json_content, indent=2, ensure_ascii=False).encode('utf-8'))
            filename = "converted_text.json"
        
        elif format_type == "XML":
            xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
    <content>{text}</content>
    <metadata>
        <word_count>{len(text.split())}</word_count>
        <character_count>{len(text)}</character_count>
    </metadata>
</document>"""
            buffer.write(xml_content.encode('utf-8'))
            filename = "converted_text.xml"
        
        elif format_type == "CSV":
            lines = text.split('\n')
            csv_content = "Line Number,Content\n"
            for i, line in enumerate(lines, 1):
                csv_content += f'{i},"{line}"\n'
            buffer.write(csv_content.encode('utf-8'))
            filename = "converted_text.csv"
        
        else:
            # ڈیفالٹ TXT
            buffer.write(text.encode('utf-8'))
            filename = "converted_text.txt"
        
        buffer.seek(0)
        return buffer, filename, f"Successfully converted to {format_type}"
    
    except Exception as e:
        return None, "", f"Conversion error: {str(e)}"

def transcribe_webui_simple_progress(modelName, languageName, urlData, multipleFiles, microphoneData, task,
                                    chunk_length, compute_type, beam_size, vad_filter, min_silence_duration_ms,
                                    progress=gr.Progress()):
    global last_model
    global model

    progress(0, desc="Loading Audio..")
    logging.info(f"languageName:{languageName}")
    logging.info(f"urlData:{urlData}")
    logging.info(f"multipleFiles:{multipleFiles}")
    logging.info(f"microphoneData:{microphoneData}")
    logging.info(f"task: {task}")
    logging.info(f"chunk_length: {chunk_length}")

    if last_model == None or modelName != last_model:
        logging.info("first or new model")
        progress(0.1, desc="Loading Model..")
        model = None
        model = WhisperModel(modelName, device="auto",compute_type=compute_type, cpu_threads=os.cpu_count(),)
        print('loaded')
    else:
        logging.info("Model not changed")
    last_model = modelName

    srt_sub = Subtitle("srt")

    files = []
    if multipleFiles:
        files+=multipleFiles
    if urlData:
        files.append(urlData)
    if microphoneData:
        files.append(microphoneData)
    logging.info(files)

    languageName = None if languageName == "Automatic Detection" else get_language_from_name(languageName).code

    files_out = []
    transcription_text = ""
    segments_text = ""
    
    # نیا: Output folder for this transcription
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    transcription_folder = f"transcription_{timestamp}"
    
    for file in progress.tqdm(files, desc="Working..."):
        start_time = time.time()
        segments, info = model.transcribe(
            file,
            beam_size=beam_size,
            vad_filter=vad_filter,
            language=languageName,
            vad_parameters=dict(min_silence_duration_ms=min_silence_duration_ms),
            condition_on_previous_text=False,
            chunk_length=chunk_length,
        )

        file_name = Path(file).stem
        files_out_srt = srt_sub.write_subtitle(segments, file_name, modelName, progress)
        
        # Transcription کے لیے صرف متن
        plain_text = ""
        segments_list = list(segments)
        for segment in segments_list:
            plain_text += segment.text + " "
        
        transcription_text += plain_text.strip() + "\n\n"
        
        # Segments کے لیے ٹائمر کے ساتھ
        segments_detailed = ""
        for segment in segments_list:
            segments_detailed += f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}\n"
        segments_text += segments_detailed + "\n"
        
        # نیا: Save files to output folder
        if plain_text.strip():
            save_to_output_folder(plain_text.strip(), f"{file_name}_transcription.txt", transcription_folder)
        
        if segments_detailed.strip():
            save_to_output_folder(segments_detailed.strip(), f"{file_name}_segments.txt", transcription_folder)
        
        logging.info(print(f"transcribe: {time.time() - start_time} sec."))
        files_out += [files_out_srt]
    
    return files_out, transcription_text, segments_text, f"✅ Files saved to: {OUTPUT_FOLDER}/{transcription_folder}/"

# Text Cleaner functions - وہی رہیں گے
def clean_text_function(text):
    """ٹیکسٹ صاف کریں"""
    if not text:
        return "", "No text to clean"
    cleaned = clean_text_special(text)
    
    # نیا: Save to output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cleaned_text_{timestamp}.txt"
    save_to_output_folder(cleaned, filename, "text_cleaner")
    
    return cleaned, f"Text cleaned successfully! ✅ Saved as: {filename}"

def clean_and_trim(text):
    """ٹیکسٹ صاف کریں اور اضافی لائنیں ہٹائیں"""
    if not text:
        return "", "No text to clean"
    cleaned = clean_text_special(text)
    trimmed = remove_extra_lines(cleaned)
    
    # نیا: Save to output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cleaned_trimmed_{timestamp}.txt"
    save_to_output_folder(trimmed, filename, "text_cleaner")
    
    return trimmed, f"Text cleaned and trimmed! ✅ Saved as: {filename}"

def download_cleaned_text(text):
    """صاف شدہ متن ڈاؤن لوڈ کریں"""
    if not text:
        return None, "No text to download"
    file_buffer = text_to_file(text)
    return file_buffer, "cleaned_text.txt"

# نیا SRT سے متن نکالنے کا فنکشن
def process_srt_file(srt_file):
    """SRT فائل اپلوڈ کرنے اور اس سے متن نکالنے کے لیے"""
    if srt_file is None:
        return "Please upload an SRT file first."
    
    extracted_text = extract_text_from_srt(srt_file)
    
    # نیا: Save to output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"srt_extracted_{timestamp}.txt"
    save_to_output_folder(extracted_text, filename, "srt_files")
    
    return extracted_text, f"✅ Extracted text saved as: {filename}"

# نیا ٹائمر ہٹانے کا فنکشن
def remove_timers(timed_text_input):
    """ٹائمر والے ٹیکسٹ سے ٹائمر ہٹائیں"""
    if not timed_text_input:
        return "Please enter text with timers first.", "📥 Input: 0 words, 0 characters", "📤 Output: 0 words, 0 characters"
    
    clean_text, input_count, output_count = remove_timers_from_text(timed_text_input)
    
    # نیا: Save to output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"no_timers_{timestamp}.txt"
    save_to_output_folder(clean_text, filename, "timer_removed")
    
    return clean_text, input_count, output_count, f"✅ Saved as: {filename}"

# فائل کنورژن کا فنکشن
def convert_text_to_format(text_input, format_type):
    """ٹیکسٹ کو منتخب فارمیٹ میں تبدیل کریں"""
    if not text_input:
        return None, "Please enter text first"
    
    result, filename, status = convert_to_format(text_input, format_type)
    
    # نیا: Save to output folder
    if result:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"converted_{timestamp}.{format_type.lower()}"
        folder_path = os.path.join(OUTPUT_FOLDER, "converted_files")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        file_path = os.path.join(folder_path, save_filename)
        with open(file_path, 'wb') as f:
            f.write(result.getvalue())
        
        status += f" ✅ Saved as: {save_filename}"
    
    return result, status

# فائل پڑھنے کا فنکشن
def process_uploaded_file(uploaded_file):
    """اپلوڈ کی گئی فائل کا مواد پڑھیں"""
    if uploaded_file is None:
        return "Please upload a file first."
    
    content = read_file_content(uploaded_file)
    
    # نیا: Save to output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = Path(uploaded_file.name).stem
    filename = f"uploaded_{original_name}_{timestamp}.txt"
    save_to_output_folder(content, filename, "uploaded_files")
    
    return content, f"✅ File content saved as: {filename}"

# باقی functions وہی رہیں گے
def clear_input():
    return "", "📥 Input: 0 words, 0 characters", "📤 Output: 0 words, 0 characters"

def clear_file_input():
    return None, ""

def clear_converter_input():
    return "", "TXT"

def clear_text_cleaner():
    return "", ""

def copy_to_input(text):
    return text

def copy_to_converter_input(text):
    return text

def copy_cleaned_text(cleaned_text):
    return cleaned_text

with gr.Blocks(title="Fast Whisper WebUI", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# Fast Whisper WebUI")

    with gr.Accordion("📌 Click Here to Show / Hide Info", open=False):
        gr.Markdown("""
### Want to talk or ask something?
Just click the YouTube link below! You'll find my email there and can message me easily. 👇  
YouTube Channel: ***@nzg73***  
***https://youtube.com/@NZG73***   

### Contact Email 👇👇👀  
E-mail: ***nzgnzg73@gmail.com***  

### CPU / GPU Requirements (VIP Info)
### Model Best Name
CPU ***(small)*** – 461 MB  
CPU/GPU ***(medium)*** – 1.42 GB  
4 VRAM GPU – ***(Systran/faster-whisper-large-v1)*** – 3.09 GB
""")

    gr.Markdown(description)
    gr.Markdown(f"### 📁 Automatic Output Folder: `{OUTPUT_FOLDER}/`")

    
    with gr.Tab("Audio Transcription"):
        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(choices=whisper_models, value="small", label="Model", interactive=True)
                language_dropdown = gr.Dropdown(choices=sorted(get_language_names()), value="Urdu", label="Language", interactive=True)
                url_input = gr.Text(label="URL", info="(YouTube, etc.)", interactive=True)
                file_upload = gr.File(label="Upload Files", file_count="multiple")
                audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input Audio")
                
            with gr.Column():
                task_dropdown = gr.Dropdown(choices=["transcribe", "translate"], label="Task", value="transcribe", interactive=True)
                chunk_length_input = gr.Number(label='chunk_length', value=30, interactive=True)
                compute_type_dropdown = gr.Dropdown(label="compute_type", choices=compute_types, value="auto", interactive=True)
                beam_size_input = gr.Number(label='beam_size', value=5, interactive=True)
                vad_filter_checkbox = gr.Checkbox(label='vad_filter', info='Use vad_filter', value=True)
                min_silence_input = gr.Number(label='Vad min_silence_duration_ms', value=500, interactive=True)
        
        transcribe_btn = gr.Button("Transcribe Audio", variant="primary")
        
        with gr.Row():
            download_output = gr.File(label="Download SRT")
            transcription_output = gr.Textbox(label="Transcription", lines=6, show_copy_button=True)
            segments_output = gr.Textbox(label="Segments", lines=6, show_copy_button=True)
            save_status = gr.Textbox(label="Save Status", lines=2, interactive=False)
    
    # نیا ٹیب SRT فائل پروسیسنگ کے لیے
    with gr.Tab("SRT to Text"):
        gr.Markdown("### Upload SRT file to extract plain text")
        
        with gr.Row():
            with gr.Column():
                srt_upload = gr.File(label="Upload SRT File", file_count="single", file_types=[".srt"])
                extract_srt_btn = gr.Button("Extract Text from SRT", variant="primary")
            
            with gr.Column():
                srt_text_output = gr.Textbox(label="Extracted Text", lines=10, show_copy_button=True)
                srt_status = gr.Textbox(label="Status", lines=1, interactive=False)
    
    # نیا ٹیب ٹائمر ہٹانے کے لیے
    with gr.Tab("Remove Timers"):
        gr.Markdown("### Remove timers from text")
        gr.Markdown("Enter text with timers like: `0:00`, `0x00`, `[12.34s - 15.67s]`, etc.")
        
        with gr.Row():
            with gr.Column():
                timed_text_input = gr.Textbox(
                    label="Input Text with Timers", 
                    lines=8,
                    placeholder="Example:\n1\n00:00:00,000 --> 00:00:02,799\nمیرا نام محمد نمان ہے اور میں پاکستان میں رہتا ہوں",
                    show_copy_button=True
                )
                input_count_output = gr.Textbox(
                    label="Input Count",
                    lines=1,
                    interactive=False
                )
                with gr.Row():
                    remove_timers_btn = gr.Button("Remove Timers", variant="primary")
                    clear_input_btn = gr.Button("Clear Input")
                    copy_output_btn = gr.Button("Copy Output to Input")
            
            with gr.Column():
                clean_text_output = gr.Textbox(
                    label="Clean Text (Timers Removed)", 
                    lines=8,
                    show_copy_button=True
                )
                output_count_output = gr.Textbox(
                    label="Output Count",
                    lines=1,
                    interactive=False
                )
                timer_status = gr.Textbox(
                    label="Save Status",
                    lines=1,
                    interactive=False
                )
    
    # نیا ٹیب Text Cleaner کے لیے
    with gr.Tab("Text Cleaner"):
        gr.Markdown("### Clean Text - Remove Special Characters")
        gr.Markdown("**Characters that will be removed:** `, . : ' \" ; - _ > < \\ • | ` ~ ° ^`")
        
        with gr.Row():
            with gr.Column():
                cleaner_input = gr.Textbox(
                    label="Paste your text here:", 
                    lines=8,
                    placeholder="Paste text here...",
                    show_copy_button=True
                )
                
                with gr.Row():
                    clean_btn = gr.Button("Clean →", variant="primary")
                    trim_btn = gr.Button("Remove Extra Blank Lines", variant="secondary")
                    clear_input_btn_cleaner = gr.Button("Clear Input", variant="secondary")
            
            with gr.Column():
                cleaner_output = gr.Textbox(
                    label="Cleaned Output:", 
                    lines=8,
                    placeholder="Cleaned text will appear here...",
                    show_copy_button=True
                )
                
                cleaner_status = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False
                )
                
                with gr.Row():
                    copy_btn = gr.Button("Copy", variant="secondary")
                    download_btn = gr.Button("Download (.txt)", variant="secondary")
                    clear_output_btn = gr.Button("Clear Output", variant="secondary")
                
                download_file = gr.File(label="Download File", visible=False)
    
    # نیا ٹیب فائل کنورٹر کے لیے
    with gr.Tab("Text Converter"):
        gr.Markdown("### Convert text to different file formats")
        
        with gr.Row():
            with gr.Column():
                converter_text_input = gr.Textbox(
                    label="Text to Convert", 
                    lines=8,
                    placeholder="Enter text to convert to different formats...",
                    show_copy_button=True
                )
                format_dropdown = gr.Dropdown(
                    choices=["TXT", "HTML", "JSON", "XML", "CSV"],
                    value="TXT",
                    label="Select Output Format",
                    interactive=True
                )
                converter_status = gr.Textbox(
                    label="Status",
                    lines=1,
                    interactive=False
                )
                with gr.Row():
                    convert_btn = gr.Button("Convert Text", variant="primary")
                    clear_converter_btn = gr.Button("Clear Input")
                    copy_converter_btn = gr.Button("Copy Output to Input")
            
            with gr.Column():
                converted_output = gr.File(label="Download Converted File")
    
    # نیا ٹیب فائل اپلوڈر کے لیے
    with gr.Tab("File Uploader"):
        gr.Markdown("### Upload and read files")
        gr.Markdown("Supported formats: TXT, SRT, PDF, HTML, XML, JSON, CSV, LOG, INI, CFG, SQL, PHP, JS, CSS, C, CPP, JAVA, PY, etc.")
        
        with gr.Row():
            with gr.Column():
                file_uploader = gr.File(
                    label="Upload File",
                    file_count="single",
                    file_types=[
                        ".txt", ".srt", ".pdf", ".html", ".htm", ".xml", ".json", ".yaml", ".yml", ".csv", 
                        ".log", ".ini", ".cfg", ".sql", ".php", ".js", ".css", 
                        ".c", ".cpp", ".java", ".py", ".rb", ".go", ".ts", ".kt", 
                        ".swift", ".sh", ".bat", ".cmd"
                    ]
                )
                file_status = gr.Textbox(
                    label="Status",
                    lines=1,
                    interactive=False
                )
                with gr.Row():
                    read_file_btn = gr.Button("Read File", variant="primary")
                    clear_file_btn = gr.Button("Clear File")
            
            with gr.Column():
                file_content_output = gr.Textbox(
                    label="File Content", 
                    lines=10,
                    show_copy_button=True  # نیا: کاپی بٹن ایڈ کیا
                )
    
    # Event handlers for Text Cleaner
    clean_btn.click(
        fn=clean_text_function,
        inputs=[cleaner_input],
        outputs=[cleaner_output, cleaner_status]
    )
    
    trim_btn.click(
        fn=clean_and_trim,
        inputs=[cleaner_input],
        outputs=[cleaner_output, cleaner_status]
    )
    
    copy_btn.click(
        fn=copy_cleaned_text,
        inputs=[cleaner_output],
        outputs=[cleaner_output]
    )
    
    download_btn.click(
        fn=download_cleaned_text,
        inputs=[cleaner_output],
        outputs=[download_file, cleaner_status]
    )
    
    clear_input_btn_cleaner.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[cleaner_input, cleaner_status]
    )
    
    clear_output_btn.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[cleaner_output, cleaner_status]
    )
    
    # Other event handlers
    transcribe_btn.click(
        fn=transcribe_webui_simple_progress,
        inputs=[
            model_dropdown, language_dropdown, url_input, file_upload, 
            audio_input, task_dropdown, chunk_length_input, compute_type_dropdown,
            beam_size_input, vad_filter_checkbox, min_silence_input
        ],
        outputs=[download_output, transcription_output, segments_output, save_status]
    )
    
    extract_srt_btn.click(
        fn=process_srt_file,
        inputs=[srt_upload],
        outputs=[srt_text_output, srt_status]
    )
    
    remove_timers_btn.click(
        fn=remove_timers,
        inputs=[timed_text_input],
        outputs=[clean_text_output, input_count_output, output_count_output, timer_status]
    )
    
    read_file_btn.click(
        fn=process_uploaded_file,
        inputs=[file_uploader],
        outputs=[file_content_output, file_status]
    )
    
    clear_input_btn.click(
        fn=clear_input,
        inputs=[],
        outputs=[timed_text_input, input_count_output, output_count_output]
    )
    
    clear_file_btn.click(
        fn=clear_file_input,
        inputs=[],
        outputs=[file_uploader, file_status]
    )
    
    clear_converter_btn.click(
        fn=clear_converter_input,
        inputs=[],
        outputs=[converter_text_input, format_dropdown]
    )
    
    copy_output_btn.click(
        fn=copy_to_input,
        inputs=[clean_text_output],
        outputs=[timed_text_input]
    )
    
    copy_converter_btn.click(
        fn=copy_to_converter_input,
        inputs=[converter_text_input],
        outputs=[converter_text_input]
    )
    
    convert_btn.click(
        fn=convert_text_to_format,
        inputs=[converter_text_input, format_dropdown],
        outputs=[converted_output, converter_status]
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=get_workers_count())
    demo.launch(server_name="0.0.0.0", server_port=8007, share=False)
