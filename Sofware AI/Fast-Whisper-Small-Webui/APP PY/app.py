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
        model = WhisperModel(modelName, device="auto",compute_type=compute_type, cpu_threads=os.cpu_count(),)#device="auto", compute_type="float16"
        print('loaded')
    else:
        logging.info("Model not changed")
    last_model = modelName

    srt_sub = Subtitle("srt")
    # vtt_sub = Subtitle("vtt")
    # txt_sub = Subtitle("txt")

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
    vtt=""
    txt=""
    for file in progress.tqdm(files, desc="Working..."):

        start_time = time.time()
        segments, info = model.transcribe(
            file,
            beam_size=beam_size,
            vad_filter=vad_filter,
            language=languageName,
            vad_parameters=dict(min_silence_duration_ms=min_silence_duration_ms),
            # max_new_tokens=128,
            condition_on_previous_text=False,
            chunk_length=chunk_length,
            )

        file_name = Path(file).stem
        files_out_srt = srt_sub.write_subtitle(segments, file_name, modelName, progress)
        # txt = txt_sub.get_subtitle(segments, progress)
        logging.info(print(f"transcribe: {time.time() - start_time} sec."))
        files_out += [files_out_srt]
    
    return files_out, vtt, txt



demo = gr.Interface(
    fn=transcribe_webui_simple_progress,
    description=description,
    article=article,
    inputs=[
        gr.Dropdown(choices=whisper_models, value="distil-whisper/distil-large-v3.5-ct2", label="Model", info="Select whisper model", interactive = True,),
        gr.Dropdown(choices=["Automatic Detection"] + sorted(get_language_names()), value="Automatic Detection", label="Language", info="Select audio voice language", interactive = True,),
        gr.Text(label="URL", info="(YouTube, etc.)", interactive = True),
        gr.File(label="Upload Files", file_count="multiple"),
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input Audio"),
        gr.Dropdown(choices=["transcribe", "translate"], label="Task", value="transcribe", interactive = True),
        gr.Number(label='chunk_length',value=30, interactive = True),
        gr.Dropdown(label="compute_type", choices=compute_types, value="auto", interactive = True),
        gr.Number(label='beam_size',value=5, interactive = True),
        gr.Checkbox(label='vad_filter',info='Use vad_filter', value=True),
        gr.Number(label='Vad min_silence_duration_ms',value=500, interactive = True),
    ],
    outputs=[
        gr.File(label="Download"),
        gr.Text(label="Transcription"), 
        gr.Text(label="Segments"),
    ],
    title="Fast Whisper WebUI"
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=get_workers_count())
    demo.launch(share=True)

