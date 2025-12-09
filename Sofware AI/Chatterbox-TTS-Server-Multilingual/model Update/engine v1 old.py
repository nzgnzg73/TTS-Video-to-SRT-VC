# File: engine v1 old_offline.py
# Core TTS model loading and speech generation logic (offline-patched).
import os
os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
import logging
import random
import numpy as np
import torch
import gc
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from typing import Optional, Tuple
from pathlib import Path
from chatterbox.tts import ChatterboxTTS  # Main TTS engine class
from chatterbox.models.s3gen.const import S3GEN_SR  # Default sample rate from the engine

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

from safetensors.torch import load_file as load_safetensors
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.mtl_tts import Conditionals, SUPPORTED_LANGUAGES

class PatchedChatterboxTTS(ChatterboxMultilingualTTS):
    """Patched version fixing attention issue"""
    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'PatchedChatterboxTTS':
        print("🚀 Using PatchedChatterboxTTS.from_local to load the model.")
        ckpt_dir = Path(ckpt_dir)

        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True))
        ve.to(device).eval()

        from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
        from transformers import LlamaConfig, LlamaModel

        class PatchedT3(T3):
            def __init__(self, hp=None):
                super().__init__(hp)
                cfg = self.cfg
                cfg._attn_implementation = "eager"
                self.tfmr = LlamaModel(cfg)

        t3 = PatchedT3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_23lang.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", weights_only=True))
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(str(ckpt_dir / "mtl_tokenizer.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)


# --- Global Module Variables ---
multilingual_model: Optional[PatchedChatterboxTTS] = None
MULTILINGUAL_MODEL_LOADED: bool = False
chatterbox_model: Optional[ChatterboxTTS] = None
MODEL_LOADED: bool = False
model_device: Optional[str] = None


def set_seed(seed_value: int):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")


def _test_cuda_functionality() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        test_tensor = torch.tensor([1.0]).cuda().cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA functionality test failed: {e}")
        return False


def _test_mps_functionality() -> bool:
    if not torch.backends.mps.is_available():
        return False
    try:
        test_tensor = torch.tensor([1.0]).to("mps").cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS functionality test failed: {e}")
        return False


def load_model() -> bool:
    global chatterbox_model, MODEL_LOADED, model_device

    if MODEL_LOADED:
        logger.info("TTS model is already loaded.")
        return True

    try:
        device_setting = config_manager.get_string("tts_engine.device", "auto")

        if device_setting == "auto":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA functionality test passed. Using CUDA.")
            elif _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS functionality test passed. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.info("CUDA and MPS not functional or not available. Using CPU.")
        elif device_setting == "cuda":
            resolved_device_str = "cuda" if _test_cuda_functionality() else "cpu"
        elif device_setting == "mps":
            resolved_device_str = "mps" if _test_mps_functionality() else "cpu"
        else:
            resolved_device_str = "cpu"

        model_device = resolved_device_str
        logger.info(f"Final device selection: {model_device}")

        model_repo_id_config = config_manager.get_string("model.repo_id", "ResembleAI/chatterbox")

        logger.info(f"Attempting offline model load (expected local path).")

        # --- OFFLINE PATCH START ---
        try:
            print("🟢 Offline Mode: Loading ChatterboxTTS model from local directory...")
            local_model_path = Path(
                r"D:\Flie\Chatterbox-TTS-Server\Chatterbox-TTS-Server-Multilingual\models--ResembleAI--chatterbox\snapshots\05e904af2b5c7f8e482687a9d7336c5c824467d9"
            )
            chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=local_model_path, device=model_device)
            logger.info(f"Loaded model locally from: {local_model_path}")
        except Exception as e_off:
            logger.error(f"Offline model load failed: {e_off}", exc_info=True)
            chatterbox_model = None
            MODEL_LOADED = False
            return False
        # --- OFFLINE PATCH END ---

        MODEL_LOADED = True
        if chatterbox_model:
            logger.info(f"TTS Model loaded successfully on {model_device}. Engine sample rate: {chatterbox_model.sr} Hz.")
        else:
            logger.error("Model loading sequence completed, but chatterbox_model is None.")
            MODEL_LOADED = False
            return False

        return True

    except Exception as e:
        logger.error(f"Unexpected error during model loading: {e}", exc_info=True)
        chatterbox_model = None
        MODEL_LOADED = False
        return False


def load_multilingual_model() -> bool:
    global multilingual_model, MULTILINGUAL_MODEL_LOADED, model_device
    global chatterbox_model, MODEL_LOADED

    if MULTILINGUAL_MODEL_LOADED:
        logger.info("Multilingual TTS model already loaded.")
        return True

    if model_device is None:
        logger.error("Main model device not determined. Load main model first.")
        return False

    if chatterbox_model is not None:
        logger.info("Unloading standard model to free memory...")
        chatterbox_model = None
        MODEL_LOADED = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Standard model unloaded and memory cleared.")
        
                   # --- OFFLINE PATCH START ---
    try:
        logger.info(f"Loading multilingual model (PatchedChatterboxTTS) on {model_device}...")
        multilingual_model = PatchedChatterboxTTS.from_local(ckpt_dir=Path(
            r"D:\Flie\Chatterbox-TTS-Server\Chatterbox-TTS-Server-Multilingual\models--ResembleAI--chatterbox\snapshots\05e904af2b5c7f8e482687a9d7336c5c824467d9"
        ), device=model_device)
        chatterbox_model = multilingual_model
        MULTILINGUAL_MODEL_LOADED = True
        MODEL_LOADED = True
        logger.info("PatchedChatterboxTTS model loaded successfully for all languages.")
        return True
    except Exception as e:
        logger.error(f"Error loading multilingual model: {e}", exc_info=True)
        multilingual_model = None
        chatterbox_model = None
        MULTILINGUAL_MODEL_LOADED = False
        MODEL_LOADED = False
        return False
    
           # --- OFFLINE PATCH END ---

def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    global chatterbox_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        return None, None

    active_model = chatterbox_model

    try:
        if seed != 0:
            set_seed(seed)

        is_multilingual = isinstance(active_model, ChatterboxMultilingualTTS)

        if is_multilingual:
            logger.info(f"Synthesizing with multilingual model for language: {language}")
            wav_tensor = active_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                language_id=language,
            )
        else:
            logger.info("Synthesizing with standard English model.")
            wav_tensor = active_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        return wav_tensor, active_model.sr

    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        return None, None
