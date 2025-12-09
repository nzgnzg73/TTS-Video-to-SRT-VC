# File: engine_offline_v3.py
# Offline multilingual TTS engine (patched version) — v3
# This version includes an automatic checkpoint <-> model size adjuster for text embedding layers.
# ✅ It will attempt to slice or pad text embedding tensors so checkpoints with a different vocab size
#    can still be loaded (uses strict=False). Read printed logs to see what adjustments were made.

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
from chatterbox.tts import ChatterboxTTS
from chatterbox.models.s3gen.const import S3GEN_SR
from config import config_manager

logger = logging.getLogger(__name__)

from safetensors.torch import load_file as load_safetensors
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.mtl_tts import Conditionals, SUPPORTED_LANGUAGES

# ----------------------------
# Patched class with adjuster
# ----------------------------
class PatchedChatterboxTTS(ChatterboxMultilingualTTS):
    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'PatchedChatterboxTTS':
        print("🚀 Using PatchedChatterboxTTS.from_local to load the model (v3, with size-adjust).")
        ckpt_dir = Path(ckpt_dir)

        # --- Voice encoder ---
        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True))
        ve.to(device).eval()

        # --- Build patched T3 that forces eager attn implementation ---
        from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
        from transformers import LlamaConfig, LlamaModel

        class PatchedT3(T3):
            def __init__(self, hp=None):
                super().__init__(hp)
                cfg = self.cfg
                # ensure eager attention to avoid runtime errors on some systems
                cfg._attn_implementation = "eager"
                self.tfmr = LlamaModel(cfg)

        # instantiate model (uses library T3Config.multilingual())
        t3 = PatchedT3(T3Config.multilingual())

        # ------------------------------
        # Load checkpoint with safe-adjust
        # ------------------------------
        ckpt_file = ckpt_dir / "t3_mtl23ls_v2.safetensors"  # ✅ NEW model filename
        print(f"🔁 Loading checkpoint: {ckpt_file}")

        raw_state = load_safetensors(ckpt_file)
        if "model" in raw_state.keys():
            raw_state = raw_state["model"][0]

        # Convert to tensors if needed (safetensors normally returns tensors already)
        ckpt_state = {k: (v if torch.is_tensor(v) else torch.tensor(v)) for k, v in raw_state.items()}

        # Model's expected state dict (shapes we must match)
        model_state = t3.state_dict()

        # Keys that usually hold token embeddings / heads — adjust these if mismatch occurs
        check_keys = ["text_emb.weight", "text_head.weight"]

        for key in check_keys:
            if key in ckpt_state and key in model_state:
                ck = ckpt_state[key]
                mk = model_state[key]
                if ck.shape != mk.shape:
                    ck_rows, ck_cols = ck.shape
                    mk_rows, mk_cols = mk.shape
                    # If embedding dimension differs (cols), we cannot safely fix
                    if ck_cols != mk_cols:
                        print(f"⚠️ Column-dim mismatch for {key}: checkpoint {ck.shape}, model {mk.shape}. Skipping adjust.")
                        continue

                    if ck_rows > mk_rows:
                        # slice checkpoint (keep first tokens) — most common tokens reside early
                        print(f"ℹ️ Slicing {key}: {ck_rows} -> {mk_rows}")
                        ck = ck[:mk_rows].contiguous()
                    elif ck_rows < mk_rows:
                        # pad checkpoint with small random values
                        print(f"ℹ️ Padding {key}: {ck_rows} -> {mk_rows}")
                        pad_rows = mk_rows - ck_rows
                        # create small-random padding with same dtype/device
                        pad = torch.randn((pad_rows, ck_cols), dtype=ck.dtype) * 0.02
                        ck = torch.cat([ck, pad], dim=0).contiguous()

                    ckpt_state[key] = ck

        # Try loading adjusted state (non-strict) — will ignore missing/unexpected keys
        try:
            t3.load_state_dict(ckpt_state, strict=False)
            print("✅ T3 checkpoint loaded with adjustments (strict=False).")
        except Exception as e_load:
            print(f"⚠️ First load attempt failed: {e_load}. Trying filtered-intersection load...")
            # fallback: load only intersecting keys
            filtered = {k: v for k, v in ckpt_state.items() if k in model_state}
            t3.load_state_dict(filtered, strict=False)
            print("✅ T3 checkpoint loaded using intersecting keys (strict=False).")

        t3.to(device).eval()

        # --- S3Gen ---
        s3gen = S3Gen()
        s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", weights_only=True))
        s3gen.to(device).eval()

        # --- Tokenizer (NEW) ---
        # ✅ NEW tokenizer file name (Rumble v2)
        tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

        # --- Conditionals (optional) ---
        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)


# --- Global Variables ---
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
        _ = torch.tensor([1.0]).cuda().cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA test failed: {e}")
        return False


def _test_mps_functionality() -> bool:
    if not torch.backends.mps.is_available():
        return False
    try:
        _ = torch.tensor([1.0]).to("mps").cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS test failed: {e}")
        return False


def load_model() -> bool:
    global chatterbox_model, MODEL_LOADED, model_device

    if MODEL_LOADED:
        logger.info("TTS model already loaded.")
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
                logger.info("Using CPU.")
        else:
            resolved_device_str = device_setting

        model_device = resolved_device_str

        local_model_path = Path(
            r"D:\Flie\Chatterbox-TTS-Server\Chatterbox-TTS-Server-Multilingual\v3"
        )

        print("🟢 Loading base ChatterboxTTS model (offline)...")
        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=local_model_path, device=model_device)
        MODEL_LOADED = True

        logger.info(f"Base TTS model loaded successfully on {model_device}")
        return True

    except Exception as e:
        logger.error(f"Error loading base model: {e}", exc_info=True)
        return False


def load_multilingual_model() -> bool:
    global multilingual_model, MULTILINGUAL_MODEL_LOADED, chatterbox_model, MODEL_LOADED, model_device

    try:
        if chatterbox_model is not None:
            chatterbox_model = None
            MODEL_LOADED = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        ckpt_path = Path(
            r"D:\Flie\Chatterbox-TTS-Server\Chatterbox-TTS-Server-Multilingual\v3"
        )

        logger.info("🟣 Loading multilingual patched model (offline v3)...")
        multilingual_model = PatchedChatterboxTTS.from_local(ckpt_dir=ckpt_path, device=model_device)
        chatterbox_model = multilingual_model
        MULTILINGUAL_MODEL_LOADED = True
        MODEL_LOADED = True
        logger.info("✅ Multilingual model loaded successfully.")
        return True

    except Exception as e:
        logger.error(f"Failed to load multilingual model: {e}", exc_info=True)
        return False


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
        logger.error("Model not loaded. Cannot synthesize.")
        return None, None

    try:
        if seed != 0:
            set_seed(seed)

        if isinstance(chatterbox_model, ChatterboxMultilingualTTS):
            wav = chatterbox_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                language_id=language,
            )
        else:
            wav = chatterbox_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        return wav, chatterbox_model.sr

    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        return None, None
