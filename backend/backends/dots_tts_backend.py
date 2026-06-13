"""
dots.tts backend implementation.

Wraps DotsTtsRuntime from the dots.tts package for zero-shot voice cloning.
2B-parameter fully continuous autoregressive TTS with 48 kHz output.
Supports 24 languages. Three checkpoints: base, soar (best cloning), mf (fastest).

No MPS support — dots.tts only supports CUDA or CPU.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import TTSBackend
from .base import (
    is_model_cached,
    get_torch_device,
    empty_device_cache,
    manual_seed,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

# Three checkpoints — choose by quality / speed tradeoff
DOTS_TTS_HF_REPOS = {
    "base": "rednote-hilab/dots.tts-base",
    "soar": "rednote-hilab/dots.tts-soar",
    "mf": "rednote-hilab/dots.tts-mf",
}

# Recommended num_steps per variant
DOTS_TTS_NUM_STEPS = {
    "base": 10,
    "soar": 10,
    "mf": 4,
}

# Required files for cache check
_REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "vocoder.safetensors",
    "speaker_encoder.safetensors",
]


class DotsTTSBackend:
    """dots.tts backend — 2B continuous AR TTS, 48 kHz output."""

    def __init__(self):
        self.model = None  # DotsTtsRuntime instance
        self.model_size = "soar"  # default variant
        self._device = None
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        """Return CUDA or CPU. dots.tts does not support MPS."""
        return get_torch_device(allow_xpu=True, allow_mps=False)

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "soar") -> str:
        return DOTS_TTS_HF_REPOS.get(model_size, DOTS_TTS_HF_REPOS["soar"])

    def _is_model_cached(self, model_size: str = "soar") -> bool:
        repo = self._get_model_path(model_size)
        return is_model_cached(repo, required_files=_REQUIRED_FILES)

    async def load_model(self, model_size: str = "soar") -> None:
        """Load the dots.tts model."""
        if self.model is not None:
            return
        async with self._model_load_lock:
            if self.model is not None:
                return
            self.model_size = model_size
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self):
        """Synchronous model loading."""
        model_name = f"dots-tts-{self.model_size}"
        is_cached = self._is_model_cached(self.model_size)

        with model_load_progress(model_name, is_cached):
            device = self._get_device()
            self._device = device
            logger.info(f"Loading dots.tts ({self.model_size}) on {device}...")

            from dots_tts.runtime import DotsTtsRuntime

            repo = self._get_model_path(self.model_size)

            runtime = DotsTtsRuntime.from_pretrained(
                repo,
                precision="bfloat16",
                optimize=True,  # torch.compile acceleration
            )

            self.model = runtime

        logger.info(f"dots.tts ({self.model_size}) loaded successfully")

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            device = self._device
            del self.model
            self.model = None
            self._device = None
            empty_device_cache(device)
            logger.info("dots.tts unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        dots.tts processes reference audio at generation time, so the
        prompt just stores the file path and transcript. The actual audio
        is loaded by runtime.generate() via prompt_audio_path.
        """
        voice_prompt = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }
        return voice_prompt, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio using dots.tts.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with ref_audio and ref_text
            language: BCP-47 language code (uppercased for dots.tts)
            seed: Random seed for reproducibility
            instruct: Unused (protocol compatibility)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text")

        if ref_audio and not Path(ref_audio).exists():
            logger.warning(f"Reference audio not found: {ref_audio}")
            ref_audio = None

        # Voicebox API uses lowercase BCP-47 codes (e.g. "en", "zh"),
        # but dots.tts runtime expects uppercase (e.g. "EN", "ZH").
        dots_language = language.upper() if language else None

        # Get recommended num_steps for this variant
        num_steps = DOTS_TTS_NUM_STEPS.get(self.model_size, 10)

        def _generate_sync():
            import torch

            if seed is not None:
                manual_seed(seed, self._device)

            logger.info(
                f"[dots.tts] Generating: size={self.model_size} lang={dots_language} "
                f"num_steps={num_steps} has_ref={ref_audio is not None}"
            )

            result = self.model.generate(
                text=text,
                prompt_audio_path=ref_audio,
                prompt_text=ref_text,
                language=dots_language,
                num_steps=num_steps,
                # Guidance scale 1.2 is the default recommended by dots.tts authors.
                # Higher values increase fidelity but may reduce naturalness.
                guidance_scale=1.2,
            )

            # Convert tensor -> numpy
            audio_tensor = result["audio"]
            if isinstance(audio_tensor, torch.Tensor):
                audio = audio_tensor.float().cpu().squeeze().numpy().astype(np.float32)
            else:
                audio = np.asarray(audio_tensor, dtype=np.float32)

            sample_rate = result.get("sample_rate", 48000)

            return audio, sample_rate

        return await asyncio.to_thread(_generate_sync)
