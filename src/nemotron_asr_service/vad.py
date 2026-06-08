from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from nemotron_asr_service.config import VadSettings


@dataclass(frozen=True)
class VadUpdate:
    speech_started: bool = False
    speech_stopped: bool = False
    speech_audio: np.ndarray | None = None
    audio_start_ms: int | None = None
    audio_end_ms: int | None = None


class ServerVadEngine(Protocol):
    def process(self, audio: np.ndarray) -> list[VadUpdate]: ...

    def reset(self) -> None: ...


class SileroVadEngine:
    def __init__(self, settings: VadSettings) -> None:
        self.settings = settings
        self.sample_rate = settings.sample_rate
        if self.sample_rate not in {8000, 16000}:
            raise ValueError("Silero VAD supports only 8000 or 16000 Hz audio")
        self.window_size = 512 if self.sample_rate == 16000 else 256
        self.model = _load_silero_model(settings.model_path)
        self.reset()

    def reset(self) -> None:
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
        self.triggered = False
        self.temp_end_sample = 0
        self.current_sample = 0
        self.tail = np.zeros(0, dtype=np.float32)
        self.pre_speech: deque[np.ndarray] = deque()
        self.pre_speech_samples = 0

    def process(self, audio: np.ndarray) -> list[VadUpdate]:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return []

        combined = np.concatenate([self.tail, audio]) if self.tail.size else audio
        full_len = (combined.size // self.window_size) * self.window_size
        self.tail = combined[full_len:].copy()
        updates: list[VadUpdate] = []

        for offset in range(0, full_len, self.window_size):
            window = combined[offset : offset + self.window_size].astype(np.float32, copy=False)
            updates.extend(self._process_window(window))
        return updates

    def _process_window(self, window: np.ndarray) -> list[VadUpdate]:
        import torch

        self.current_sample += window.size
        speech_prob = float(self.model(torch.from_numpy(window), self.sample_rate).item())
        updates: list[VadUpdate] = []

        if speech_prob >= self.settings.threshold and not self.triggered:
            self.triggered = True
            prefix = list(self.pre_speech)
            self.pre_speech.clear()
            self.pre_speech_samples = 0
            speech_audio = np.concatenate([*prefix, window]) if prefix else window.copy()
            start_sample = max(0, self.current_sample - window.size - speech_audio.size + window.size)
            updates.append(
                VadUpdate(
                    speech_started=True,
                    speech_audio=speech_audio,
                    audio_start_ms=_samples_to_ms(start_sample, self.sample_rate),
                )
            )
            return updates

        if not self.triggered:
            self._remember_pre_speech(window)
            return updates

        speech_stopped = False
        if speech_prob >= self.settings.threshold - 0.15:
            self.temp_end_sample = 0
        else:
            if self.temp_end_sample == 0:
                self.temp_end_sample = self.current_sample
            silence_samples = int(self.sample_rate * self.settings.silence_duration_ms / 1000)
            if self.current_sample - self.temp_end_sample >= silence_samples:
                speech_stopped = True
                self.triggered = False
                self.temp_end_sample = 0

        updates.append(
            VadUpdate(
                speech_stopped=speech_stopped,
                speech_audio=window.copy(),
                audio_end_ms=_samples_to_ms(self.current_sample, self.sample_rate) if speech_stopped else None,
            )
        )
        return updates

    def _remember_pre_speech(self, window: np.ndarray) -> None:
        max_samples = int(self.sample_rate * self.settings.prefix_padding_ms / 1000)
        if max_samples <= 0:
            self.pre_speech.clear()
            self.pre_speech_samples = 0
            return
        self.pre_speech.append(window.copy())
        self.pre_speech_samples += window.size
        while self.pre_speech and self.pre_speech_samples > max_samples:
            first = self.pre_speech[0]
            excess = self.pre_speech_samples - max_samples
            if excess >= first.size:
                self.pre_speech.popleft()
                self.pre_speech_samples -= first.size
            else:
                self.pre_speech[0] = first[excess:].copy()
                self.pre_speech_samples -= excess


def _load_silero_model(model_path: str | None = None):
    import importlib.util

    model_path = model_path or os.environ.get("SILERO_VAD_MODEL_PATH")
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"SILERO_VAD_MODEL_PATH does not exist: {path}")

        import torch

        torch.set_num_threads(1)
        return torch.jit.load(str(path), map_location="cpu").eval()

    import torch

    spec = importlib.util.find_spec("silero_vad")
    if spec and spec.submodule_search_locations:
        package_dir = Path(next(iter(spec.submodule_search_locations)))
        model_path = package_dir / "data" / "silero_vad.jit"
        if model_path.exists():
            torch.set_num_threads(1)
            return torch.jit.load(str(model_path), map_location="cpu").eval()

    raise RuntimeError("Silero VAD model not found. Set SILERO_VAD_MODEL_PATH or install the silero-vad package.")


def _samples_to_ms(samples: int, sample_rate: int) -> int:
    return int(samples / sample_rate * 1000)
