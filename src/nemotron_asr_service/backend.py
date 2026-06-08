from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from nemotron_asr_service.config import AsrSettings, Settings


_LANG_TAG_RE = re.compile(r"\s*<([a-z]{2}-[A-Z]{2})>\s*$")


@dataclass(frozen=True)
class TranscriptUpdate:
    text: str
    is_final: bool = False
    language: str | None = None


class StreamingAsrSession(Protocol):
    @property
    def has_audio(self) -> bool: ...

    def append_audio(self, audio: np.ndarray) -> TranscriptUpdate | None: ...

    def finalize(self) -> TranscriptUpdate: ...

    def reset(self) -> None: ...


class StreamingAsrBackend(Protocol):
    def load(self) -> None: ...

    def create_session(self, config: AsrSettings) -> StreamingAsrSession: ...


class NemoStreamingBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = None
        self.device = None
        self.compute_dtype = None
        self._torch = None
        self._buffer_cls = None

    def load(self) -> None:
        model_path = Path(self.settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"NEMO_MODEL_PATH does not exist: {model_path}")
        if model_path.suffix != ".nemo":
            raise ValueError(f"NEMO_MODEL_PATH must point to a .nemo file: {model_path}")

        import torch
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        self._torch = torch
        self._buffer_cls = CacheAwareStreamingAudioBuffer
        self.device = torch.device(self.settings.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.compute_dtype = torch.float32

        self.model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=self.device)
        self._apply_model_config(self.settings.asr)
        self.model = self.model.to(device=self.device, dtype=self.compute_dtype)
        self.model.eval()

        warmup = self.create_session(self.settings.asr)
        warmup.append_audio(np.zeros(self.settings.sample_rate, dtype=np.float32))
        warmup.finalize()

    def create_session(self, config: AsrSettings) -> StreamingAsrSession:
        if self.model is None or self._torch is None or self._buffer_cls is None:
            raise RuntimeError("ASR backend has not been loaded")
        self._apply_model_config(config)
        return NemoStreamingSession(
            model=self.model,
            torch_module=self._torch,
            buffer_cls=self._buffer_cls,
            compute_dtype=self.compute_dtype,
            config=config,
            sample_rate=self.settings.sample_rate,
            final_padding_ms=self.settings.final_padding_ms,
        )

    def _apply_model_config(self, config: AsrSettings) -> None:
        if self.model is None:
            return

        if config.att_context_size is not None:
            if not hasattr(self.model.encoder, "set_default_att_context_size"):
                raise ValueError("model does not support att_context_size overrides")
            self.model.encoder.set_default_att_context_size(att_context_size=config.att_context_size)

        if hasattr(self.model, "set_inference_prompt"):
            self.model.set_inference_prompt(config.target_lang or "auto")

        decoding = getattr(self.model, "decoding", None)
        if decoding is not None and hasattr(decoding, "set_strip_lang_tags"):
            decoding.set_strip_lang_tags(config.strip_lang_tags)


class NemoStreamingSession:
    def __init__(
        self,
        *,
        model,
        torch_module,
        buffer_cls,
        compute_dtype,
        config: AsrSettings,
        sample_rate: int,
        final_padding_ms: int,
    ) -> None:
        self.model = model
        self.torch = torch_module
        self.buffer_cls = buffer_cls
        self.compute_dtype = compute_dtype
        self.config = config
        self.sample_rate = sample_rate
        self.final_padding_ms = final_padding_ms
        self._stream_id: int | None = None
        self._last_text = ""
        self._has_audio = False
        self.reset()

    @property
    def has_audio(self) -> bool:
        return self._has_audio

    def append_audio(self, audio: np.ndarray) -> TranscriptUpdate | None:
        if audio.size == 0:
            return None

        audio = np.asarray(audio, dtype=np.float32)
        if self._stream_id is None:
            self.buffer.append_audio(audio, stream_id=-1)
            self._stream_id = 0
        else:
            self.buffer.append_audio(audio, stream_id=self._stream_id)

        self._has_audio = True
        return self._drain(final=False)

    def finalize(self) -> TranscriptUpdate:
        if self._has_audio and self.final_padding_ms > 0:
            padding_samples = int(self.sample_rate * self.final_padding_ms / 1000)
            if padding_samples > 0:
                self.append_audio(np.zeros(padding_samples, dtype=np.float32))

        update = self._drain(final=True)
        text = update.text if update is not None else self._last_text
        text, language = _extract_language(text)
        result = TranscriptUpdate(text=text, is_final=True, language=language)
        self.reset()
        return result

    def reset(self) -> None:
        self.buffer = self.buffer_cls(model=self.model, online_normalization=False, pad_and_drop_preencoded=False)
        self._stream_id = None
        self._last_text = ""
        self._has_audio = False
        self._step_num = 0
        self._pred_out_stream = None
        self._previous_hypotheses = None
        (
            self._cache_last_channel,
            self._cache_last_time,
            self._cache_last_channel_len,
        ) = self.model.encoder.get_initial_cache_state(batch_size=1)

    def _drain(self, *, final: bool) -> TranscriptUpdate | None:
        latest: TranscriptUpdate | None = None
        for chunk_audio, chunk_lengths in self.buffer:
            chunk_audio = chunk_audio.to(self.compute_dtype)
            with self.torch.inference_mode():
                (
                    self._pred_out_stream,
                    transcribed_texts,
                    self._cache_last_channel,
                    self._cache_last_time,
                    self._cache_last_channel_len,
                    self._previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=self._cache_last_channel,
                    cache_last_time=self._cache_last_time,
                    cache_last_channel_len=self._cache_last_channel_len,
                    keep_all_outputs=self.buffer.is_buffer_empty() or final,
                    previous_hypotheses=self._previous_hypotheses,
                    previous_pred_out=self._pred_out_stream,
                    drop_extra_pre_encoded=self._drop_extra_pre_encoded(),
                    return_transcription=True,
                )

            self._step_num += 1
            texts = _extract_transcriptions(transcribed_texts)
            if not texts:
                continue
            text = texts[0].strip()
            if text and text != self._last_text:
                self._last_text = text
                latest = TranscriptUpdate(text=text, is_final=False)
        return latest

    def _drop_extra_pre_encoded(self) -> int:
        if self._step_num == 0:
            return 0
        return int(getattr(self.model.encoder.streaming_cfg, "drop_extra_pre_encoded", 0))


def _extract_transcriptions(hyps) -> list[str]:
    if hyps is None:
        return []
    if not isinstance(hyps, (list, tuple)):
        hyps = [hyps]
    result: list[str] = []
    for hyp in hyps:
        result.append(str(getattr(hyp, "text", hyp) or ""))
    return result


def _extract_language(text: str) -> tuple[str, str | None]:
    match = _LANG_TAG_RE.search(text)
    if not match:
        return text, None
    return _LANG_TAG_RE.sub("", text).strip(), match.group(1)
