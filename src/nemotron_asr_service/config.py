from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class AsrSettings:
    target_lang: str = "auto"
    att_context_size: list[int] | None = None
    strip_lang_tags: bool = True

    def updated(self, **kwargs: object) -> "AsrSettings":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class VadSettings:
    threshold: float = 0.6
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    sample_rate: int = 16000
    model_path: str | None = None

    def updated(self, **kwargs: object) -> "VadSettings":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class SessionSettings:
    asr: AsrSettings
    vad: VadSettings
    turn_detection: str | None = "server_vad"

    def turn_detection_payload(self) -> dict[str, object] | None:
        if self.turn_detection is None:
            return None
        return {
            "type": "server_vad",
            "threshold": self.vad.threshold,
            "prefix_padding_ms": self.vad.prefix_padding_ms,
            "silence_duration_ms": self.vad.silence_duration_ms,
        }


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "/models/nemotron-3.5-asr-streaming-0.6b.nemo"
    device: str | None = None
    sample_rate: int = 16000
    max_sessions: int = 1
    final_padding_ms: int = 200
    partial_interval_ms: int = 500
    asr: AsrSettings = AsrSettings(att_context_size=[56, 3])
    vad: VadSettings = VadSettings()
    turn_detection: str | None = "server_vad"

    @classmethod
    def from_env(cls) -> "Settings":
        sample_rate = _env_int("AUDIO_SAMPLE_RATE", 16000)
        return cls(
            host=os.environ.get("HOST", "0.0.0.0"),
            port=_env_int("PORT", 8000),
            model_path=os.environ.get("NEMO_MODEL_PATH", "/models/nemotron-3.5-asr-streaming-0.6b.nemo"),
            device=os.environ.get("NEMO_DEVICE") or None,
            sample_rate=sample_rate,
            max_sessions=_env_int("MAX_SESSIONS", 1),
            final_padding_ms=_env_int("FINAL_PADDING_MS", 200),
            partial_interval_ms=_env_int("PARTIAL_INTERVAL_MS", 500),
            asr=AsrSettings(
                target_lang=os.environ.get("TARGET_LANG", "auto"),
                att_context_size=_parse_att_context(os.environ.get("ATT_CONTEXT_SIZE", "[56,3]")),
                strip_lang_tags=_env_bool("STRIP_LANG_TAGS", True),
            ),
            vad=VadSettings(
                threshold=_env_float("VAD_THRESHOLD", 0.6),
                prefix_padding_ms=_env_int("VAD_PREFIX_PADDING_MS", 300),
                silence_duration_ms=_env_int("VAD_SILENCE_DURATION_MS", 500),
                sample_rate=sample_rate,
                model_path=os.environ.get("SILERO_VAD_MODEL_PATH") or None,
            ),
            turn_detection=_parse_turn_detection(os.environ.get("TURN_DETECTION", "server_vad")),
        )

    def session_defaults(self) -> SessionSettings:
        return SessionSettings(asr=self.asr, vad=self.vad, turn_detection=self.turn_detection)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_att_context(value: str | None) -> list[int] | None:
    if value is None or value == "" or value.lower() in {"none", "null"}:
        return None
    parsed: Any = json.loads(value)
    if (
        not isinstance(parsed, list)
        or len(parsed) != 2
        or not all(isinstance(item, int) for item in parsed)
    ):
        raise ValueError("ATT_CONTEXT_SIZE must be a JSON list of two integers, for example [56,3]")
    return parsed


def _parse_turn_detection(value: str | None) -> str | None:
    if value is None:
        return "server_vad"
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "manual", "off", "disabled"}:
        return None
    if normalized != "server_vad":
        raise ValueError("TURN_DETECTION must be server_vad or none")
    return "server_vad"
