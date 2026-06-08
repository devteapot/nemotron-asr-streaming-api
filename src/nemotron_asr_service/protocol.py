from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any

from nemotron_asr_service.config import AsrSettings, SessionSettings, VadSettings


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def error_event(message: str, error_type: str = "invalid_request_error") -> dict[str, object]:
    return {
        "type": "error",
        "event_id": new_id("evt"),
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def session_created_event(session_id: str, settings: SessionSettings) -> dict[str, object]:
    return {
        "type": "session.created",
        "event_id": new_id("evt"),
        "session": _session_payload(session_id, settings),
    }


def _session_payload(session_id: str, settings: SessionSettings) -> dict[str, object]:
    return {
        "id": session_id,
        "object": "realtime.session",
        "modalities": ["audio", "text"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": settings.vad.sample_rate},
                "turn_detection": settings.turn_detection_payload(),
                "transcription": {
                    "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
                    "language": settings.asr.target_lang,
                },
            },
        },
        "nemotron": {
            "target_lang": settings.asr.target_lang,
            "att_context_size": settings.asr.att_context_size,
            "strip_lang_tags": settings.asr.strip_lang_tags,
        },
    }


def parse_session_update(current: SessionSettings, payload: dict[str, Any]) -> SessionSettings:
    session = payload.get("session")
    if not isinstance(session, dict):
        return current

    asr = current.asr
    vad = current.vad
    turn_detection = current.turn_detection

    nemotron = session.get("nemotron")
    if isinstance(nemotron, dict):
        asr = _parse_asr_update(asr, nemotron)

    audio = session.get("audio")
    if isinstance(audio, dict):
        audio_input = audio.get("input")
        if isinstance(audio_input, dict):
            transcription = audio_input.get("transcription")
            if isinstance(transcription, dict):
                asr = _parse_asr_update(asr, transcription)
            if "turn_detection" in audio_input:
                turn_detection, vad = _parse_turn_detection(audio_input["turn_detection"], vad)

    if "turn_detection" in session:
        turn_detection, vad = _parse_turn_detection(session["turn_detection"], vad)

    return SessionSettings(asr=asr, vad=vad, turn_detection=turn_detection)


def _parse_asr_update(current: AsrSettings, payload: dict[str, Any]) -> AsrSettings:
    kwargs: dict[str, object] = {}
    if "target_lang" in payload:
        kwargs["target_lang"] = str(payload["target_lang"])
    if "language" in payload:
        kwargs["target_lang"] = str(payload["language"])
    if "att_context_size" in payload:
        value = payload["att_context_size"]
        if value is not None and (
            not isinstance(value, list)
            or len(value) != 2
            or not all(isinstance(item, int) for item in value)
        ):
            raise ValueError("att_context_size must be null or a list of two integers")
        kwargs["att_context_size"] = value
    if "strip_lang_tags" in payload:
        kwargs["strip_lang_tags"] = bool(payload["strip_lang_tags"])
    return replace(current, **kwargs)


def _parse_turn_detection(value: Any, current_vad: VadSettings) -> tuple[str | None, VadSettings]:
    if value is None:
        return None, current_vad
    if not isinstance(value, dict):
        raise ValueError("turn_detection must be null or an object")
    detection_type = value.get("type", "server_vad")
    if detection_type != "server_vad":
        raise ValueError("only server_vad turn_detection is supported")

    kwargs: dict[str, object] = {}
    if "threshold" in value:
        threshold = float(value["threshold"])
        if threshold < 0 or threshold > 1:
            raise ValueError("turn_detection.threshold must be between 0 and 1")
        kwargs["threshold"] = threshold
    if "prefix_padding_ms" in value:
        kwargs["prefix_padding_ms"] = int(value["prefix_padding_ms"])
    if "silence_duration_ms" in value:
        kwargs["silence_duration_ms"] = int(value["silence_duration_ms"])
    return "server_vad", replace(current_vad, **kwargs)
