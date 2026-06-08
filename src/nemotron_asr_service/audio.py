from __future__ import annotations

import base64
import binascii

import numpy as np


def decode_pcm16_base64(payload: str) -> np.ndarray:
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("audio must be valid base64") from exc

    if len(raw) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(raw) % 2 != 0:
        raise ValueError("pcm16 audio byte length must be even")

    pcm = np.frombuffer(raw, dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def encode_pcm16_base64(audio: np.ndarray) -> str:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")
