from __future__ import annotations

import base64
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nemotron_asr_service.app import create_app
from nemotron_asr_service.backend import NemoStreamingSession, TranscriptUpdate
from nemotron_asr_service.config import AsrSettings, Settings, VadSettings
from nemotron_asr_service.vad import VadUpdate, _load_silero_model


class FakeBackend:
    def load(self) -> None:
        self.loaded = True

    def create_session(self, config: AsrSettings):
        return FakeAsrSession()


class FakeAsrSession:
    def __init__(self) -> None:
        self.text = ""
        self._has_audio = False

    @property
    def has_audio(self) -> bool:
        return self._has_audio

    def append_audio(self, audio: np.ndarray):
        self._has_audio = True
        self.text = "hello" if not self.text else "hello world"
        return TranscriptUpdate(text=self.text)

    def finalize(self):
        text = self.text or "hello world"
        self.reset()
        return TranscriptUpdate(text=text, is_final=True, language="en-US")

    def reset(self) -> None:
        self.text = ""
        self._has_audio = False


@dataclass
class FakeVad:
    settings: VadSettings
    speaking: bool = False

    def reset(self) -> None:
        self.speaking = False

    def process(self, audio: np.ndarray):
        if np.max(np.abs(audio)) > 0 and not self.speaking:
            self.speaking = True
            return [VadUpdate(speech_started=True, speech_audio=audio, audio_start_ms=0)]
        if np.max(np.abs(audio)) > 0:
            return [VadUpdate(speech_audio=audio)]
        if self.speaking:
            self.speaking = False
            return [VadUpdate(speech_stopped=True, audio_end_ms=1000)]
        return []


def test_server_vad_flow_emits_speech_and_transcript_events():
    app = create_app(_settings(), backend=FakeBackend(), vad_factory=lambda settings: FakeVad(settings))

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json({"type": "input_audio_buffer.append", "audio": _audio(np.ones(640, dtype=np.float32) * 0.1)})

            started = ws.receive_json()
            assert started["type"] == "input_audio_buffer.speech_started"
            delta = ws.receive_json()
            assert delta["type"] == "conversation.item.input_audio_transcription.delta"
            assert delta["text"] == "hello"

            ws.send_json({"type": "input_audio_buffer.append", "audio": _audio(np.zeros(640, dtype=np.float32))})
            stopped = ws.receive_json()
            assert stopped["type"] == "input_audio_buffer.speech_stopped"
            completed = ws.receive_json()
            assert completed["type"] == "conversation.item.input_audio_transcription.completed"
            assert completed["transcript"] == "hello"


def test_manual_mode_bypasses_vad_and_commit_finalizes():
    settings = _settings(turn_detection=None)
    app = create_app(settings, backend=FakeBackend(), vad_factory=lambda settings: FakeVad(settings))

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json({"type": "input_audio_buffer.append", "audio": _audio(np.ones(640, dtype=np.float32) * 0.1)})
            delta = ws.receive_json()
            assert delta["type"] == "conversation.item.input_audio_transcription.delta"
            ws.send_json({"type": "input_audio_buffer.commit"})
            completed = ws.receive_json()
            assert completed["type"] == "conversation.item.input_audio_transcription.completed"
            assert completed["transcript"] == "hello"


def test_session_update_can_disable_turn_detection_for_manual_mode():
    app = create_app(_settings(), backend=FakeBackend(), vad_factory=lambda settings: FakeVad(settings))

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {"audio": {"input": {"turn_detection": None}}},
                }
            )
            updated = ws.receive_json()
            assert updated["type"] == "session.created"
            assert updated["session"]["audio"]["input"]["turn_detection"] is None

            ws.send_json({"type": "input_audio_buffer.append", "audio": _audio(np.ones(640, dtype=np.float32) * 0.1)})
            delta = ws.receive_json()
            assert delta["type"] == "conversation.item.input_audio_transcription.delta"
            ws.send_json({"type": "input_audio_buffer.commit"})
            assert ws.receive_json()["type"] == "conversation.item.input_audio_transcription.completed"


def test_invalid_audio_emits_error():
    app = create_app(_settings(), backend=FakeBackend(), vad_factory=lambda settings: FakeVad(settings))

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json({"type": "input_audio_buffer.append", "audio": "not base64"})
            event = ws.receive_json()
            assert event["type"] == "error"
            assert "base64" in event["error"]["message"]


def test_session_limit_rejects_second_connection():
    app = create_app(_settings(max_sessions=1), backend=FakeBackend(), vad_factory=lambda settings: FakeVad(settings))

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as first:
            assert first.receive_json()["type"] == "session.created"
            with client.websocket_connect("/v1/realtime") as second:
                event = second.receive_json()
                assert event["type"] == "error"
                assert event["error"]["type"] == "session_limit_reached"


def test_settings_reads_silero_vad_model_path(monkeypatch):
    monkeypatch.setenv("SILERO_VAD_MODEL_PATH", "/opt/nemotron-asr/silero_vad.jit")

    settings = Settings.from_env()

    assert settings.vad.model_path == "/opt/nemotron-asr/silero_vad.jit"


def test_missing_silero_vad_model_path_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="SILERO_VAD_MODEL_PATH does not exist"):
        _load_silero_model(str(tmp_path / "missing.jit"))


def test_silero_vad_loader_reads_env_model_path(monkeypatch, tmp_path):
    monkeypatch.setenv("SILERO_VAD_MODEL_PATH", str(tmp_path / "missing.jit"))

    with pytest.raises(FileNotFoundError, match="SILERO_VAD_MODEL_PATH does not exist"):
        _load_silero_model()


def test_nemo_streaming_session_reprocesses_accumulated_audio_for_partials():
    session = NemoStreamingSession(
        model=FakeNemoModel(),
        torch_module=FakeTorchModule(),
        buffer_cls=FakeNemoBuffer,
        compute_dtype="float32",
        config=AsrSettings(),
        sample_rate=16000,
        final_padding_ms=0,
        partial_interval_ms=1000,
    )

    assert session.append_audio(np.ones(8000, dtype=np.float32)) is None
    update = session.append_audio(np.ones(8000, dtype=np.float32))

    assert update is not None
    assert update.text == "16000 samples"
    assert session.finalize().text == "16000 samples"


def _settings(max_sessions: int = 1, turn_detection: str | None = "server_vad") -> Settings:
    return Settings(
        model_path="/tmp/fake.nemo",
        max_sessions=max_sessions,
        turn_detection=turn_detection,
        asr=AsrSettings(att_context_size=[56, 3]),
        vad=VadSettings(),
    )


def _audio(samples: np.ndarray) -> str:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


class FakeTorchModule:
    def inference_mode(self):
        return nullcontext()


class FakeNemoEncoder:
    streaming_cfg = type("StreamingCfg", (), {"drop_extra_pre_encoded": 1})()

    def get_initial_cache_state(self, batch_size: int):
        return None, None, None


class FakeNemoModel:
    encoder = FakeNemoEncoder()

    def conformer_stream_step(self, *, processed_signal, **_kwargs):
        return None, [f"{processed_signal.samples} samples"], None, None, None, None


class FakeNemoTensor:
    def __init__(self, samples: int) -> None:
        self.samples = samples

    def to(self, _dtype):
        return self


class FakeNemoBuffer:
    def __init__(self, **_kwargs) -> None:
        self.audio: np.ndarray | None = None
        self._empty = False

    def append_audio(self, audio: np.ndarray, stream_id: int = -1):
        self.audio = np.asarray(audio)
        return None, None, 0

    def __iter__(self):
        if self.audio is None:
            return
        self._empty = True
        yield FakeNemoTensor(self.audio.size), None

    def is_buffer_empty(self):
        return self._empty
