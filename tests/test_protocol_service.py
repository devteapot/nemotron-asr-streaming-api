from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np
from fastapi.testclient import TestClient

from nemotron_asr_service.app import create_app
from nemotron_asr_service.backend import TranscriptUpdate
from nemotron_asr_service.config import AsrSettings, Settings, VadSettings
from nemotron_asr_service.vad import VadUpdate


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
