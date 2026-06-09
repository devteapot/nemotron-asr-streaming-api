from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Callable

from fastapi import WebSocket, WebSocketDisconnect

from nemotron_asr_service.audio import decode_pcm16_base64
from nemotron_asr_service.backend import StreamingAsrBackend, StreamingAsrSession, TranscriptUpdate
from nemotron_asr_service.config import SessionSettings, Settings, VadSettings
from nemotron_asr_service.protocol import error_event, new_id, parse_session_update, session_created_event
from nemotron_asr_service.vad import ServerVadEngine, VadUpdate


VadFactory = Callable[[VadSettings], ServerVadEngine]


@dataclass
class ConnectionState:
    session_id: str
    settings: SessionSettings
    asr: StreamingAsrSession
    vad: ServerVadEngine | None
    item_id: str | None = None
    last_text: str = ""


class RealtimeAsrServer:
    def __init__(self, *, settings: Settings, backend: StreamingAsrBackend, vad_factory: VadFactory) -> None:
        self.settings = settings
        self.backend = backend
        self.vad_factory = vad_factory
        self._active_sessions = 0
        self._active_lock = asyncio.Lock()
        self._ready = False
        self._startup_error: str | None = None

    async def startup(self) -> None:
        try:
            await asyncio.to_thread(self.backend.load)
            if self.settings.turn_detection == "server_vad":
                await asyncio.to_thread(lambda: self.vad_factory(self.settings.vad).reset())
            self._ready = True
        except Exception as exc:
            self._startup_error = str(exc)
            raise

    async def shutdown(self) -> None:
        self._ready = False

    def health(self) -> dict[str, object]:
        return {
            "ready": self._ready,
            "active_sessions": self._active_sessions,
            "max_sessions": self.settings.max_sessions,
            "error": self._startup_error,
        }

    async def handle_websocket(self, websocket: WebSocket) -> None:
        await websocket.accept()
        if not await self._try_acquire_session():
            await self._send_json(websocket, error_event("session limit reached", "session_limit_reached"))
            await websocket.close(code=1008)
            return

        try:
            state = self._create_connection_state()
            if not await self._send_json(websocket, session_created_event(state.session_id, state.settings)):
                return
            await self._receive_loop(websocket, state)
        finally:
            await self._release_session()

    async def _receive_loop(self, websocket: WebSocket, state: ConnectionState) -> None:
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                return

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                if not await self._send_json(websocket, error_event("message must be valid JSON")):
                    return
                continue

            if not isinstance(message, dict):
                if not await self._send_json(websocket, error_event("message must be a JSON object")):
                    return
                continue

            try:
                await self._handle_message(websocket, state, message)
            except WebSocketDisconnect:
                return
            except RuntimeError as exc:
                if _is_closed_websocket_error(exc):
                    return
                if not await self._send_json(websocket, error_event(str(exc))):
                    return
            except Exception as exc:
                if not await self._send_json(websocket, error_event(str(exc))):
                    return

    async def _handle_message(self, websocket: WebSocket, state: ConnectionState, message: dict) -> None:
        event_type = message.get("type")
        if event_type == "session.update":
            await self._handle_session_update(websocket, state, message)
        elif event_type == "input_audio_buffer.append":
            await self._handle_audio_append(websocket, state, message)
        elif event_type == "input_audio_buffer.commit":
            await self._finalize_turn(websocket, state, forced=True)
        elif event_type == "input_audio_buffer.clear":
            self._reset_turn(state)
        else:
            await websocket.send_json(error_event(f"unsupported event type: {event_type}", "unknown_event"))

    async def _handle_session_update(self, websocket: WebSocket, state: ConnectionState, message: dict) -> None:
        if state.item_id is not None or state.asr.has_audio:
            await websocket.send_json(error_event("session.update is not allowed during an active turn"))
            return

        new_settings = parse_session_update(state.settings, message)
        state.settings = new_settings
        state.asr = self.backend.create_session(new_settings.asr)
        state.vad = self.vad_factory(new_settings.vad) if new_settings.turn_detection == "server_vad" else None
        await websocket.send_json(session_created_event(state.session_id, state.settings))

    async def _handle_audio_append(self, websocket: WebSocket, state: ConnectionState, message: dict) -> None:
        audio_payload = message.get("audio")
        if not isinstance(audio_payload, str):
            await websocket.send_json(error_event("input_audio_buffer.append requires an audio string"))
            return

        audio = decode_pcm16_base64(audio_payload)
        if state.vad is None:
            self._ensure_turn(state)
            update = await asyncio.to_thread(state.asr.append_audio, audio)
            await self._emit_partial_if_changed(websocket, state, update)
            return

        vad_updates = await asyncio.to_thread(state.vad.process, audio)
        for vad_update in vad_updates:
            await self._handle_vad_update(websocket, state, vad_update)

    async def _handle_vad_update(self, websocket: WebSocket, state: ConnectionState, update: VadUpdate) -> None:
        if update.speech_started:
            self._reset_asr_turn(state)
            self._ensure_turn(state)
            await websocket.send_json(
                {
                    "type": "input_audio_buffer.speech_started",
                    "event_id": new_id("evt"),
                    "item_id": state.item_id,
                    "audio_start_ms": update.audio_start_ms,
                }
            )

        if update.speech_audio is not None and update.speech_audio.size > 0:
            self._ensure_turn(state)
            asr_update = await asyncio.to_thread(state.asr.append_audio, update.speech_audio)
            await self._emit_partial_if_changed(websocket, state, asr_update)

        if update.speech_stopped:
            if state.item_id is not None:
                await websocket.send_json(
                    {
                        "type": "input_audio_buffer.speech_stopped",
                        "event_id": new_id("evt"),
                        "item_id": state.item_id,
                        "audio_end_ms": update.audio_end_ms,
                    }
                )
            await self._finalize_turn(websocket, state, forced=False)

    async def _emit_partial_if_changed(
        self,
        websocket: WebSocket,
        state: ConnectionState,
        update: TranscriptUpdate | None,
    ) -> None:
        if update is None or update.is_final or not update.text:
            return

        text = update.text
        if text == state.last_text:
            return
        delta = text[len(state.last_text) :] if text.startswith(state.last_text) else text
        state.last_text = text
        if not delta:
            return

        await websocket.send_json(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "event_id": new_id("evt"),
                "item_id": state.item_id,
                "content_index": 0,
                "delta": delta,
                "text": text,
            }
        )

    async def _finalize_turn(self, websocket: WebSocket, state: ConnectionState, *, forced: bool) -> None:
        if state.item_id is None and not state.asr.has_audio:
            if forced:
                await websocket.send_json(error_event("no active audio buffer to commit"))
            return

        self._ensure_turn(state)
        final = await asyncio.to_thread(state.asr.finalize)
        transcript = final.text or state.last_text
        await websocket.send_json(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "event_id": new_id("evt"),
                "item_id": state.item_id,
                "content_index": 0,
                "transcript": transcript,
                "language": final.language,
            }
        )
        state.item_id = None
        state.last_text = ""

    def _create_connection_state(self) -> ConnectionState:
        session_settings = self.settings.session_defaults()
        return ConnectionState(
            session_id=new_id("sess"),
            settings=session_settings,
            asr=self.backend.create_session(session_settings.asr),
            vad=self.vad_factory(session_settings.vad) if session_settings.turn_detection == "server_vad" else None,
        )

    def _ensure_turn(self, state: ConnectionState) -> None:
        if state.item_id is None:
            state.item_id = new_id("item")
            state.last_text = ""

    def _reset_turn(self, state: ConnectionState) -> None:
        self._reset_asr_turn(state)
        if state.vad is not None:
            state.vad.reset()

    def _reset_asr_turn(self, state: ConnectionState) -> None:
        state.asr.reset()
        state.item_id = None
        state.last_text = ""

    async def _try_acquire_session(self) -> bool:
        async with self._active_lock:
            if self._active_sessions >= self.settings.max_sessions:
                return False
            self._active_sessions += 1
            return True

    async def _release_session(self) -> None:
        async with self._active_lock:
            self._active_sessions = max(0, self._active_sessions - 1)

    async def _send_json(self, websocket: WebSocket, event: dict[str, object]) -> bool:
        try:
            await websocket.send_json(event)
            return True
        except WebSocketDisconnect:
            return False
        except RuntimeError as exc:
            if _is_closed_websocket_error(exc):
                return False
            raise


def _is_closed_websocket_error(exc: RuntimeError) -> bool:
    return 'Cannot call "send" once a close message has been sent' in str(exc)
