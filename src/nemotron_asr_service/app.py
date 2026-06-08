from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, WebSocket

from nemotron_asr_service.backend import NemoStreamingBackend, StreamingAsrBackend
from nemotron_asr_service.config import Settings, VadSettings
from nemotron_asr_service.service import RealtimeAsrServer
from nemotron_asr_service.vad import ServerVadEngine, SileroVadEngine


VadFactory = Callable[[VadSettings], ServerVadEngine]


def create_app(
    settings: Settings | None = None,
    *,
    backend: StreamingAsrBackend | None = None,
    vad_factory: VadFactory | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    backend = backend or NemoStreamingBackend(settings)
    vad_factory = vad_factory or (lambda vad_settings: SileroVadEngine(vad_settings))
    server = RealtimeAsrServer(settings=settings, backend=backend, vad_factory=vad_factory)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await server.startup()
        yield
        await server.shutdown()

    app = FastAPI(title="Nemotron ASR Streaming API", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return server.health()

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await server.handle_websocket(websocket)

    app.state.asr_server = server
    return app
