#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import wave

import websockets


async def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a 16 kHz mono PCM16 WAV to the ASR WebSocket.")
    parser.add_argument("--url", default="ws://localhost:8000/v1/realtime")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--manual", action="store_true", help="Disable server VAD and commit explicitly.")
    args = parser.parse_args()

    with wave.open(args.wav, "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
            raise SystemExit("WAV must be mono PCM16 at 16 kHz")
        audio = wav.readframes(wav.getnframes())

    bytes_per_ms = 16000 * 2 // 1000
    chunk_bytes = max(bytes_per_ms * args.chunk_ms, 2)

    async with websockets.connect(args.url) as ws:
        receiver = asyncio.create_task(_print_events(ws))
        if args.manual:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"audio": {"input": {"turn_detection": None}}},
                    }
                )
            )

        for offset in range(0, len(audio), chunk_bytes):
            chunk = audio[offset : offset + chunk_bytes]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )
            await asyncio.sleep(args.chunk_ms / 1000)

        if args.manual:
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        else:
            silence = b"\x00\x00" * int(16000 * 0.8)
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(silence).decode("ascii"),
                    }
                )
            )

        await asyncio.sleep(3)
        receiver.cancel()


async def _print_events(ws) -> None:
    async for message in ws:
        print(message, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
