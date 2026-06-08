# Nemotron ASR Streaming API

Thin WebSocket service for running NVIDIA Nemotron 3.5 ASR Streaming from a DGX Spark or other NVIDIA GPU node. The service exposes a small OpenAI-Realtime-like transcription API and keeps model inference NeMo-native.

## Architecture

```text
Reachy / runtime
  continuous PCM16 mono 16 kHz audio

      ws://dgx-spark:8000/v1/realtime

DGX Spark container
  FastAPI WebSocket adapter
  Silero server-side VAD
  NeMo cache-aware Nemotron ASR
```

Server-side VAD is the default, matching OpenAI Realtime's provider-owned turn detection pattern. Manual commit mode is also supported by setting turn detection to `null`.

## Run On DGX Spark

Manage the model file yourself and mount it into the container:

```bash
docker buildx build --platform linux/arm64 -t nemotron-asr-streaming-api:0.1.0 .

docker run --rm --gpus all \
  --name nemotron-asr \
  --shm-size=8g \
  -p 8000:8000 \
  -v /srv/models/nemotron:/models:ro \
  -e NEMO_MODEL_PATH=/models/nemotron-3.5-asr-streaming-0.6b.nemo \
  nemotron-asr-streaming-api:0.1.0
```

Health is ready only after model load, VAD load, and warmup:

```bash
curl http://localhost:8000/healthz
```

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `NEMO_MODEL_PATH` | `/models/nemotron-3.5-asr-streaming-0.6b.nemo` | Local mounted `.nemo` path. |
| `TARGET_LANG` | `auto` | Nemotron prompt language, for example `en-US`, `it-IT`, or `auto`. |
| `ATT_CONTEXT_SIZE` | `[56,3]` | NeMo streaming context; `[56,3]` is 320 ms. |
| `STRIP_LANG_TAGS` | `true` | Removes terminal `<lang-LOCALE>` tags from transcript text. |
| `TURN_DETECTION` | `server_vad` | Set to `none` for manual commit mode by default. |
| `VAD_THRESHOLD` | `0.6` | Silero speech threshold. |
| `VAD_PREFIX_PADDING_MS` | `300` | Audio retained before speech start. |
| `VAD_SILENCE_DURATION_MS` | `500` | Silence needed before speech stop. |
| `MAX_SESSIONS` | `1` | Concurrent WebSocket sessions. |

## WebSocket API

Endpoint: `ws://<host>:8000/v1/realtime`

Client events:

```json
{ "type": "session.update", "session": { "audio": { "input": { "turn_detection": { "type": "server_vad", "threshold": 0.6, "prefix_padding_ms": 300, "silence_duration_ms": 500 } } } } }
{ "type": "input_audio_buffer.append", "audio": "<base64 pcm16le mono 16k>" }
{ "type": "input_audio_buffer.commit" }
{ "type": "input_audio_buffer.clear" }
```

Set turn detection to `null` for manual mode:

```json
{ "type": "session.update", "session": { "audio": { "input": { "turn_detection": null } } } }
```

Server events:

```json
{ "type": "session.created", "session": { "...": "..." } }
{ "type": "input_audio_buffer.speech_started", "item_id": "item_..." }
{ "type": "conversation.item.input_audio_transcription.delta", "item_id": "item_...", "delta": "hello", "text": "hello" }
{ "type": "input_audio_buffer.speech_stopped", "item_id": "item_..." }
{ "type": "conversation.item.input_audio_transcription.completed", "item_id": "item_...", "transcript": "hello world" }
```

## Smoke Test

Install the client extra locally or in the container:

```bash
python -m pip install ".[client]"
python scripts/stream_wav.py --url ws://localhost:8000/v1/realtime --wav ./sample.wav
```

The WAV must be mono PCM16 at 16 kHz.
