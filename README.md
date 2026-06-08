# Nemotron ASR Streaming API

Thin WebSocket service for running NVIDIA Nemotron 3.5 ASR Streaming from a DGX Spark or other NVIDIA GPU node. The service exposes a small OpenAI-Realtime-like transcription API and keeps model inference NeMo-native.

The Docker image is based on `nvcr.io/nvidia/nemo:25.11.01`, the prebuilt NeMo
container currently documented by the NeMo Toolkit package. The model card lists
this model's runtime engine as NeMo 26.06 and recommends NeMo from GitHub
`main`, while the available container does not include the prompt-conditioned
RNN-T module required by the checkpoint. The image therefore keeps CUDA/PyTorch
from the base container, replaces `/opt/NeMo` with a pinned NeMo source ref that
contains `EncDecRNNTBPEModelWithPrompt`, and installs that source with
`--no-deps` so Torch, TorchAudio, and TorchVision are not replaced.

## Architecture

```text
Reachy / runtime
  continuous PCM16 mono 16 kHz audio

      ws://dgx-spark:8000/v1/realtime

DGX Spark container
  FastAPI WebSocket adapter
  Silero server-side VAD loaded from a bundled TorchScript model file
  NeMo cache-aware Nemotron ASR over accumulated turn audio
```

Server-side VAD is the default, matching OpenAI Realtime's provider-owned turn detection pattern. Manual commit mode is also supported by setting turn detection to `null`.

For transcript quality, incoming PCM chunks are accumulated per turn and
reprocessed through NeMo's cache-aware buffer at a throttled interval. NeMo's
buffer performs feature extraction when audio is appended, so preprocessing tiny
independent WebSocket chunks produces poor recognition. `PARTIAL_INTERVAL_MS`
controls how often partial transcripts are recomputed during an active turn.

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

The default build arguments are:

```text
BASE_IMAGE=nvcr.io/nvidia/nemo:25.11.01
NEMO_SOURCE_REF=d947ef7be814e3034dfed298f0c1c3c2137bced5
SILERO_VAD_VERSION=6.2.1
```

Override `NEMO_SOURCE_REF` only when moving to a newer NeMo source revision or
to a published NeMo container that already includes the model class.

`NEMO_MODEL_PATH` is evaluated inside the container, not on the host. If the
host file is:

```text
/srv/models/nvidia/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo
```

then either mount the host model tree at the same path:

```bash
docker run --rm --gpus all \
  --name nemotron-asr \
  --shm-size=8g \
  -p 8000:8000 \
  -v /srv/models:/srv/models:ro \
  -e NEMO_MODEL_PATH=/srv/models/nvidia/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
  nemotron-asr-streaming-api:0.1.0
```

or mount just the model directory to `/models`:

```bash
docker run --rm --gpus all \
  --name nemotron-asr \
  --shm-size=8g \
  -p 8000:8000 \
  -v /srv/models/nvidia/nemotron-3.5-asr-streaming-0.6b:/models:ro \
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
| `SILERO_VAD_MODEL_PATH` | `/opt/nemotron-asr/silero_vad.jit` | Server VAD TorchScript model path. The Docker image bundles this file. |
| `TARGET_LANG` | `auto` | Nemotron prompt language, for example `en-US`, `it-IT`, or `auto`. |
| `ATT_CONTEXT_SIZE` | `[56,3]` | NeMo streaming context; `[56,3]` is 320 ms. |
| `STRIP_LANG_TAGS` | `true` | Removes terminal `<lang-LOCALE>` tags from transcript text. |
| `TURN_DETECTION` | `server_vad` | Set to `none` for manual commit mode by default. |
| `VAD_THRESHOLD` | `0.6` | Silero speech threshold. |
| `VAD_PREFIX_PADDING_MS` | `300` | Audio retained before speech start. |
| `VAD_SILENCE_DURATION_MS` | `500` | Silence needed before speech stop. |
| `PARTIAL_INTERVAL_MS` | `500` | Minimum new audio before recomputing a partial transcript from accumulated turn audio. |
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

For local non-container server runs with `TURN_DETECTION=server_vad`, either set
`SILERO_VAD_MODEL_PATH` to a local `silero_vad.jit` file or install the optional
VAD package data with:

```bash
python -m pip install ".[vad]"
```
