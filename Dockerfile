ARG BASE_IMAGE=nvcr.io/nvidia/nemo:25.11.01
FROM ${BASE_IMAGE}

ARG SILERO_VAD_VERSION=6.2.1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    NEMO_MODEL_PATH=/models/nemotron-3.5-asr-streaming-0.6b.nemo \
    SILERO_VAD_MODEL_PATH=/opt/nemotron-asr/silero_vad.jit \
    TARGET_LANG=auto \
    ATT_CONTEXT_SIZE='[56,3]' \
    STRIP_LANG_TAGS=true \
    TURN_DETECTION=server_vad \
    VAD_THRESHOLD=0.6 \
    VAD_PREFIX_PADDING_MS=300 \
    VAD_SILENCE_DURATION_MS=500 \
    MAX_SESSIONS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts

RUN python -c "import numpy, torch; import nemo.collections.asr as nemo_asr; print('base torch', torch.__version__, 'cuda', torch.version.cuda); print('base numpy', numpy.__version__); print('base nemo_asr', nemo_asr.__name__)"

RUN python -m pip install "fastapi>=0.115" "uvicorn>=0.30" "websockets>=13.0"

RUN mkdir -p "$(dirname "${SILERO_VAD_MODEL_PATH}")" \
    && python -m pip download --no-deps --only-binary=:all: --dest /tmp/silero-vad "silero-vad==${SILERO_VAD_VERSION}" \
    && python -c "import os, zipfile; from pathlib import Path; wheel = next(Path('/tmp/silero-vad').glob('silero_vad-*.whl')); source = 'silero_vad/data/silero_vad.jit'; target = Path(os.environ['SILERO_VAD_MODEL_PATH']); target.write_bytes(zipfile.ZipFile(wheel).read(source)); print('silero_model', target, target.stat().st_size)" \
    && rm -rf /tmp/silero-vad

RUN python -m pip install --no-deps ".[client]"

RUN python -c "import torch; import nemo.collections.asr as nemo_asr; from nemotron_asr_service.vad import _load_silero_model; _load_silero_model(); print('final torch', torch.__version__, 'cuda', torch.version.cuda); print('final nemo_asr', nemo_asr.__name__)"

EXPOSE 8000

CMD ["python", "-m", "nemotron_asr_service"]
