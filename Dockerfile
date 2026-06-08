ARG BASE_IMAGE=nvcr.io/nvidia/nemo:26.04
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    NEMO_MODEL_PATH=/models/nemotron-3.5-asr-streaming-0.6b.nemo \
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

RUN python -c "import torch; import nemo.collections.asr as nemo_asr; print('base torch', torch.__version__, 'cuda', torch.version.cuda); print('base nemo_asr', nemo_asr.__name__)"

RUN python -m pip install "fastapi>=0.115" "uvicorn[standard]>=0.30" "websockets>=13.0"

RUN python -m pip install --no-deps "silero-vad>=5.1" \
    && python -c "import importlib.util; from pathlib import Path; spec = importlib.util.find_spec('silero_vad'); assert spec and spec.submodule_search_locations; model = Path(next(iter(spec.submodule_search_locations))) / 'data' / 'silero_vad.jit'; assert model.exists(), model; print('silero_model', model)"

RUN python -m pip install --no-deps ".[client]"

RUN python -c "import torch; import nemo.collections.asr as nemo_asr; from nemotron_asr_service.vad import _load_silero_model; _load_silero_model(); print('final torch', torch.__version__, 'cuda', torch.version.cuda); print('final nemo_asr', nemo_asr.__name__)"

EXPOSE 8000

CMD ["python", "-m", "nemotron_asr_service"]
