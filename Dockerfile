FROM nvcr.io/nvidia/pytorch:26.05-py3

ARG NEMO_COMMIT=160a7428769067f24ae45e04030ce738d0407727

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

RUN python -m pip install --upgrade pip "setuptools<82" wheel \
    && python -m pip install Cython packaging \
    && python -m pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@${NEMO_COMMIT}" \
    && python -m pip uninstall -y torchaudio \
    && python -m pip install ".[client]" \
    && python -c "import torch; import nemo.collections.asr as nemo_asr; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('nemo_asr', nemo_asr.__name__)"

EXPOSE 8000

CMD ["python", "-m", "nemotron_asr_service"]
