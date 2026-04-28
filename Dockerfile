# syntax=docker/dockerfile:1.6
# Reliquary Ledger validator/miner image.
#
# Multi-stage CUDA 12.8 + torch 2.7 + flash-attn 2.8.3 + reliquary-inference.
# Tracks the same toolchain that romain13190/reliquary uses on its SN81
# mainnet validator (parallel-work credit: romain13190/reliquary@6ee8592).
# Flash Attention 2 is
# required because the GRAIL sketch verifier is bit-sensitive to the
# attention kernel's reduction order.
#
# Build:
#   docker build -t ghcr.io/reliquadotai/reliquary-ledger:latest .
#
# Run miner (testnet 462):
#   docker run --rm --gpus all --env-file .env \
#     -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
#     -e RELIQUARY_INFERENCE_ROLE=miner \
#     ghcr.io/reliquadotai/reliquary-ledger:latest
#
# Run validator (testnet 462):
#   docker run --rm --gpus all --env-file .env \
#     -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
#     -e RELIQUARY_INFERENCE_ROLE=validator \
#     -p 9108:9108 -p 9180:9180 \
#     ghcr.io/reliquadotai/reliquary-ledger:latest

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update -qq && apt-get install -y -qq \
        python3.12 python3.12-venv python3-pip \
        git build-essential wget curl ca-certificates jq \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv so system pip stays clean.
RUN python3.12 -m venv /opt/reliquary-venv
ENV PATH="/opt/reliquary-venv/bin:${PATH}"

# torch 2.7.0 + CUDA 12.8 (matches our cross-GPU audit fleet).
RUN pip install --upgrade pip wheel setuptools \
 && pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# flash-attn prebuilt wheel for torch 2.7 / cu12 / cp312 / cxx11abi=TRUE.
# Do NOT rename the wheel on download: pip parses the version + platform
# tags from the filename and fails with "Invalid wheel filename" otherwise.
ARG FA_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
RUN wget -q "${FA_URL}" -P /tmp/ \
 && pip install /tmp/flash_attn-*.whl \
 && rm /tmp/flash_attn-*.whl

# Source + install. We install the package in editable mode so operators
# can volume-mount their own checkout if they need a one-off patch.
WORKDIR /opt/reliquary-ledger
COPY . /opt/reliquary-ledger
RUN pip install -e ".[dev]"

# bittensor 10.x ships with async-substrate-interface 2.0 which conflicts
# with its own scalecodec import path — roll back to the 1.x line that
# matches our CI baseline.
RUN pip uninstall -y cyscale 2>/dev/null || true \
 && pip install 'async-substrate-interface<2.0.0' \
 && pip install --force-reinstall --no-deps scalecodec==1.2.12

# boto3 for R2 (validator state mirror, weight-only mode).
RUN pip install boto3

# Runtime defaults — ATTN_IMPL is bit-sensitive; do not change.
ENV GRAIL_ATTN_IMPL=flash_attention_2
ENV RELIQUARY_INFERENCE_REQUIRE_FLASH_ATTENTION=1

COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

# Health on 9180, Prometheus on 9108. Bind to 0.0.0.0 inside the container;
# the operator decides via -p whether to expose them on the host.
EXPOSE 9108 9180

HEALTHCHECK --interval=30s --timeout=5s --retries=4 --start-period=120s \
  CMD curl -fsS http://127.0.0.1:9180/healthz || exit 1

ENTRYPOINT ["/opt/entrypoint.sh"]
