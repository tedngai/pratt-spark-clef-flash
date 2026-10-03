# Clef-Flash on an NVIDIA GB10 / DGX Spark (aarch64 / Blackwell, CUDA 13).
#
# The base image is pinned by digest and already ships torch 2.11.0+cu130 -- the exact
# version the model card was tested with -- plus the CUDA toolchain (nvcc), triton,
# accelerate, safetensors and fastapi. We layer an isolated venv (with
# --system-site-packages) so transformers and the fused kernels can be installed without
# disturbing the base image's vLLM environment. torch itself is inherited from the base.
#
# Build args:
#   BASE_IMAGE             pinned base (override for a different host/torch)
#   TRANSFORMERS_VERSION   default 5.10.2 (what the model card used)
#   INSTALL_KERNELS        1 to build flash-linear-attention + causal-conv1d
#   CUDA_ARCH_LIST         CUDA arch for the causal-conv1d compile (GB10 = 12.1)
#   MAX_JOBS               parallel compile jobs

ARG BASE_IMAGE=vllm/vllm-openai@sha256:9fe761ad1e8258a5e8f1c28243b1567c19b0a78e3a909c72611fdd1771f0f966
FROM ${BASE_IMAGE}

ARG TRANSFORMERS_VERSION=5.10.2
ARG INSTALL_KERNELS=1
ARG CUDA_ARCH_LIST=12.1
ARG MAX_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

RUN python3 -m venv --system-site-packages /opt/clef-venv \
 && /opt/clef-venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/clef-venv/bin/pip install --no-cache-dir \
        "transformers==${TRANSFORMERS_VERSION}" \
        "accelerate" \
        "pillow>=10" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "hf_transfer>=0.1.9"

# Fused kernels that enable transformers' Qwen3.5 fast path (is_fast_path_available):
#   * flash-linear-attention -> Triton kernels (chunk_gated_delta_rule, FusedRMSNormGated)
#   * causal-conv1d         -> compiled CUDA extension (causal_conv1d_fn/update)
# causal-conv1d has no wheels, so it is compiled from source. TORCH_CUDA_ARCH_LIST must
# match the target GPU because the build has no GPU to auto-detect (GB10 -> 12.1).
RUN if [ "${INSTALL_KERNELS}" = "1" ]; then \
        set -eux; \
        /opt/clef-venv/bin/pip install --no-cache-dir "setuptools" "wheel" "packaging" "ninja"; \
        /opt/clef-venv/bin/pip install --no-cache-dir "flash-linear-attention==0.5.2"; \
        TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
            /opt/clef-venv/bin/pip install --no-cache-dir --no-build-isolation "causal-conv1d==1.7.0"; \
    else \
        echo "INSTALL_KERNELS=0: skipping fused kernels (PyTorch fallback will be used)"; \
    fi

ENV PATH=/opt/clef-venv/bin:$PATH \
    VIRTUAL_ENV=/opt/clef-venv

WORKDIR /app
COPY server.py /app/server.py
COPY server_hardened.py /app/server_hardened.py
COPY healthcheck.py /app/healthcheck.py

EXPOSE 8001
ENTRYPOINT ["python", "/app/server.py"]
