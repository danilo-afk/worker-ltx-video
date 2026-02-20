# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    build-essential \
    g++ \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Upgrade PyTorch if needed
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

WORKDIR /

# Install Python runtime dependencies
RUN uv pip install runpod requests websocket-client

# Add application code and scripts
ADD src/start.sh src/network_volume.py handler.py test_input.json ./
RUN chmod +x /start.sh

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

ENV PIP_NO_INPUT=1

# Helper script to switch Manager network mode
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# ============ ComfyUI-LTXVideo ============
# Custom node para geração de vídeo com LTX-2
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    cd ComfyUI-LTXVideo && \
    uv pip install --no-cache-dir -r requirements.txt

# VideoHelperSuite (VHS) - output de vídeo (VHS_VideoCombine)
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    uv pip install --no-cache-dir -r requirements.txt

# RES4LYF - sampler avançado res_2s (usado no workflow oficial LTX-2)
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/ClownsharkBatwing/RES4LYF.git && \
    cd RES4LYF && \
    uv pip install --no-cache-dir -r requirements.txt

# ComfyUI-Impact-Pack - ImpactExecutionOrderController
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    cd ComfyUI-Impact-Pack && \
    python install.py
# ==========================================

CMD ["/start.sh"]
