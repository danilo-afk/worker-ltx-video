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
# ==========================================

CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE=ltx-video

WORKDIR /comfyui

# Create necessary directories
RUN mkdir -p models/checkpoints models/vae models/unet models/clip \
    models/text_encoders models/diffusion_models models/loras \
    models/LLM/gemma-3-12b-it-qat-q4_0-unquantized

# ============ LTX-Video Models ============
# Checkpoint principal (FP8 - ~19GB, requer menos VRAM)
RUN if [ "$MODEL_TYPE" = "ltx-video" ]; then \
      echo "Downloading LTX-2 checkpoint (FP8)..." && \
      wget --progress=dot:giga \
        -O models/checkpoints/ltx-2-19b-dev-fp8.safetensors \
        https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors; \
    fi

# Upscalers (spatial + temporal)
RUN if [ "$MODEL_TYPE" = "ltx-video" ]; then \
      echo "Downloading spatial upscaler..." && \
      wget --progress=dot:giga \
        -O models/checkpoints/ltx-2-spatial-upscaler-x2-1.0.safetensors \
        https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors && \
      echo "Downloading temporal upscaler..." && \
      wget --progress=dot:giga \
        -O models/checkpoints/ltx-2-temporal-upscaler-x2-1.0.safetensors \
        https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-temporal-upscaler-x2-1.0.safetensors; \
    fi

# Distilled LoRA (para two-stage pipeline)
RUN if [ "$MODEL_TYPE" = "ltx-video" ]; then \
      echo "Downloading distilled LoRA..." && \
      wget --progress=dot:giga \
        -O models/loras/ltx-2-19b-distilled-lora-384.safetensors \
        https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors; \
    fi

# Text Encoder: Gemma 3 12B (QAT Q4_0 unquantized) - 5 shards, ~24.4GB total
# Requer aceitar licença do Google no HuggingFace + token
RUN if [ "$MODEL_TYPE" = "ltx-video" ]; then \
      echo "Downloading Gemma 3 12B text encoder configs..." && \
      cd models/LLM/gemma-3-12b-it-qat-q4_0-unquantized && \
      wget --progress=dot:giga --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main/config.json && \
      wget --progress=dot:giga --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main/tokenizer.json && \
      wget --progress=dot:giga --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main/tokenizer.model && \
      wget --progress=dot:giga --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main/tokenizer_config.json && \
      wget --progress=dot:giga --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main/special_tokens_map.json && \
      wget --progress=dot:giga --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main/generation_config.json && \
      wget --progress=dot:giga --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main/model.safetensors.index.json && \
      echo "Downloading Gemma 3 model shards (5 of 5)..." && \
      wget --progress=dot:giga --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main/model-00001-of-00005.safetensors && \
      wget --progress=dot:giga --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main/model-00002-of-00005.safetensors && \
      wget --progress=dot:giga --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main/model-00003-of-00005.safetensors && \
      wget --progress=dot:giga --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main/model-00004-of-00005.safetensors && \
      wget --progress=dot:giga --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main/model-00005-of-00005.safetensors && \
      echo "Gemma 3 text encoder downloaded successfully!"; \
    fi
# ==========================================

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2
COPY --from=downloader /comfyui/models /comfyui/models
