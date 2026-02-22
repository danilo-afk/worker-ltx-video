#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode
comfy-manager-set-mode offline || echo "worker-ltx-video - Could not set ComfyUI-Manager network_mode" >&2

# ============ Auto-download modelos para Network Volume ============
VOLUME="/runpod-volume"
if [ -d "$VOLUME" ]; then
  echo "worker-ltx-video: Network volume detectado em $VOLUME"

  # Checkpoint padrão para produção: distilled FP8 (mais estável para I2V no endpoint atual).
  CKPT_NAME="${LTX_CKPT_NAME:-ltx-2-19b-distilled-fp8.safetensors}"
  case "$CKPT_NAME" in
    ltx-2-19b-dev-fp8.safetensors)
      CKPT_URL="https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors"
      ;;
    ltx-2-19b-distilled-fp8.safetensors)
      CKPT_URL="https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-fp8.safetensors"
      ;;
    *)
      echo "worker-ltx-video: LTX_CKPT_NAME inválido: $CKPT_NAME" >&2
      exit 1
      ;;
  esac
  CKPT="$VOLUME/models/checkpoints/$CKPT_NAME"
  GEMMA_DIR="$VOLUME/models/text_encoders/gemma-3-fp8"
  GEMMA_MODEL="$GEMMA_DIR/model.safetensors"
  TOKENIZER="$GEMMA_DIR/tokenizer.model"

  # Tamanhos mínimos esperados (bytes) - detecta downloads incompletos
  CKPT_MIN=27000000000    # checkpoint real: 27GB
  GEMMA_MIN=13000000000   # gemma ~13GB

  check_size() {
    local file="$1" min="$2"
    if [ -f "$file" ]; then
      local size
      size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
      if [ "$size" -lt "$min" ]; then
        echo "worker-ltx-video: $file corrompido (${size} bytes < ${min}), re-baixando..."
        rm -f "$file"
        return 1
      fi
    else
      return 1
    fi
    return 0
  }

  if ! check_size "$CKPT" "$CKPT_MIN"; then
    echo "worker-ltx-video: Baixando checkpoint LTX-2 ($CKPT_NAME)..."
    mkdir -p "$VOLUME/models/checkpoints"
    wget --progress=dot:giga \
      -O "$CKPT" \
      "$CKPT_URL"
  fi

  # Compatibilidade com workflows antigos/oficiais que usam outros nomes de arquivo.
  ln -sf "$CKPT_NAME" "$VOLUME/models/checkpoints/ltx-2-19b-dev-fp8.safetensors"
  ln -sf "$CKPT_NAME" "$VOLUME/models/checkpoints/ltx-2-19b-distilled.safetensors"

  # Marker para forçar re-download quando fonte muda
  GEMMA_MARKER="$GEMMA_DIR/.source-comfy-org-fp8-scaled"
  if [ ! -f "$GEMMA_MARKER" ] && [ -f "$GEMMA_MODEL" ]; then
    echo "worker-ltx-video: Modelo Gemma antigo (GitMylo) detectado, trocando para Comfy-Org oficial..."
    rm -f "$GEMMA_MODEL"
  fi

  if ! check_size "$GEMMA_MODEL" "$GEMMA_MIN"; then
    echo "worker-ltx-video: Baixando Gemma 3 FP8 Scaled (Comfy-Org oficial)..."
    mkdir -p "$GEMMA_DIR"
    wget --progress=dot:giga \
      -O "$GEMMA_MODEL" \
      "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"
    touch "$GEMMA_MARKER"
  fi

  if [ ! -f "$TOKENIZER" ]; then
    echo "worker-ltx-video: Baixando tokenizer Gemma..."
    mkdir -p "$GEMMA_DIR"
    wget -q -O "$GEMMA_DIR/tokenizer.model" \
      "https://huggingface.co/jscheah/gemma3-tokenizer/resolve/main/tokenizer.model"
    wget -q -O "$GEMMA_DIR/tokenizer_config.json" \
      "https://huggingface.co/jscheah/gemma3-tokenizer/resolve/main/tokenizer_config.json"
    wget -q -O "$GEMMA_DIR/tokenizer.json" \
      "https://huggingface.co/jscheah/gemma3-tokenizer/resolve/main/tokenizer.json"
    wget -q -O "$GEMMA_DIR/special_tokens_map.json" \
      "https://huggingface.co/jscheah/gemma3-tokenizer/resolve/main/special_tokens_map.json"
  fi

  # config.json e generation_config.json (exigidos por Gemma3ForConditionalGeneration.from_pretrained)
  if [ ! -f "$GEMMA_DIR/config.json" ]; then
    echo "worker-ltx-video: Baixando config.json do text_encoder..."
    wget -q -O "$GEMMA_DIR/config.json" \
      "https://huggingface.co/Lightricks/LTX-2/resolve/main/text_encoder/config.json"
    wget -q -O "$GEMMA_DIR/generation_config.json" \
      "https://huggingface.co/Lightricks/LTX-2/resolve/main/text_encoder/generation_config.json"
    echo "worker-ltx-video: config.json criado"
  fi

  # preprocessor_config.json (exigido pelo LTXVGemmaCLIPModelLoader)
  if [ ! -f "$GEMMA_DIR/preprocessor_config.json" ]; then
    cat > "$GEMMA_DIR/preprocessor_config.json" << 'PPEOF'
{"do_convert_rgb":true,"do_normalize":true,"do_pan_and_scan":false,"do_rescale":true,"do_resize":true,"image_mean":[0.5,0.5,0.5],"image_processor_type":"Gemma3ImageProcessor","image_seq_length":256,"image_std":[0.5,0.5,0.5],"processor_class":"Gemma3Processor","resample":2,"rescale_factor":0.00392156862745098,"size":{"height":896,"width":896}}
PPEOF
    echo "worker-ltx-video: preprocessor_config.json criado"
  fi

  # Distilled LoRA (melhora qualidade com modelo dev)
  LORA="$VOLUME/models/loras/ltx-2-19b-distilled-lora-384.safetensors"
  if [ ! -f "$LORA" ]; then
    echo "worker-ltx-video: Baixando Distilled LoRA..."
    mkdir -p "$VOLUME/models/loras"
    wget --progress=dot:giga \
      -O "$LORA" \
      "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors"
  fi

  # Spatial Upscaler 2x (para pipeline de 2 estágios)
  UPSCALER="$VOLUME/models/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
  if [ ! -f "$UPSCALER" ]; then
    echo "worker-ltx-video: Baixando Spatial Upscaler 2x..."
    mkdir -p "$VOLUME/models/latent_upscale_models"
    wget --progress=dot:giga \
      -O "$UPSCALER" \
      "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors"
  fi

  # Compatibilidade com caminhos legados
  mkdir -p "$VOLUME/models/upscale_models"
  ln -sf "../latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors" \
    "$VOLUME/models/upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors"

  echo "worker-ltx-video: Modelos prontos no volume!"
else
  echo "worker-ltx-video: Sem network volume, usando modelos do container"
fi
# ===================================================================

echo "worker-ltx-video: Starting ComfyUI"

: "${COMFY_LOG_LEVEL:=DEBUG}"

if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout &

    echo "worker-ltx-video: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout &

    echo "worker-ltx-video: Starting RunPod Handler"
    python -u /handler.py
fi
