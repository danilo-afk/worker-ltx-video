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

  CKPT="$VOLUME/models/checkpoints/ltx-2-19b-dev-fp8.safetensors"
  GEMMA_DIR="$VOLUME/models/text_encoders/gemma-3-fp8"
  GEMMA_MODEL="$GEMMA_DIR/model.safetensors"
  TOKENIZER="$GEMMA_DIR/tokenizer.model"

  if [ ! -f "$CKPT" ]; then
    echo "worker-ltx-video: Baixando LTX-2 checkpoint FP8..."
    mkdir -p "$VOLUME/models/checkpoints"
    wget --progress=dot:giga \
      -O "$CKPT" \
      "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors"
  fi

  if [ ! -f "$GEMMA_MODEL" ]; then
    echo "worker-ltx-video: Baixando Gemma 3 FP8 text encoder..."
    mkdir -p "$GEMMA_DIR"
    wget --progress=dot:giga \
      -O "$GEMMA_MODEL" \
      "https://huggingface.co/GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn/resolve/main/gemma_3_12B_it_fp8_e4m3fn.safetensors"
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

  # preprocessor_config.json (exigido pelo LTXVGemmaCLIPModelLoader)
  if [ ! -f "$GEMMA_DIR/preprocessor_config.json" ]; then
    cat > "$GEMMA_DIR/preprocessor_config.json" << 'PPEOF'
{"do_convert_rgb":true,"do_normalize":true,"do_pan_and_scan":false,"do_rescale":true,"do_resize":true,"image_mean":[0.5,0.5,0.5],"image_processor_type":"Gemma3ImageProcessor","image_seq_length":256,"image_std":[0.5,0.5,0.5],"processor_class":"Gemma3Processor","resample":2,"rescale_factor":0.00392156862745098,"size":{"height":896,"width":896}}
PPEOF
    echo "worker-ltx-video: preprocessor_config.json criado"
  fi

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
