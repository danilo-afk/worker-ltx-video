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

  # Evita corrida entre múltiplos workers escrevendo os mesmos modelos no volume.
  LOCK_FILE="$VOLUME/.model-bootstrap.lock"
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    if ! flock -w 1800 9; then
      echo "worker-ltx-video: timeout aguardando lock de bootstrap ($LOCK_FILE)" >&2
      exit 1
    fi
    echo "worker-ltx-video: lock de bootstrap adquirido"
  else
    echo "worker-ltx-video: flock não encontrado; bootstrap seguirá sem lock" >&2
  fi

  # Checkpoint padrão para produção: distilled FP8 (mais estável para I2V no endpoint atual).
  CKPT_NAME="${LTX_CKPT_NAME:-ltx-2-19b-distilled-fp8.safetensors}"
  case "$CKPT_NAME" in
    ltx-2-19b-dev-fp8.safetensors)
      CKPT_URL="https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors"
      CKPT_MIN=25000000000
      ;;
    ltx-2-19b-distilled-fp8.safetensors)
      CKPT_URL="https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-fp8.safetensors"
      CKPT_MIN=8500000000
      ;;
    *)
      echo "worker-ltx-video: LTX_CKPT_NAME inválido: $CKPT_NAME" >&2
      exit 1
      ;;
  esac
  CKPT="$VOLUME/models/checkpoints/$CKPT_NAME"
  GEMMA_DIR="$VOLUME/models/text_encoders/gemma-3-fp8"
  GEMMA_OFFICIAL_DIR="$VOLUME/models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized"
  GEMMA_MODEL="$GEMMA_DIR/model.safetensors"
  TOKENIZER="$GEMMA_DIR/tokenizer.model"

  # Tamanhos mínimos esperados (bytes) - detecta downloads incompletos.
  # O valor do checkpoint depende do arquivo escolhido (dev vs distilled).
  GEMMA_MIN=8500000000

  validate_safetensors_coverage() {
    local file="$1"
    python - "$file" <<'PY'
import json
import os
import struct
import sys

path = sys.argv[1]
size = os.path.getsize(path)
if size < 8:
    raise SystemExit("arquivo menor que 8 bytes")

with open(path, "rb") as f:
    header_len_raw = f.read(8)
    if len(header_len_raw) != 8:
        raise SystemExit("falha ao ler tamanho do header")
    header_len = struct.unpack("<Q", header_len_raw)[0]
    if header_len <= 0 or (8 + header_len) > size:
        raise SystemExit(
            f"header inválido: header_len={header_len}, file_size={size}"
        )
    header_bytes = f.read(header_len)
    if len(header_bytes) != header_len:
        raise SystemExit("header incompleto")

try:
    header = json.loads(header_bytes)
except Exception as e:
    raise SystemExit(f"header JSON inválido: {e}")

max_end = 0
for key, value in header.items():
    if key == "__metadata__":
        continue
    if not isinstance(value, dict):
        raise SystemExit(f"tensor {key} inválido: entrada não é dict")
    data_offsets = value.get("data_offsets")
    if not isinstance(data_offsets, list) or len(data_offsets) != 2:
        raise SystemExit(f"tensor {key} sem data_offsets válidos")
    start, end = data_offsets
    if not isinstance(start, int) or not isinstance(end, int):
        raise SystemExit(f"tensor {key} offsets não inteiros")
    if start < 0 or end < start:
        raise SystemExit(f"tensor {key} offsets inválidos: {start}, {end}")
    max_end = max(max_end, end)

required_size = 8 + header_len + max_end
if required_size > size:
    raise SystemExit(
        f"arquivo incompleto: required_size={required_size}, file_size={size}"
    )
PY
  }

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
      if [[ "$file" == *.safetensors ]]; then
        if ! validate_safetensors_coverage "$file" >/tmp/worker_safetensors_check.log 2>&1; then
          echo "worker-ltx-video: $file inválido (safetensors), re-baixando..."
          cat /tmp/worker_safetensors_check.log >&2 || true
          rm -f "$file"
          return 1
        fi
      fi
    else
      return 1
    fi
    return 0
  }

  download_with_validation() {
    local file="$1" min="$2" url="$3" label="$4"
    local max_attempts=3
    local attempt=1
    while [ "$attempt" -le "$max_attempts" ]; do
      echo "worker-ltx-video: Baixando ${label} (tentativa ${attempt}/${max_attempts})..."
      mkdir -p "$(dirname "$file")"
      rm -f "$file"
      if wget --progress=dot:giga -O "$file" "$url" && check_size "$file" "$min"; then
        return 0
      fi
      echo "worker-ltx-video: Falha ao validar ${label} na tentativa ${attempt}" >&2
      attempt=$((attempt + 1))
    done
    echo "worker-ltx-video: ERRO ao baixar ${label} após ${max_attempts} tentativas." >&2
    return 1
  }

  resolve_download_url() {
    local label="$1"
    shift
    local candidate
    for candidate in "$@"; do
      [ -z "$candidate" ] && continue
      local code
      code=$(curl -L -s -o /dev/null -w "%{http_code}" --range 0-0 "$candidate" || true)
      if [ "$code" = "200" ] || [ "$code" = "206" ]; then
        echo "$candidate"
        return 0
      fi
      echo "worker-ltx-video: ${label} URL indisponível (${code}): $candidate" >&2
    done
    return 1
  }

  if ! check_size "$CKPT" "$CKPT_MIN"; then
    if ! download_with_validation "$CKPT" "$CKPT_MIN" "$CKPT_URL" "checkpoint LTX-2 ($CKPT_NAME)"; then
      exit 1
    fi
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

  GEMMA_URL_OVERRIDE="${LTX_GEMMA_URL:-}"
  GEMMA_URL=$(resolve_download_url \
    "Gemma" \
    "$GEMMA_URL_OVERRIDE" \
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors" \
    "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main/model-00001-of-00005.safetensors")
  if [ -z "$GEMMA_URL" ]; then
    echo "worker-ltx-video: ERRO sem URL válida para Gemma." >&2
    exit 1
  fi
  echo "worker-ltx-video: Gemma URL selecionada: $GEMMA_URL"
  if ! check_size "$GEMMA_MODEL" "$GEMMA_MIN"; then
    if ! download_with_validation "$GEMMA_MODEL" "$GEMMA_MIN" "$GEMMA_URL" "Gemma 3 FP8 Scaled"; then
      exit 1
    fi
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

  # Compatibilidade com o path do workflow oficial LTX-2 I2V.
  mkdir -p "$GEMMA_OFFICIAL_DIR"
  ln -sf "../gemma-3-fp8/model.safetensors" \
    "$GEMMA_OFFICIAL_DIR/model.safetensors"
  ln -sf "../gemma-3-fp8/model.safetensors" \
    "$GEMMA_OFFICIAL_DIR/model-00001-of-00005.safetensors"
  ln -sf "../gemma-3-fp8/tokenizer.model" \
    "$GEMMA_OFFICIAL_DIR/tokenizer.model"
  ln -sf "../gemma-3-fp8/tokenizer_config.json" \
    "$GEMMA_OFFICIAL_DIR/tokenizer_config.json"
  ln -sf "../gemma-3-fp8/tokenizer.json" \
    "$GEMMA_OFFICIAL_DIR/tokenizer.json"
  ln -sf "../gemma-3-fp8/special_tokens_map.json" \
    "$GEMMA_OFFICIAL_DIR/special_tokens_map.json"
  ln -sf "../gemma-3-fp8/config.json" \
    "$GEMMA_OFFICIAL_DIR/config.json"
  ln -sf "../gemma-3-fp8/generation_config.json" \
    "$GEMMA_OFFICIAL_DIR/generation_config.json"
  ln -sf "../gemma-3-fp8/preprocessor_config.json" \
    "$GEMMA_OFFICIAL_DIR/preprocessor_config.json"

  # Distilled LoRA (melhora qualidade com modelo dev)
  LORA="$VOLUME/models/loras/ltx-2-19b-distilled-lora-384.safetensors"
  LORA_MIN=1000000
  LORA_URL="https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors"
  if ! check_size "$LORA" "$LORA_MIN"; then
    if ! download_with_validation "$LORA" "$LORA_MIN" "$LORA_URL" "Distilled LoRA"; then
      exit 1
    fi
  fi

  # Spatial Upscaler 2x (para pipeline de 2 estágios)
  UPSCALER="$VOLUME/models/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
  UPSCALER_MIN=1000000
  UPSCALER_URL="https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors"
  if ! check_size "$UPSCALER" "$UPSCALER_MIN"; then
    if ! download_with_validation "$UPSCALER" "$UPSCALER_MIN" "$UPSCALER_URL" "Spatial Upscaler 2x"; then
      exit 1
    fi
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
