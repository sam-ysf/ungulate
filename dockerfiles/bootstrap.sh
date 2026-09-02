#!/bin/bash

VOLUME_NAME="ungulate_data"
VOLUME_DIR="/root/.config/ungulate/server"
MODEL_EMBEDDINGS="https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
MODEL_OCR=https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf 
MODEL_MMPROJ=https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF/resolve/main/mmproj-F16.gguf

# Maybe bootstrap ungulate configuration
if [ ! -f "$VOLUME_DIR/db.json" ]; then
    echo -e "\e[0;34m => [bootstrapping] Copied default database configuration to Docker volume '$VOLUME_NAME'\e[0m"
    cp "/root/.config/ungulate/bootstrap/config/db.json" "$VOLUME_DIR/db.json" 
fi

if [ ! -f "$VOLUME_DIR/model.json" ]; then
    echo -e "\e[0;34m => [bootstrapping] Copied default model configuration to Docker volume '$VOLUME_NAME'\e[0m"
    cp "/root/.config/ungulate/bootstrap/config/model.json" "$VOLUME_DIR/model.json" 
fi

if [ ! -f "$VOLUME_DIR/net.json" ]; then
    echo -e "\e[0;34m => [bootstrapping] Copied default network configuration to Docker volume '$VOLUME_NAME'\e[0m"
    cp "/root/.config/ungulate/bootstrap/config/net.json" "$VOLUME_DIR/net.json" 
fi

# Maybe download default models
if [ ! -f "$VOLUME_DIR/ocr-model.gguf" ]; then
    echo -e "\e[0;34m => [bootstrapping] Missing OCR model, fetching and saving default multimodal model to Docker volume '$VOLUME_NAME'\e[0m"
    echo -e "\e[0;34m => => $MODEL_OCR\e[0m"
    curl -L $MODEL_OCR --output $VOLUME_DIR/ocr-model.gguf
    echo -e "\e[0;34m => => $MODEL_MMPROJ\e[0m"
    curl -L $MODEL_MMPROJ --output $VOLUME_DIR/ocr-mmproj-f16.gguf
fi

if [ ! -f "$VOLUME_DIR/ocr-mmproj-f16.gguf" ]; then
    echo -e "\e[0;34m => [bootstrapping] Missing OCR model, fetching and saving default multimodal model to Docker volume '$VOLUME_NAME'\e[0m"
    echo -e "\e[0;34m => => $MODEL_OCR\e[0m"
    curl -L $MODEL_OCR --output $VOLUME_DIR/ocr-model.gguf
    echo -e "\e[0;34m => => $MODEL_MMPROJ\e[0m"
    curl -L $MODEL_MMPROJ --output $VOLUME_DIR/ocr-mmproj-f16.gguf
fi

if [ ! -f "$VOLUME_DIR/embeddings-model.gguf" ]; then
    echo -e "\e[0;34m => [bootstrapping] Missing embeddings model, fetching and saving default model to Docker volume '$VOLUME_NAME'\e[0m"
    echo -e "\e[0;34m => => $MODEL_EMBEDDINGS\e[0m"
    curl -L $MODEL_EMBEDDINGS --output $VOLUME_DIR/embeddings-model.gguf
fi

# Start server
/usr/local/bin/ungulate