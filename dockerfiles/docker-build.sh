#!/usr/bin/env sh
set -eu

LLAMA_ARCHIVE="llama.cpp.b9382.tar.xz"
MINJA_ARCHIVE="minja.tar.xz"
CONTROL_PANEL_PORT="6767"
UNGULATE_PORT="6868"
GPU_FLAGS="${GPU_FLAGS:---gpus all}"
IMAGE_NAME="${IMAGE_NAME:-ungulate:latest}"

if [ ! -f "$LLAMA_ARCHIVE" ]; then
    echo "Missing archive: $LLAMA_ARCHIVE"
    echo "Place llama.cpp.b9382.tar.xz next to this script before running docker build."
    exit 1
fi

if [ ! -f "$MINJA_ARCHIVE" ]; then
    echo "Missing archive: $MINJA_ARCHIVE"
    echo "Place minja.tar.xz next to this script before running docker build."
    exit 1
fi

docker build -t "$IMAGE_NAME" -f Dockerfile .

sudo docker volume create ungulate_data

echo "Run with NVIDIA GPU support using:"
echo "sudo docker run -it --rm -p $CONTROL_PANEL_PORT:$CONTROL_PANEL_PORT -p $UNGULATE_PORT:$UNGULATE_PORT -v ungulate_data:/root/.config/ungulate/server $GPU_FLAGS $IMAGE_NAME"
