#!/usr/bin/env bash
set -e

IMAGE_NAME="mia-bot-train:latest"

# Check if image exists, build if not
if [[ "$(docker images -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
    echo "[+] Building Docker image ${IMAGE_NAME}..."
    docker build -t ${IMAGE_NAME} .
fi

# Ensure data directory exists on host
mkdir -p "$(pwd)/data"

echo "[+] Launching mia-bot training container..."
docker run --rm -it \
    --gpus all \
    --shm-size 2gb \
    -v "$(pwd)/data:/app/data" \
    ${IMAGE_NAME} "$@"

