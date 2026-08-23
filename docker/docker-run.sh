#!/usr/bin/env bash
set -e

# Resolve repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="mia-bot-train:latest"

# Check if image exists on host, build if not
if [[ "$(docker images -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
    echo "[+] Building Docker image ${IMAGE_NAME}..."
    docker build -t ${IMAGE_NAME} -f "${SCRIPT_DIR}/Dockerfile" "${REPO_ROOT}"
fi

# Ensure host data directory exists
mkdir -p "${REPO_ROOT}/data"

echo "[+] Launching MIA-BOT training container with GPU acceleration..."
docker run --rm -it \
    --gpus all \
    --ipc host \
    --shm-size 2gb \
    -v "${REPO_ROOT}/data:/app/data" \
    ${IMAGE_NAME} "$@"

