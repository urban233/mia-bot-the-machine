FROM python:3.11-slim

# Install system build dependencies for C++ extensions and git dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set thread and GPU optimization variables
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install PyTorch with CUDA 12.4 support first (optimized cache layer)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu124

# Install RocketSim, RLGym-sim and RLGym-PPO dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    rocketsim>=2.1.0 \
    "rlgym-sim @ git+https://github.com/AechPro/rocket-league-gym-sim@main" \
    "rlgym-ppo @ git+https://github.com/AechPro/rlgym-ppo"

# Copy collision meshes for RocketSim arena simulation and project files
COPY collision_meshes ./collision_meshes
COPY train.py export.py ./

# Create data directory mount point for checkpoints
RUN mkdir -p /app/data

# Entrypoint configuration
ENTRYPOINT ["python", "train.py"]
CMD ["--resume"]

