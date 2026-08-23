# ==============================================================================
# MIA-BOT: GPU-Accelerated RL Training Container
# ==============================================================================
FROM python:3.11-slim

# Install minimal OS dependencies for C++ extensions, git dependencies, and SSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Performance and GPU optimization environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    MIA_HARDWARE_PROFILE=ryzen_3400g_rtx_4060 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Layer 1: Install PyTorch with CUDA 12.4 support (isolated cache layer)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu124

# Layer 2: Install RocketSim and RLGym-PPO simulation dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    rocketsim>=2.1.0 \
    "rlgym-sim @ git+https://github.com/AechPro/rocket-league-gym-sim@main" \
    "rlgym-ppo @ git+https://github.com/AechPro/rlgym-ppo"

# Layer 3: Copy simulation arena collision meshes and source packages
COPY collision_meshes ./collision_meshes
COPY src ./src

# Create mount points for checkpoint persistence and policy export
RUN mkdir -p /app/data /app/out

# Allow graceful checkpoint save on container stop
STOPSIGNAL SIGINT

# Default entrypoint runs training with auto-resume
ENTRYPOINT ["python", "-m", "mia_bot.train"]
CMD ["--resume"]

