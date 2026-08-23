"""Hardware configuration and optimization subsystem for MIA-BOT training.

Provides hardware abstraction profiles, PyTorch CUDA/Tensor Core optimizations,
and dynamic system resource detection for local and containerized training.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class HardwareConfig:
    """Hardware and execution parameters for RLGym-PPO training."""

    name: str
    description: str
    n_proc: int = 6
    min_inference_size: int = 64
    ppo_batch_size: int = 50_000
    ppo_minibatch_size: Optional[int] = 25_000
    ppo_epochs: int = 10
    ts_per_iteration: int = 50_000
    exp_buffer_size: int = 100_000
    policy_layer_sizes: Tuple[int, ...] = (256, 256, 256)
    critic_layer_sizes: Tuple[int, ...] = (256, 256, 256)
    device: str = "cuda"
    shm_buffer_size: int = 8192
    instance_launch_delay: Optional[float] = 0.05
    enable_tf32: bool = True
    enable_cudnn_benchmark: bool = True
    num_threads: int = 1


# ==============================================================================
# BUILT-IN HARDWARE PROFILES
# ==============================================================================

HARDWARE_PROFILES: Dict[str, HardwareConfig] = {
    "ryzen_3400g_rtx_4060": HardwareConfig(
        name="ryzen_3400g_rtx_4060",
        description="AMD Ryzen 5 3400G (4C/8T, 16GB RAM) + NVIDIA RTX 4060 (8GB VRAM, Ada Lovelace)",
        n_proc=6,  # 6 workers leaves 2 threads for learner process, CUDA driver dispatch & IPC
        min_inference_size=64,  # Matched for 6 workers * 2 cars (12 simulation agents)
        ppo_batch_size=50_000,
        ppo_minibatch_size=25_000,  # 2 minibatches per epoch maximizes Tensor Core throughput
        ppo_epochs=10,
        ts_per_iteration=50_000,
        exp_buffer_size=100_000,
        device="cuda",
        shm_buffer_size=8192,
        instance_launch_delay=0.05,
        enable_tf32=True,
        enable_cudnn_benchmark=True,
        num_threads=1,
    ),
    "mid_tier_cpu_gpu": HardwareConfig(
        name="mid_tier_cpu_gpu",
        description="Mid-tier 6C/12T or 8C/16T CPU + RTX 3060/4060 (8-12GB VRAM)",
        n_proc=8,
        min_inference_size=80,
        ppo_batch_size=50_000,
        ppo_minibatch_size=25_000,
        ppo_epochs=10,
        ts_per_iteration=50_000,
        exp_buffer_size=100_000,
        device="cuda",
        shm_buffer_size=8192,
        instance_launch_delay=0.05,
        enable_tf32=True,
        enable_cudnn_benchmark=True,
        num_threads=1,
    ),
    "rtx_4090_workstation": HardwareConfig(
        name="rtx_4090_workstation",
        description="High-end workstation (16+ CPU cores, 32GB+ RAM) + RTX 4090 (24GB VRAM)",
        n_proc=16,
        min_inference_size=128,
        ppo_batch_size=100_000,
        ppo_minibatch_size=50_000,
        ppo_epochs=10,
        ts_per_iteration=100_000,
        exp_buffer_size=200_000,
        device="cuda",
        shm_buffer_size=16384,
        instance_launch_delay=0.02,
        enable_tf32=True,
        enable_cudnn_benchmark=True,
        num_threads=1,
    ),
    "cpu_only_entry": HardwareConfig(
        name="cpu_only_entry",
        description="CPU-only execution fallback (no dedicated CUDA GPU)",
        n_proc=4,
        min_inference_size=32,
        ppo_batch_size=20_000,
        ppo_minibatch_size=10_000,
        ppo_epochs=8,
        ts_per_iteration=20_000,
        exp_buffer_size=40_000,
        device="cpu",
        shm_buffer_size=4096,
        instance_launch_delay=0.1,
        enable_tf32=False,
        enable_cudnn_benchmark=False,
        num_threads=1,
    ),
}

DEFAULT_PROFILE = "ryzen_3400g_rtx_4060"


def detect_auto_hardware_config() -> HardwareConfig:
    """Dynamically detects system resources and generates an optimized HardwareConfig."""
    cpu_threads = os.cpu_count() or 4
    n_proc = max(2, cpu_threads - 2) if cpu_threads > 4 else max(1, cpu_threads - 1)

    has_cuda = False
    device = "cpu"
    enable_tf32 = False
    enable_cudnn_benchmark = False

    try:
        import torch

        if torch.cuda.is_available():
            has_cuda = True
            device = "cuda"
            enable_tf32 = True
            enable_cudnn_benchmark = True
    except Exception:
        pass

    if has_cuda:
        desc = f"Auto-detected hardware: {cpu_threads} CPU threads, CUDA GPU acceleration enabled"
        return HardwareConfig(
            name="auto",
            description=desc,
            n_proc=min(n_proc, 16),
            min_inference_size=64 if n_proc <= 8 else 128,
            ppo_batch_size=50_000,
            ppo_minibatch_size=25_000,
            ppo_epochs=10,
            ts_per_iteration=50_000,
            exp_buffer_size=100_000,
            device=device,
            shm_buffer_size=8192,
            instance_launch_delay=0.05,
            enable_tf32=enable_tf32,
            enable_cudnn_benchmark=enable_cudnn_benchmark,
            num_threads=1,
        )
    else:
        desc = f"Auto-detected hardware: {cpu_threads} CPU threads, CPU-only fallback"
        return HardwareConfig(
            name="auto",
            description=desc,
            n_proc=min(n_proc, 6),
            min_inference_size=32,
            ppo_batch_size=20_000,
            ppo_minibatch_size=10_000,
            ppo_epochs=8,
            ts_per_iteration=20_000,
            exp_buffer_size=40_000,
            device="cpu",
            shm_buffer_size=4096,
            instance_launch_delay=0.1,
            enable_tf32=False,
            enable_cudnn_benchmark=False,
            num_threads=1,
        )


def get_hardware_config(profile_name: Optional[str] = None) -> HardwareConfig:
    """Resolves a HardwareConfig by profile name, environment variable, or default.

    Precedence order:
      1. Explicit `profile_name` parameter
      2. Environment variable `MIA_HARDWARE_PROFILE`
      3. Default profile (`ryzen_3400g_rtx_4060`)
    """
    selected = profile_name or os.environ.get("MIA_HARDWARE_PROFILE") or DEFAULT_PROFILE
    selected = selected.strip().lower()

    if selected == "auto":
        return detect_auto_hardware_config()

    if selected in HARDWARE_PROFILES:
        return HARDWARE_PROFILES[selected]

    available = list(HARDWARE_PROFILES.keys()) + ["auto"]
    sys.__stdout__.write(
        f"[!] Warning: Unknown hardware profile '{selected}'. Available: {available}. Falling back to default: '{DEFAULT_PROFILE}'\n"
    )
    sys.__stdout__.flush()
    return HARDWARE_PROFILES[DEFAULT_PROFILE]


def apply_hardware_optimizations(config: HardwareConfig) -> Dict[str, object]:
    """Applies PyTorch, CUDA, cuDNN, and multi-threading optimizations based on HardwareConfig."""
    import torch

    telemetry: Dict[str, object] = {
        "profile": config.name,
        "device": config.device,
        "n_proc": config.n_proc,
        "tf32_enabled": False,
        "cudnn_benchmark_enabled": False,
        "gpu_name": None,
    }

    # Bounded CPU intra-op threading to prevent process contention
    try:
        torch.set_num_threads(config.num_threads)
    except Exception:
        pass

    # CUDA & Tensor Core optimizations
    if config.device.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                telemetry["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                telemetry["gpu_name"] = "NVIDIA CUDA Device"

            if config.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                telemetry["cudnn_benchmark_enabled"] = True

            if config.enable_tf32:
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    if hasattr(torch, "set_float32_matmul_precision"):
                        torch.set_float32_matmul_precision("high")
                    telemetry["tf32_enabled"] = True
                except Exception:
                    pass
        else:
            sys.__stdout__.write(
                "[!] CUDA device requested but torch.cuda.is_available() is False. Falling back to CPU.\n"
            )
            sys.__stdout__.flush()
            telemetry["device"] = "cpu"

    return telemetry

