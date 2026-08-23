<p align="center">
  <img src="assets/logo.svg" width="256" height="256" alt="MIA-BOT mark">
</p>

<h1 align="center">MIA-BOT: The Machine</h1>

A standalone, reinforcement learning-driven Rocket League agent trained with `rlgym-sim` and `rlgym-ppo`, optimized for CPU inference and deployment via RLBot. Built and tested with **Bazel 9** (Bzlmod, `rules_python 2.3.x`, `rules_pkg`, `rules_oci`).

---

## Prerequisites

* **Bazelisk / Bazel 9** (for hermetic builds, tests, packaging, and container images)
* **Python 3.11** (managed hermetically by Bazel, or via `uv` for standalone usage)
* **NVIDIA GPU** with CUDA support (for training)
* **[RLBot GUI](https://rlbot.org/)** (v4 / classic)

---

## Bazel Quick Start Workflow

### 1. Run Automated Test Suite
Run unit and integration tests (math, observation features, checkpoint rotation, policy model, and e2e pipeline):

```bash
bazel test //...
```

### 2. Build RLBot Distribution Release Bundle
Hermetically package the bot into `mia_bot.zip` and `mia_bot.tar.gz` via `rules_pkg`:

```bash
bazel build //:bot_bundle_zip //:bot_bundle_tar
# Output generated at: bazel-bin/mia_bot.zip
```

### 3. Build & Load Container Image for GPU Training
Build and load the container image into your local Docker daemon:

```bash
bazel run //container:training_tarball
```

### 4. Run Training via Bazel
Run GPU-accelerated training directly:

```bash
bazel run //:train -- --resume --n-proc 12
```

### 5. Export Policy via Bazel
Trace latest PPO checkpoint into TorchScript `policy.pt`:

```bash
bazel run //:export -- --checkpoints-dir data/checkpoints --output policy.pt
```

### 6. Run RLBot Bundler Script
Generate unpacked `dist/mia_bot/` folder for RLBot GUI:

```bash
bazel run //:bundle
```

---

## Docker GPU Training Workflow (WSL2 / Linux)

Training `rlgym-sim` inside Docker under Linux/WSL2 provides maximum IPC throughput and isolation.

### 1. Run with Helper Script
```bash
chmod +x ./docker/docker-run.sh
./docker/docker-run.sh --resume --n-proc 12
```

### 2. Run with Docker Compose
```bash
# Build image
docker compose -f docker/docker-compose.yml build

# Start interactive training
docker compose -f docker/docker-compose.yml run --rm train --n-proc 12

# Export policy weights
docker compose -f docker/docker-compose.yml run --rm export
```

### 3. Windows PowerShell
```powershell
.\docker\docker-run.ps1 -PassthroughArgs "--resume", "--n-proc", "12"
```

---

## Standalone / uv Workflow

For quick iterations without Bazel:

**1. Install dependencies:**
```powershell
uv sync
```

**2. Train the bot:**
```powershell
uv run python -m mia_bot.train
```

**3. Export trained policy to TorchScript:**
```powershell
uv run python -m mia_bot.export
```

**4. Bundle files for distribution:**
```powershell
uv run python -m mia_bot.bundle
```

**5. Play in RLBot:**
* Open **RLBot GUI**.
* Click **Add** $\rightarrow$ **Load Folder** and select `dist/mia_bot` (or load `dist/mia_bot.zip` / `bazel-bin/mia_bot.zip`).
* Add `ML_Demon_Bot` to a team and click **Start Match**.

---

## Training CLI Flags & Options

`train.py` supports configurable arguments for automated checkpointing, auto-resuming, and parallel worker allocation:

```bash
bazel run //:train -- [OPTIONS]
# or: uv run python -m mia_bot.train [OPTIONS]
# or: ./docker/docker-run.sh [OPTIONS]
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--resume`, `-r` | `flag` | `False` | Automatically finds and resumes from the highest-step checkpoint across all previous runs. |
| `--save-every` | `int` | `100_000` | Timestep interval between saving checkpoints (~1–2 minutes at 1.5k ts/s). |
| `--max-checkpoints` | `int` | `3` | Maximum number of concurrent checkpoint folders to retain before automatically pruning older ones. |
| `--n-proc` | `int` | `10` | Number of parallel CPU simulation processes running physics workers. |

### Interactive Terminal Controls

While training is running, enter commands into the terminal:

* **`p` + Enter**: Pause / unpause environment stepping.
* **`c` + Enter**: Force an immediate checkpoint save.
* **`q` + Enter**: Save a checkpoint and cleanly exit after the current iteration.

---

## Skill Milestones & Estimated Runtimes

*Hardware baseline: 10 CPU workers + RTX 4060 (~1,500 ts/s $\approx$ 5.4M timesteps/hour)*

| Target Skill Level | Timestep Range | Est. Training Time | Observable Mechanics |
| --- | --- | --- | --- |
| **Bronze / Silver** | 2M – 5M | ~30 – 60 min | Drives toward ball, consistent ground hits, stops wall-circling. |
| **Gold / Platinum** | 10M – 25M | ~2 – 5 hours | Accurate net shots, basic single-jump challenges, goal line defense. |
| **Diamond / Champion** | 50M – 100M | ~10 – 20 hours | Fast aerials, wall rebounds, backpost rotations, boost pickup logic. |
| **Grand Champion / SSL** | 250M – 500M+ | ~48 – 100+ hours | Air dribbles, flip resets, fast kickoffs, ceiling shots. |

---

## Project Structure

```text
mia-bot/
├── MODULE.bazel          # Bzlmod module definition (Bazel 9, rules_python, rules_pkg, rules_oci)
├── .bazelversion         # Pinned Bazel version (9.2.0)
├── .bazelrc              # Hermetic build flags & execution presets
├── .bazelignore          # Excludes data/, dist/, .venv/ from Bazel
├── .dockerignore         # Docker context exclusions
├── BUILD.bazel           # Root Bazel targets (aliases to //src/mia_bot and release packaging)
├── WORKSPACE.bazel       # Root workspace marker for IDE extension compatibility
├── src/
│   └── mia_bot/
│       ├── BUILD.bazel   # Package targets (bot_lib, bot_bin, train, export, bundle)
│       ├── __init__.py   # Package initialization
│       ├── bot.py        # Standalone in-game inference agent (with runfiles support)
│       ├── bundle.py     # Distribution packager
│       ├── export.py     # Traces PyTorch checkpoint into TorchScript policy.pt
│       ├── main.py       # Quick smoke test entry point
│       └── train.py      # Headless PPO training script
├── docker/
│   ├── Dockerfile        # Layer-cached GPU training container definition
│   ├── docker-compose.yml# Compose specification for train & export
│   ├── docker-run.sh     # Linux/WSL2 launch helper
│   └── docker-run.ps1    # PowerShell launch helper
├── container/
│   └── BUILD.bazel       # rules_oci container image definitions
├── tests/
│   ├── BUILD.bazel       # py_test targets (unit & end-to-end pipeline)
│   ├── test_math.py
│   ├── test_observation.py
│   ├── test_checkpoints.py
│   ├── test_policy_model.py
│   ├── test_bundle.py
│   └── test_e2e_pipeline.py
├── .github/workflows/
│   └── ci.yml            # GitHub Actions CI workflow
├── appearance.cfg        # Car cosmetics and paint finish
├── bot.cfg               # RLBot metadata and configuration
├── policy.pt             # Exported actor model weights
├── pyproject.toml        # Standalone Python dependency metadata (src layout)
├── requirements.txt      # RLBot GUI runtime dependencies
├── requirements_lock.txt # Hermetically pinned dependencies for rules_python
└── data/checkpoints/     # Auto-rotated model checkpoints (ignored by Bazel)
```
