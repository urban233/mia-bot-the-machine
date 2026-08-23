import argparse
import math
import os
import shutil
import sys
from pathlib import Path
import rlgym_sim
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import (
    FaceBallReward,
    TouchBallReward,
    VelocityBallToGoalReward,
    VelocityPlayerToBallReward,
)
from rlgym_sim.utils.terminal_conditions.common_conditions import (
    GoalScoredCondition,
    TimeoutCondition,
)
from rlgym_ppo import Learner


# ==============================================================================
# HEURISTIC ELO & RANK TIERS
# ==============================================================================
RANKS = [
    (0,           "Unranked / Rookie",      "100 - 450",   "Basic throttle/steering discovery"),
    (2_000_000,   "Bronze / Silver",        "450 - 700",   "Intentional ground touches"),
    (5_000_000,   "Gold",                   "700 - 850",   "Net aiming & basic saves"),
    (15_000_000,  "Platinum",               "850 - 1000",  "Single-jump challenges & recoveries"),
    (35_000_000,  "Diamond",                "1000 - 1200", "Wall reads & basic aerial touches"),
    (75_000_000,  "Champion",               "1200 - 1500", "Fast aerials & backpost defense"),
    (150_000_000, "Grand Champion",         "1500 - 1850", "Consistent flicks, passes & air dribbles"),
    (300_000_000, "Supersonic Legend (SSL)", "1850+",      "Flip resets, ceiling shots & fast kickoffs"),
]


def calculate_heuristic_elo(steps: int):
    current_tier = RANKS[0]
    next_tier = RANKS[1]

    for i in range(len(RANKS) - 1):
        if steps >= RANKS[i][0]:
            current_tier = RANKS[i]
            next_tier = RANKS[i + 1] if i + 1 < len(RANKS) else None

    return current_tier, next_tier


def print_elo_banner(cumulative_steps: int, steps_per_sec: float = 1350.0):
    current_tier, next_tier = calculate_heuristic_elo(cumulative_steps)
    min_steps, rank_name, mmr_range, mechanic = current_tier

    if next_tier:
        target_steps, next_rank_name, _, _ = next_tier
        steps_left = max(0, target_steps - cumulative_steps)
        time_left_sec = steps_left / max(steps_per_sec, 1.0)
        hours = int(time_left_sec // 3600)
        minutes = int((time_left_sec % 3600) // 60)
        progress_pct = min(100.0, (cumulative_steps / target_steps) * 100)
        next_info = f"{next_rank_name} ({steps_left:,} steps left | ~{hours}h {minutes:02d}m)"
    else:
        progress_pct = 100.0
        next_info = "Max Rank Reached (SSL+ Mastery)"

    bar_width = 30
    filled = int(bar_width * (progress_pct / 100.0))
    bar = "█" * filled + "░" * (bar_width - filled)

    banner = f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                            MIA-BOT RANK PROGRESS                             │
├──────────────────────────────────────────────────────────────────────────────┤
│  Current Steps : {cumulative_steps:>12,} steps                                     │
│  Estimated Rank: {rank_name:<24} (Heuristic MMR: {mmr_range})     │
│  Current Focus : {mechanic:<58} │
│  Next Rank     : {next_info:<58} │
│  Rank Progress : [{bar}] {progress_pct:>5.1f}%                   │
└──────────────────────────────────────────────────────────────────────────────┘
"""
    sys.__stdout__.write(banner + "\n")
    sys.__stdout__.flush()


# ==============================================================================
# CHECKPOINT MANAGEMENT & ROTATION
# ==============================================================================
def find_latest_checkpoint(checkpoints_dir: str = "data/checkpoints") -> str | None:
    base_path = Path(checkpoints_dir)
    if not base_path.exists():
        return None

    run_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
    step_dirs = [d for d in latest_run.iterdir() if d.is_dir() and d.name.isdigit()]
    if not step_dirs:
        return None

    latest_step_dir = max(step_dirs, key=lambda d: int(d.name))
    return str(latest_step_dir.resolve())


def prune_old_checkpoints(checkpoints_dir: str = "data/checkpoints", max_keep: int = 3):
    base_path = Path(checkpoints_dir)
    if not base_path.exists():
        return

    run_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not run_dirs:
        return

    latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
    step_dirs = [d for d in latest_run.iterdir() if d.is_dir() and d.name.isdigit()]

    if len(step_dirs) > max_keep:
        step_dirs.sort(key=lambda d: int(d.name))
        to_remove = step_dirs[:-max_keep]

        for old_dir in to_remove:
            try:
                shutil.rmtree(old_dir, ignore_errors=True)
                sys.__stdout__.write(f"[Checkpoint Rotation] Pruned old checkpoint: {old_dir.name}\n")
                sys.__stdout__.flush()
            except Exception as e:
                sys.__stdout__.write(f"[!] Failed to remove {old_dir.name}: {e}\n")


# ==============================================================================
# STDOUT INTERCEPTOR (CAPTURES REPORTS & CHECKPOINTS LIVE)
# ==============================================================================
class StdoutInterceptor:
    def __init__(self, original_stdout, max_keep: int = 3):
        self.original_stdout = original_stdout
        self.max_keep = max_keep
        self.buffer = ""

    def write(self, text):
        self.original_stdout.write(text)
        self.buffer += text

        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            self.buffer = lines[-1]
            for line in lines[:-1]:
                self.process_line(line)

    def process_line(self, line: str):
        # 1. Detect iteration steps report
        if "Cumulative Timesteps:" in line:
            try:
                val = line.split("Cumulative Timesteps:")[1].strip().replace(",", "")
                steps = int(val.split()[0])
                print_elo_banner(steps)
            except Exception:
                pass

        # 2. Detect checkpoint save event
        if "Checkpoint" in line and "saved!" in line:
            prune_old_checkpoints(max_keep=self.max_keep)

    def flush(self):
        self.original_stdout.flush()


# ==============================================================================
# ENVIRONMENT BUILDER
# ==============================================================================
def build_env():
    reward_functions = (
        VelocityPlayerToBallReward(),
        FaceBallReward(),
        TouchBallReward(aerial_weight=0.5),
        VelocityBallToGoalReward(),
    )
    reward_weights = (
        0.2,
        0.1,
        1.5,
        2.0,
    )

    reward_fn = CombinedReward(
        reward_functions=reward_functions,
        reward_weights=reward_weights,
    )

    terminal_conditions = [
        GoalScoredCondition(),
        TimeoutCondition(round(300 * 15 / 8)),
    ]

    return rlgym_sim.make(
        tick_skip=8,
        team_size=1,
        spawn_opponents=True,
        terminal_conditions=terminal_conditions,
        reward_fn=reward_fn,
    )


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLGym-PPO Training Script with Live ELO Progress")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from latest available checkpoint")
    parser.add_argument("--save-every", type=int, default=100_000, help="Timesteps between checkpoints (default: 100k)")
    parser.add_argument("--max-checkpoints", type=int, default=3, help="Max concurrent checkpoints to retain (default: 3)")
    parser.add_argument("--n-proc", type=int, default=10, help="Number of parallel physics workers (default: 10)")
    args = parser.parse_args()

    checkpoint_folder = None
    if args.resume:
        checkpoint_folder = find_latest_checkpoint()
        if checkpoint_folder:
            print(f"[+] Resuming training from checkpoint: {checkpoint_folder}")
        else:
            print("[!] No existing checkpoint found. Starting fresh run.")

    # Redirect stdout to capture all rlgym-ppo terminal output in real time
    sys.stdout = StdoutInterceptor(sys.__stdout__, max_keep=args.max_checkpoints)

    learner = Learner(
        env_create_function=build_env,
        n_proc=args.n_proc,
        min_inference_size=128,
        ppo_batch_size=50000,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),
        ts_per_iteration=50000,
        exp_buffer_size=100000,
        device="cuda",
        save_every_ts=args.save_every,
        checkpoint_load_folder=checkpoint_folder,
    )

    learner.learn()
