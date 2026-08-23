import argparse
from pathlib import Path
import torch
import torch.nn as nn


class StandaloneInferencePolicy(nn.Module):
    """Matches the exact rlgym-ppo checkpoint architecture and extracts action means."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(89, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 16),  # 8 action means + 8 action stds
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.model(obs)
        # Slices only the first 8 values (action means) for deterministic bot execution
        return out[..., :8]


def export_policy(checkpoints_dir: str = "data/checkpoints", output_file: str = "policy.pt"):
    base_path = Path(checkpoints_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Checkpoints directory does not exist: {base_path.resolve()}")

    # 1. Locate the latest run folder
    run_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run folders found inside '{checkpoints_dir}'")

    latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
    print(f"Found run directory: {latest_run.name}")

    # 2. Locate the highest timestep checkpoint folder (e.g., 400046)
    step_dirs = [d for d in latest_run.iterdir() if d.is_dir() and d.name.isdigit()]
    if not step_dirs:
        raise FileNotFoundError(f"No numbered checkpoint folders found inside '{latest_run.name}'")

    latest_step_dir = max(step_dirs, key=lambda d: int(d.name))
    policy_src = latest_step_dir / "PPO_POLICY.pt"
    print(f"Loading checkpoint weights from: {policy_src}")

    # 3. Load weights into the standalone model architecture
    state_dict = torch.load(policy_src, map_location="cpu")
    policy = StandaloneInferencePolicy()
    if any(k.startswith("model.") for k in state_dict.keys()):
        policy.load_state_dict(state_dict)
    else:
        policy.model.load_state_dict(state_dict)
    policy.eval()

    # 4. Trace and export as standalone TorchScript
    dummy_obs = torch.zeros((1, 89), dtype=torch.float32)
    traced_model = torch.jit.trace(policy, dummy_obs)
    traced_model.save(output_file)

    print(f"\n[+] Successfully exported TorchScript policy to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export latest rlgym-ppo checkpoint to TorchScript policy.pt")
    parser.add_argument("--checkpoints-dir", default="data/checkpoints", help="Directory containing checkpoint runs")
    parser.add_argument("--output", "-o", default="policy.pt", help="Output TorchScript file path")
    args = parser.parse_args()

    export_policy(checkpoints_dir=args.checkpoints_dir, output_file=args.output)

