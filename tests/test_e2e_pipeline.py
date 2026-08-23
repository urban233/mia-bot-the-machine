import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import torch

from mia_bot.export import StandaloneInferencePolicy, export_policy
from mia_bot.bot import MLBot, get_rotation_matrix


class TestE2EPipeline(unittest.TestCase):
    def test_e2e_training_export_and_packaging_pipeline(self):
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # 1. Simulate training output / checkpoint creation
            checkpoints_root = temp_dir / "data" / "checkpoints"
            run_dir = checkpoints_root / "rlgym-ppo-run-e2e"
            step_dir = run_dir / "500000"
            step_dir.mkdir(parents=True, exist_ok=True)

            policy_model = StandaloneInferencePolicy()
            dummy_state_dict = policy_model.model.state_dict()
            torch.save(dummy_state_dict, step_dir / "PPO_POLICY.pt")

            # 2. Test policy export step
            exported_policy_path = temp_dir / "policy.pt"
            export_policy(
                checkpoints_dir=str(checkpoints_root),
                output_file=str(exported_policy_path),
            )

            self.assertTrue(exported_policy_path.exists())
            self.assertGreater(exported_policy_path.stat().st_size, 0)

            # 3. Test standalone inference using the exported policy
            loaded_model = torch.jit.load(str(exported_policy_path))
            loaded_model.eval()

            dummy_obs = torch.randn(1, 89)
            with torch.no_grad():
                actions = loaded_model(dummy_obs).squeeze().numpy()

            self.assertEqual(actions.shape, (8,))
            # Validate action ranges
            self.assertTrue(-10.0 <= actions[0] <= 10.0)

            # 4. Test distribution packaging
            dist_dir = temp_dir / "dist"
            bundle_root = temp_dir / "bot_workspace"
            bundle_root.mkdir(parents=True)

            shutil.copy2("src/mia_bot/bot.py", bundle_root / "bot.py")
            shutil.copy2("bot.cfg", bundle_root / "bot.cfg")
            shutil.copy2("appearance.cfg", bundle_root / "appearance.cfg")
            shutil.copy2("requirements.txt", bundle_root / "requirements.txt")
            shutil.copy2(exported_policy_path, bundle_root / "policy.pt")

            zip_output = dist_dir / "mia_bot.zip"
            target_dir = dist_dir / "mia_bot"
            target_dir.mkdir(parents=True, exist_ok=True)

            for fname in ["bot.py", "bot.cfg", "appearance.cfg", "requirements.txt", "policy.pt"]:
                shutil.copy2(bundle_root / fname, target_dir / fname)

            with zipfile.ZipFile(zip_output, "w", zipfile.ZIP_DEFLATED) as archive:
                for file_path in target_dir.rglob("*"):
                    archive.write(file_path, file_path.relative_to(dist_dir))

            self.assertTrue(zip_output.exists())
            self.assertGreater(zip_output.stat().st_size, 0)

            # Verify zip archive content
            with zipfile.ZipFile(zip_output, "r") as archive:
                names = archive.namelist()
                expected_members = [
                    "mia_bot/bot.py",
                    "mia_bot/bot.cfg",
                    "mia_bot/appearance.cfg",
                    "mia_bot/requirements.txt",
                    "mia_bot/policy.pt",
                ]
                for member in expected_members:
                    self.assertIn(member, names, f"{member} missing from release zip")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
