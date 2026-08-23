import shutil
import tempfile
import unittest
from pathlib import Path
import torch
from mia_bot.export import StandaloneInferencePolicy, export_policy


class TestPolicyModel(unittest.TestCase):
    def test_policy_architecture_forward(self):
        policy = StandaloneInferencePolicy()
        policy.eval()

        dummy_input = torch.randn(4, 89)
        with torch.no_grad():
            output = policy(dummy_input)

        # Output should slice only the 8 action means
        self.assertEqual(output.shape, (4, 8))
        self.assertEqual(output.dtype, torch.float32)

    def test_policy_torchscript_tracing(self):
        policy = StandaloneInferencePolicy()
        policy.eval()

        dummy_obs = torch.zeros((1, 89), dtype=torch.float32)
        traced_model = torch.jit.trace(policy, dummy_obs)

        # Validate output of traced model matches original model
        test_input = torch.randn(2, 89)
        with torch.no_grad():
            orig_out = policy(test_input)
            traced_out = traced_model(test_input)

        self.assertTrue(torch.allclose(orig_out, traced_out, atol=1e-5))

    def test_export_policy_flow(self):
        temp_dir = Path(tempfile.mkdtemp())
        try:
            run_dir = temp_dir / "test_run"
            step_dir = run_dir / "100000"
            step_dir.mkdir(parents=True)

            policy = StandaloneInferencePolicy()
            dummy_state_dict = policy.model.state_dict()
            torch.save(dummy_state_dict, step_dir / "PPO_POLICY.pt")

            out_policy = temp_dir / "exported_policy.pt"
            export_policy(checkpoints_dir=str(temp_dir), output_file=str(out_policy))

            self.assertTrue(out_policy.exists())
            loaded = torch.jit.load(str(out_policy))
            loaded.eval()
            res = loaded(torch.zeros((1, 89)))
            self.assertEqual(res.shape, (1, 8))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
