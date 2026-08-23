import shutil
import tempfile
import unittest
from pathlib import Path
from mia_bot.train import calculate_heuristic_elo, find_latest_checkpoint, prune_old_checkpoints


class TestCheckpoints(unittest.TestCase):
    def test_calculate_heuristic_elo(self):
        # 0 steps -> Unranked
        curr, nxt = calculate_heuristic_elo(0)
        self.assertIn("Unranked", curr[1])
        self.assertIsNotNone(nxt)

        # 3,000,000 steps -> Bronze / Silver
        curr, nxt = calculate_heuristic_elo(3_000_000)
        self.assertIn("Bronze / Silver", curr[1])

        # 350,000,000 steps -> SSL
        curr, nxt = calculate_heuristic_elo(350_000_000)
        self.assertIn("SSL", curr[1])
        self.assertIsNone(nxt)

    def test_find_latest_checkpoint(self):
        temp_dir = Path(tempfile.mkdtemp())
        try:
            run_dir = temp_dir / "run-1"
            step_100 = run_dir / "100"
            step_200 = run_dir / "200"
            step_100.mkdir(parents=True)
            step_200.mkdir(parents=True)

            latest = find_latest_checkpoint(str(temp_dir))
            self.assertIsNotNone(latest)
            self.assertTrue(latest.endswith("200"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_prune_old_checkpoints(self):
        temp_dir = Path(tempfile.mkdtemp())
        try:
            run_dir = temp_dir / "run-1"
            run_dir.mkdir(parents=True)
            for step in [100, 200, 300, 400, 500]:
                (run_dir / str(step)).mkdir()

            # Prune with max_keep=3 -> should keep 300, 400, 500
            prune_old_checkpoints(checkpoints_dir=str(temp_dir), max_keep=3)
            remaining = sorted([int(d.name) for d in run_dir.iterdir() if d.is_dir()])
            self.assertEqual(remaining, [300, 400, 500])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
