import configparser
from pathlib import Path
import unittest


class TestBundle(unittest.TestCase):
    def test_bot_cfg_structure(self):
        bot_cfg_path = Path("bot.cfg")
        self.assertTrue(bot_cfg_path.exists(), "bot.cfg must exist in root")

        parser = configparser.ConfigParser()
        parser.read(bot_cfg_path)

        self.assertIn("Locations", parser.sections())
        self.assertIn("Details", parser.sections())
        self.assertEqual(parser.get("Locations", "python_file"), "./bot.py")
        self.assertEqual(parser.get("Locations", "looks_config"), "./appearance.cfg")

    def test_appearance_cfg_structure(self):
        appearance_cfg_path = Path("appearance.cfg")
        self.assertTrue(appearance_cfg_path.exists(), "appearance.cfg must exist in root")

        parser = configparser.ConfigParser()
        parser.read(appearance_cfg_path)

        self.assertIn("Bot Loadout", parser.sections())
        self.assertIn("Bot Loadout Orange", parser.sections())

    def test_required_bundle_files_exist(self):
        required_root = [
            "bot.cfg",
            "appearance.cfg",
            "requirements.txt",
        ]
        for filename in required_root:
            self.assertTrue(Path(filename).exists(), f"File {filename} is required for bundling")

        bot_py_candidates = [Path("src/mia_bot/bot.py"), Path("bot.py")]
        self.assertTrue(any(p.exists() for p in bot_py_candidates), "bot.py must exist in src/mia_bot/ or root")


if __name__ == "__main__":
    unittest.main()
