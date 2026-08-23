import shutil
import zipfile
from pathlib import Path


def create_rlbot_bundle():
    root_dir = Path(__file__).parent.resolve()
    target_dir = root_dir / "dist" / "mia_bot"
    zip_path = root_dir / "dist" / "mia_bot.zip"

    files_to_copy = [
        "bot.py",
        "bot.cfg",
        "appearance.cfg",
        "requirements.txt",
        "policy.pt",
    ]

    missing = [f for f in files_to_copy if not (root_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files before packaging: {missing}")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_name in files_to_copy:
        shutil.copy2(root_dir / file_name, target_dir / file_name)
        print(f"Copied: {file_name}")

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in target_dir.rglob("*"):
            archive.write(file_path, file_path.relative_to(root_dir / "dist"))

    print(f"\n[+] Bundle successfully created in: {target_dir.resolve()}")


if __name__ == "__main__":
    create_rlbot_bundle()
