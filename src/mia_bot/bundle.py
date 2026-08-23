import os
import shutil
import zipfile
from pathlib import Path


def find_repo_root() -> Path:
    if "BUILD_WORKSPACE_DIRECTORY" in os.environ:
        return Path(os.environ["BUILD_WORKSPACE_DIRECTORY"])
    p = Path(__file__).resolve()
    for parent in [p.parent, p.parent.parent, p.parent.parent.parent, Path.cwd()]:
        if (parent / "bot.cfg").exists() or (parent / "MODULE.bazel").exists():
            return parent
    return Path.cwd()


def create_rlbot_bundle():
    root_dir = find_repo_root()
    target_dir = root_dir / "dist" / "mia_bot"
    zip_path = root_dir / "dist" / "mia_bot.zip"

    bot_src = root_dir / "src" / "mia_bot" / "bot.py"
    if not bot_src.exists():
        bot_src = root_dir / "bot.py"

    root_files = [
        "bot.cfg",
        "appearance.cfg",
        "requirements.txt",
    ]

    missing = [f for f in root_files if not (root_dir / f).exists()]
    if not bot_src.exists():
        missing.append(str(bot_src))

    if missing:
        raise FileNotFoundError(f"Missing required files before packaging: {missing}")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy bot.py into bundle
    shutil.copy2(bot_src, target_dir / "bot.py")
    print(f"Copied: bot.py (from {bot_src})")

    # Copy config files
    for file_name in root_files:
        shutil.copy2(root_dir / file_name, target_dir / file_name)
        print(f"Copied: {file_name}")

    # Copy policy.pt if present
    policy_path = root_dir / "policy.pt"
    if policy_path.exists():
        shutil.copy2(policy_path, target_dir / "policy.pt")
        print("Copied: policy.pt")
    else:
        print("Note: policy.pt not found in root, skipping.")

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in target_dir.rglob("*"):
            archive.write(file_path, file_path.relative_to(root_dir / "dist"))

    print(f"\n[+] Bundle successfully created in: {target_dir.resolve()}")


if __name__ == "__main__":
    if "BUILD_WORKSPACE_DIRECTORY" in os.environ:
        os.chdir(os.environ["BUILD_WORKSPACE_DIRECTORY"])
    create_rlbot_bundle()
