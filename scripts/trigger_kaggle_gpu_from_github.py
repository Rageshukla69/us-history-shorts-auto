"""
Automated GitHub Actions -> Ephemeral Kaggle GPU Kernel Dispatcher (`scripts/trigger_kaggle_gpu_from_github.py` — v13.0):
- If `KAGGLE_USERNAME` and `KAGGLE_KEY` are configured in GitHub Secrets (and `USE_KAGGLE_GPU` is enabled),
  this script pushes an ephemeral batch GPU kernel (`kaggle kernels push` with `enable_gpu: true`) to Kaggle.
- Supports BOTH Kaggle's new `KGAT_...` API tokens (`KAGGLE_API_TOKEN` / `~/.kaggle/access_token`) AND legacy
  32-char API keys (`~/.kaggle/kaggle.json`) with automatic fallback between both modes.
- Runs inside `/tmp` on the Kaggle GPU VM so `/kaggle/working` stays under 40 MB and disconnects immediately
  on completion (0 idle minutes consumed from your 30 hrs/week quota).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _configure_kaggle_auth(kaggle_user: str, kaggle_key: str, mode: str) -> None:
    kaggle_cfg_dir = Path.home() / ".kaggle"
    kaggle_cfg_dir.mkdir(parents=True, exist_ok=True)
    access_tok_file = kaggle_cfg_dir / "access_token"
    kaggle_json_file = kaggle_cfg_dir / "kaggle.json"

    if mode == "token":
        os.environ["KAGGLE_API_TOKEN"] = kaggle_key
        os.environ["KAGGLE_USERNAME"] = kaggle_user
        os.environ.pop("KAGGLE_KEY", None)
        access_tok_file.write_text(kaggle_key, encoding="utf-8")
        access_tok_file.chmod(0o600)
        kaggle_json_file.unlink(missing_ok=True)
    else:
        os.environ.pop("KAGGLE_API_TOKEN", None)
        os.environ["KAGGLE_USERNAME"] = kaggle_user
        os.environ["KAGGLE_KEY"] = kaggle_key
        access_tok_file.unlink(missing_ok=True)
        kaggle_json_file.write_text(
            json.dumps({"username": kaggle_user, "key": kaggle_key}),
            encoding="utf-8",
        )
        kaggle_json_file.chmod(0o600)


def main() -> int:
    raw_user = os.getenv("KAGGLE_USERNAME", "").strip()
    kaggle_key = (os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_TOKEN") or "").strip()
    use_kaggle = os.getenv("USE_KAGGLE_GPU", "auto").strip().lower()

    if use_kaggle == "false" or not (raw_user and kaggle_key):
        print("[KaggleDispatch] KAGGLE_USERNAME / KAGGLE_KEY not set (or USE_KAGGLE_GPU=false); running pipeline directly on current runner.")
        return 2

    # Kaggle kernel IDs require lowercase username slug
    kaggle_user_slug = raw_user.lower()
    kernel_slug = os.getenv("KAGGLE_KERNEL_SLUG", "auto-us-history-comfyui-shorts").strip().lower()
    kernel_id = f"{kaggle_user_slug}/{kernel_slug}"

    try:
        import six  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "six"], check=False)

    passthrough_keys = [
        "AZURE_SPEECH_KEY",
        "AZURE_SPEECH_REGION",
        "AZURE_TTS_VOICE",
        "BG_MUSIC_VOLUME",
        "FOUNDRY_API_KEY",
        "FOUNDRY_ENDPOINT",
        "FOUNDRY_IMAGE_MODEL",
        "FOUNDRY_TEXT_API_KEY",
        "FOUNDRY_TEXT_ENDPOINT",
        "FOUNDRY_TEXT_MODEL",
        "YOUTUBE_TOKEN_JSON",
        "PEXELS_API_KEY",
        "MAX_AI_IMAGES_PER_SHORT",
        "MAX_COMFY_VIDEOS_PER_SHORT",
        "CUSTOM_TOPIC",
        "DRY_RUN",
        "GITHUB_TOKEN",
        "GITHUB_REPO",
    ]
    env_dict = {k: os.getenv(k, "") for k in passthrough_keys if os.getenv(k, "")}
    env_dict["MAX_AI_IMAGES_PER_SHORT"] = "10"
    env_dict["MAX_COMFY_VIDEOS_PER_SHORT"] = "4"
    env_dict["DRY_RUN"] = os.getenv("DRY_RUN", "true") or "true"

    runner_script = f"""import os, subprocess, sys
from pathlib import Path

ENV_VARS = {json.dumps(env_dict)}
for k, v in ENV_VARS.items():
    os.environ[k] = str(v)

os.environ["PYTHONUNBUFFERED"] = "1"
subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg fonts-montserrat fonts-liberation fonts-dejavu-core libmagic1 libnss3 libnspr4 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libdrm2 libxkbcommon0 libatspi2.0-0t64 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2t64 || apt-get install -y -qq ffmpeg fonts-montserrat fonts-liberation fonts-dejavu-core libmagic1 libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libatspi2.0-0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2", shell=True, check=False)
subprocess.run("curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y -qq nodejs", shell=True, check=False)

repo_slug = os.environ.get("GITHUB_REPO", "Rageshukla69/us-history-shorts-auto")
gh_tok = os.environ.get("GITHUB_TOKEN", "")
repo_url = f"https://x-access-token:{{gh_tok}}@github.com/{{repo_slug}}.git" if gh_tok else f"https://github.com/{{repo_slug}}.git"

work_dir = Path("/tmp/us-history-shorts-auto")
if work_dir.exists():
    subprocess.run(["rm", "-rf", str(work_dir)], check=False)
subprocess.run(["git", "clone", "--depth", "1", repo_url, str(work_dir)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(work_dir / "requirements.txt"), "playwright"], check=True)
subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)
subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=str(work_dir), check=False)
subprocess.run([sys.executable, "-u", str(work_dir / "kaggle/run_kaggle_comfyui_short.py")], cwd=str(work_dir), check=True)
"""

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "kernel.py").write_text(runner_script, encoding="utf-8")
        meta = {
            "id": kernel_id,
            "title": "Auto US History ComfyUI Shorts",
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        auth_modes = ["token", "legacy"] if kaggle_key.startswith("KGAT_") else ["legacy", "token"]
        pushed = False
        for mode in auth_modes:
            _configure_kaggle_auth(raw_user, kaggle_key, mode)
            print(f"[KaggleDispatch] Pushing ephemeral GPU batch kernel '{kernel_id}' to Kaggle (auth={mode})...")
            push_proc = subprocess.run(["kaggle", "kernels", "push", "-p", str(tmp_path)], capture_output=True, text=True)
            if push_proc.returncode == 0:
                print(f"[KaggleDispatch] Kernel pushed successfully: {push_proc.stdout.strip()}")
                pushed = True
                break
            print(f"[KaggleDispatch] Auth mode '{mode}' returned code {push_proc.returncode}: {push_proc.stdout.strip()} {push_proc.stderr.strip()}")

        if not pushed:
            print("[KaggleDispatch] Could not push kernel to Kaggle; falling back to local runner.")
            return 2

        print("[KaggleDispatch] Polling Kaggle GPU kernel status until completion...")
        deadline = time.time() + 2200  # 36 minutes max
        while time.time() < deadline:
            time.sleep(25)
            st_proc = subprocess.run(["kaggle", "kernels", "status", kernel_id], capture_output=True, text=True)
            out_low = (st_proc.stdout + " " + st_proc.stderr).lower()
            print(f"[KaggleDispatch] Status: {st_proc.stdout.strip()}")
            if "complete" in out_low:
                print("[KaggleDispatch] Kaggle GPU kernel completed and automatically disconnected! Downloading output artifacts...")
                out_dir = Path("output")
                out_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", str(out_dir)], check=False)
                # Flatten output/output/* into output/* so GitHub Actions finds output/final_short.mp4 directly
                nested_out = out_dir / "output"
                if nested_out.exists() and nested_out.is_dir():
                    for item in nested_out.iterdir():
                        dest = out_dir / item.name
                        if dest.exists():
                            if dest.is_dir():
                                shutil.rmtree(dest, ignore_errors=True)
                            else:
                                dest.unlink(missing_ok=True)
                        shutil.move(str(item), str(dest))
                return 0
            if "error" in out_low or "cancel" in out_low:
                print("[KaggleDispatch] Kaggle GPU kernel reported error; fetching log and falling back to local runner.")
                subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", "/tmp/kaggle_err_out"], check=False)
                return 2

    return 2


if __name__ == "__main__":
    sys.exit(main())
