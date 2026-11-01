"""
Automated GitHub Actions -> Ephemeral Kaggle GPU Kernel Dispatcher (`scripts/trigger_kaggle_gpu_from_github.py`):
- If `KAGGLE_USERNAME` and `KAGGLE_KEY` are configured in GitHub Secrets (and `USE_KAGGLE_GPU` is enabled),
  this script pushes an ephemeral batch GPU kernel (`kaggle kernels push` with `enable_gpu: true`) to Kaggle.
- Kaggle boots a `Tesla T4 x2` GPU VM, runs ComfyUI (`LTX-Video 2B`) + `MAI-Image-2.6-Flash` + HeyGen HyperFrames,
  uploads the finished Short to YouTube, commits the updated topic history, and **Kaggle's batch hypervisor
  immediately shuts down and disconnects the GPU VM the exact millisecond the script finishes** (0 idle minutes used!).
- If `KAGGLE_USERNAME` / `KAGGLE_KEY` are not set in GitHub Secrets, returns exit code `2` so the workflow
  seamlessly runs `python src/main.py` directly on the GitHub Actions runner.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    kaggle_user = os.getenv("KAGGLE_USERNAME", "").strip()
    kaggle_key = os.getenv("KAGGLE_KEY", "").strip()
    use_kaggle = os.getenv("USE_KAGGLE_GPU", "auto").strip().lower()

    if use_kaggle == "false" or not (kaggle_user and kaggle_key):
        print("[KaggleDispatch] KAGGLE_USERNAME / KAGGLE_KEY not set (or USE_KAGGLE_GPU=false); running pipeline directly on current runner.")
        return 2

    kernel_slug = os.getenv("KAGGLE_KERNEL_SLUG", "auto-us-history-comfyui-shorts").strip()
    kernel_id = f"{kaggle_user}/{kernel_slug}"

    # Pass encrypted/masked runtime secrets into the ephemeral kernel script
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

    runner_script = f"""import os, subprocess, sys
from pathlib import Path

ENV_VARS = {json.dumps(env_dict)}
for k, v in ENV_VARS.items():
    os.environ[k] = str(v)

subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg fonts-montserrat fonts-liberation fonts-dejavu-core libmagic1 libnss3 libgbm1", shell=True, check=False)
subprocess.run("curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y -qq nodejs", shell=True, check=False)

repo_slug = os.environ.get("GITHUB_REPO", "Rageshukla69/us-history-shorts-auto")
gh_tok = os.environ.get("GITHUB_TOKEN", "")
repo_url = f"https://x-access-token:{{gh_tok}}@github.com/{{repo_slug}}.git" if gh_tok else f"https://github.com/{{repo_slug}}.git"

work_dir = Path("/kaggle/working/us-history-shorts-auto")
if work_dir.exists():
    subprocess.run(["rm", "-rf", str(work_dir)], check=False)
subprocess.run(["git", "clone", "--depth", "1", repo_url, str(work_dir)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(work_dir / "requirements.txt")], check=True)
subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=str(work_dir), check=False)
subprocess.run([sys.executable, str(work_dir / "kaggle/run_kaggle_comfyui_short.py")], cwd=str(work_dir), check=True)
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

        print(f"[KaggleDispatch] Pushing ephemeral GPU batch kernel '{kernel_id}' to Kaggle (auto-disconnects on completion)...")
        push_proc = subprocess.run(["kaggle", "kernels", "push", "-p", str(tmp_path)], capture_output=True, text=True)
        if push_proc.returncode != 0:
            print(f"[KaggleDispatch] Kaggle push notice ({push_proc.stderr or push_proc.stdout}); falling back to local runner.")
            return 2

        print(f"[KaggleDispatch] Kernel pushed! Polling status until completion...")
        deadline = time.time() + 2100  # 35 minutes max
        while time.time() < deadline:
            time.sleep(25)
            st_proc = subprocess.run(["kaggle", "kernels", "status", kernel_id], capture_output=True, text=True)
            out_low = (st_proc.stdout + " " + st_proc.stderr).lower()
            print(f"[KaggleDispatch] Status: {st_proc.stdout.strip()}")
            if "complete" in out_low:
                print("[KaggleDispatch] Kaggle GPU kernel completed and automatically disconnected!")
                return 0
            if "error" in out_low or "cancel" in out_low:
                print("[KaggleDispatch] Kaggle GPU kernel reported error; falling back to local runner.")
                return 2

    return 2


if __name__ == "__main__":
    sys.exit(main())
