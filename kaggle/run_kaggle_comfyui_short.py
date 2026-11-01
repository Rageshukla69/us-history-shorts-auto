"""
One-Shot Kaggle GPU Runner (`kaggle/run_kaggle_comfyui_short.py` — v12.0 Quota-Saver Edition):
1. Installs & configures headless ComfyUI + `LTX-Video 2B v0.9.5` (`Lightricks/LTX-Video`) + Node.js 22 + HyperFrames CLI
   inside a Kaggle GPU session (`Tesla T4 x2` / `P100`).
2. Starts ComfyUI in the background (`http://127.0.0.1:8188`).
3. Runs `src/main.py` to generate 1 complete Short (10 `MAI-Image-2.6-Flash` keyframes -> animated into realistic moving
   9:16 cinema clips by ComfyUI `LTXVImgToVideo` -> composed in HeyGen HyperFrames v12.0 -> uploaded to YouTube).
4. Pushes the updated `data/used_topics.json` and `data/upload_history.json` back to your GitHub repository (if `GITHUB_TOKEN` is set).
5. Immediately terminates ComfyUI, clears CUDA VRAM, and disconnects the Kaggle GPU VM (`auto_disconnect_kaggle_gpu`)
   so you NEVER waste a single idle minute of your 30 hrs/week Kaggle GPU quota!
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kaggle_comfy_bridge import start_local_comfyui_on_kaggle, auto_disconnect_kaggle_gpu
from main import run_pipeline


def _load_kaggle_secrets_into_env():
    """Automatically loads secrets from Kaggle's `UserSecretsClient` (Add-ons -> Secrets) if running in a Kaggle Notebook."""
    secret_names = [
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
        "GITHUB_TOKEN",
        "GITHUB_REPO",
    ]
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        user_secrets = UserSecretsClient()
        loaded = 0
        for name in secret_names:
            if not os.getenv(name):
                try:
                    val = user_secrets.get_secret(name)
                    if val:
                        os.environ[name] = str(val).strip()
                        loaded += 1
                except Exception:
                    pass
        print(f"[KaggleSetup] Loaded {loaded} secrets from Kaggle UserSecretsClient.")
    except Exception:
        pass


def _ensure_comfyui_and_ltx_installed(comfy_dir: Path = Path("/kaggle/working/ComfyUI")):
    """Installs ComfyUI and downloads `ltx-video-2b-v0.9.5.safetensors` + `t5xxl_fp16.safetensors` if not cached."""
    if not comfy_dir.exists():
        print("[KaggleSetup] Cloning ComfyUI into /kaggle/working/ComfyUI...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(comfy_dir)],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(comfy_dir / "requirements.txt")],
            check=True,
        )

    ckpt_dir = comfy_dir / "models" / "checkpoints"
    clip_dir = comfy_dir / "models" / "clip"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    ltx_file = ckpt_dir / "ltx-video-2b-v0.9.5.safetensors"
    if not ltx_file.exists():
        print("[KaggleSetup] Downloading Lightricks LTX-Video 2B v0.9.5 checkpoint (~4.3 GB)...")
        subprocess.run(
            [
                "wget",
                "-q",
                "-O",
                str(ltx_file),
                "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors",
            ],
            check=False,
        )

    t5_file = clip_dir / "t5xxl_fp16.safetensors"
    if not t5_file.exists():
        print("[KaggleSetup] Downloading T5-XXL fp16 text encoder for LTX-Video...")
        subprocess.run(
            [
                "wget",
                "-q",
                "-O",
                str(t5_file),
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
            ],
            check=False,
        )


def _sync_completed_topic_to_github():
    """If GITHUB_TOKEN is provided, commits and pushes updated `used_topics.json` & `upload_history.json` back to GitHub."""
    gh_token = os.getenv("GITHUB_TOKEN", "").strip()
    gh_repo = os.getenv("GITHUB_REPO", "Rageshukla69/us-history-shorts-auto").strip()
    if not gh_token:
        return
    try:
        subprocess.run(["git", "config", "user.email", "kaggle-gpu-bot@users.noreply.github.com"], cwd=str(REPO_ROOT), check=False)
        subprocess.run(["git", "config", "user.name", "Kaggle ComfyUI GPU Bot"], cwd=str(REPO_ROOT), check=False)
        subprocess.run(["git", "add", "data/used_topics.json", "data/upload_history.json", "data/topics_seed.json"], cwd=str(REPO_ROOT), check=False)
        diff = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=str(REPO_ROOT), check=False)
        if diff.returncode != 0:
            subprocess.run(["git", "commit", "-m", "chore(kaggle-gpu): record completed ComfyUI history short [skip ci]"], cwd=str(REPO_ROOT), check=False)
            remote_url = f"https://x-access-token:{gh_token}@github.com/{gh_repo}.git"
            subprocess.run(["git", "pull", "--rebase", "--autostash", remote_url, "main"], cwd=str(REPO_ROOT), check=False)
            subprocess.run(["git", "push", remote_url, "HEAD:main"], cwd=str(REPO_ROOT), check=False)
            print("[KaggleSetup] Synced updated topic history back to GitHub repository.")
    except Exception as exc:
        print(f"[KaggleSetup] GitHub sync notice: {exc}")


def main():
    comfy_proc = None
    try:
        _load_kaggle_secrets_into_env()
        _ensure_comfyui_and_ltx_installed()
        comfy_proc = start_local_comfyui_on_kaggle()
        os.chdir(str(REPO_ROOT))
        # temporarily disable inner exit so we can sync git before final disconnect
        os.environ["KAGGLE_AUTO_DISCONNECT"] = "false"
        run_pipeline()
        _sync_completed_topic_to_github()
    finally:
        os.environ["KAGGLE_AUTO_DISCONNECT"] = "true"
        auto_disconnect_kaggle_gpu(comfy_proc=comfy_proc, force_kernel_exit=True)


if __name__ == "__main__":
    main()
