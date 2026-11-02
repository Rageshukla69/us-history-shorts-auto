"""
One-Shot Kaggle GPU Runner (`kaggle/run_kaggle_comfyui_short.py` — v13.0 Quota-Saver Edition):
1. Installs & configures headless ComfyUI inside `/tmp/ComfyUI` (using Kaggle's fast temporary disk so
   `/kaggle/working` never overflows Kaggle's 20 GB output artifact cap!).
2. Downloads `svd_xt.safetensors` (fp16, ~4.0 GB) and `ltx-video-2b-v0.9.5.safetensors` + `t5xxl_fp8_e4m3fn.safetensors`
   into `/tmp/ComfyUI/models`, plus Node.js 22 + HyperFrames CLI + Playwright Chromium.
3. Starts ComfyUI in the background (`http://127.0.0.1:8188` with `--lowvram --fp16-vae`).
4. Runs `src/main.py` to generate 1 complete Short (10 `MAI-Image-2.6-Flash` keyframes -> animated into realistic moving
   9:16 cinema clips by ComfyUI -> composed in HeyGen HyperFrames v13.0 -> uploaded to YouTube).
5. Pushes the updated `data/used_topics.json` and `data/upload_history.json` back to your GitHub repository (if `GITHUB_TOKEN` is set).
6. Immediately terminates ComfyUI, clears CUDA VRAM, and disconnects the Kaggle GPU VM (`auto_disconnect_kaggle_gpu`)
   so you NEVER waste a single idle minute of your 30 hrs/week Kaggle GPU quota!
"""
from __future__ import annotations
import os
import shutil
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


def _ensure_comfyui_installed(comfy_dir: Path = Path("/tmp/ComfyUI")):
    """Installs ComfyUI in `/tmp/ComfyUI` (outside `/kaggle/working`) and downloads SVD-XT fp16 (~4.0 GB) or LTX-Video 2B."""
    if not comfy_dir.exists():
        print(f"[KaggleSetup] Cloning ComfyUI into {comfy_dir}...")
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

    # Download lightweight SVD-XT fp16 (4.0 GB, single file, no external 10GB CLIP required, runs fast on Kaggle T4)
    svd_file = ckpt_dir / "svd_xt.safetensors"
    if not svd_file.exists() or svd_file.stat().st_size < 1_000_000_000:
        print("[KaggleSetup] Downloading StabilityAI SVD-XT fp16 Image-to-Video checkpoint (~4.0 GB)...")
        subprocess.run(
            [
                "wget",
                "-q",
                "-O",
                str(svd_file),
                "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors",
            ],
            check=False,
        )
    os.environ.setdefault("COMFYUI_WORKFLOW_MODE", "svd_xt")
    os.environ.setdefault("COMFYUI_SVD_CHECKPOINT", "svd_xt.safetensors")


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


def _copy_final_deliverables_to_kaggle_working():
    """Copies only the final rendered MP4 and metadata JSON into `/kaggle/working/output` (< 40 MB)."""
    out_src = REPO_ROOT / "output"
    out_dst = Path("/kaggle/working/output")
    if out_src.exists() and Path("/kaggle/working").exists():
        out_dst.mkdir(parents=True, exist_ok=True)
        for ext in ("*.mp4", "*.json", "*.jpg"):
            for f in out_src.glob(ext):
                try:
                    shutil.copy2(f, out_dst / f.name)
                except Exception:
                    pass


def main():
    comfy_proc = None
    try:
        _load_kaggle_secrets_into_env()
        os.environ["MAX_AI_IMAGES_PER_SHORT"] = "10"
        os.environ["MAX_COMFY_VIDEOS_PER_SHORT"] = "10"
        _ensure_comfyui_installed(Path("/tmp/ComfyUI"))
        comfy_proc = start_local_comfyui_on_kaggle(Path("/tmp/ComfyUI"))
        os.chdir(str(REPO_ROOT))
        # temporarily disable inner exit so we can sync git before final disconnect
        os.environ["KAGGLE_AUTO_DISCONNECT"] = "false"
        run_pipeline()
        _sync_completed_topic_to_github()
        _copy_final_deliverables_to_kaggle_working()
    finally:
        os.environ["KAGGLE_AUTO_DISCONNECT"] = "true"
        auto_disconnect_kaggle_gpu(comfy_proc=comfy_proc, force_kernel_exit=True)


if __name__ == "__main__":
    main()
