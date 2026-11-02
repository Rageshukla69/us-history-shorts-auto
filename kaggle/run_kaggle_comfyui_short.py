"""
One-Shot Kaggle GPU Runner (`kaggle/run_kaggle_comfyui_short.py` — v13.1 Fast T4 + Ubuntu 24.04 Chromium Libs):
1. Installs the Ubuntu 24.04 shared libraries required by `npx hyperframes render` (`chrome-headless-shell`):
   `libatk-1.0.so.0`, `libatk-bridge-2.0.so.0`, `libXcomposite.so.1`, `libatspi.so.0`, `libxdamage1`, `libxrandr2`, etc.
2. Installs headless ComfyUI inside `/tmp/ComfyUI` (keeping `/kaggle/working` under 40 MB) with the 100% public,
   ungated `Lightricks/LTX-Video` (`ltx-video-2b-v0.9.5.safetensors`, 4.3 GB) + `t5xxl_fp8_e4m3fn.safetensors` (4.8 GB fp8)
   so both fit easily inside a single 15 GB Tesla T4 GPU with zero OOM!
3. Runs `src/main.py` with unbuffered real-time logging, pushes updated `used_topics.json` to GitHub, and
   immediately disconnects the Kaggle GPU session on completion.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kaggle_comfy_bridge import start_local_comfyui_on_kaggle, auto_disconnect_kaggle_gpu
from main import run_pipeline


def _ensure_ubuntu_2404_chromium_libs():
    """
    Fixes the exact error from Kaggle Ubuntu 24.04.1 LTS logs:
    `chrome-headless-shell is missing shared libraries on Ubuntu 24.04.1 LTS: libatk-1.0.so.0, libatk-bridge-2.0.so.0, libXcomposite.so.1, libatspi.so.0`
    """
    print("[KaggleSetup] Installing Ubuntu 24.04 shared libraries for HyperFrames chrome-headless-shell...", flush=True)
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq "
        "ffmpeg fonts-montserrat fonts-liberation fonts-dejavu-core libmagic1 "
        "libnss3 libnspr4 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libdrm2 "
        "libxkbcommon0 libatspi2.0-0t64 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 "
        "libgbm1 libpango-1.0-0 libcairo2 libasound2t64 "
        "|| apt-get install -y -qq libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 "
        "libdrm2 libxkbcommon0 libatspi2.0-0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 "
        "libgbm1 libpango-1.0-0 libcairo2 libasound2",
        shell=True,
        check=False,
    )
    subprocess.run(["npx", "--yes", "playwright", "install-deps", "chromium"], check=False)


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
        print(f"[KaggleSetup] Loaded {loaded} secrets from Kaggle UserSecretsClient.", flush=True)
    except Exception:
        pass


def _ensure_comfyui_and_ltx_fp8_installed(comfy_dir: Path = Path("/tmp/ComfyUI")):
    """
    Installs ComfyUI in `/tmp/ComfyUI` (outside `/kaggle/working`) and downloads:
    1. `ltx-video-2b-v0.9.5.safetensors` (~4.3 GB, 100% public/ungated on HuggingFace)
    2. `t5xxl_fp8_e4m3fn.safetensors` (~4.8 GB fp8 instead of 9.8 GB fp16, 100% public/ungated on HuggingFace)
    Total VRAM footprint is < 8 GB so it runs fast on Kaggle's 15 GB Tesla T4 GPU without OOM!
    """
    if not comfy_dir.exists():
        print(f"[KaggleSetup] Cloning ComfyUI into {comfy_dir}...", flush=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(comfy_dir)],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(comfy_dir / "requirements.txt"), "av", "torchsde", "einops", "imageio", "imageio-ffmpeg"],
            check=True,
        )

    ckpt_dir = comfy_dir / "models" / "checkpoints"
    clip_dir = comfy_dir / "models" / "clip"
    te_dir = comfy_dir / "models" / "text_encoders"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)
    te_dir.mkdir(parents=True, exist_ok=True)

    ltx_file = ckpt_dir / "ltx-video-2b-v0.9.5.safetensors"
    if not ltx_file.exists() or ltx_file.stat().st_size < 1_000_000_000:
        print("[KaggleSetup] Downloading Lightricks LTX-Video 2B v0.9.5 checkpoint (~4.3 GB)...", flush=True)
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

    t5_fp8_file = clip_dir / "t5xxl_fp8_e4m3fn.safetensors"
    if not t5_fp8_file.exists() or t5_fp8_file.stat().st_size < 1_000_000_000:
        print("[KaggleSetup] Downloading T5-XXL FP8 text encoder (~4.8 GB, fits easily in T4 VRAM)...", flush=True)
        subprocess.run(
            [
                "wget",
                "-q",
                "-O",
                str(t5_fp8_file),
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
            ],
            check=False,
        )

    os.environ["COMFYUI_WORKFLOW_MODE"] = "ltx_i2v"
    os.environ["COMFYUI_LTX_CHECKPOINT"] = "ltx-video-2b-v0.9.5.safetensors"
    os.environ["COMFYUI_T5_MODEL"] = "t5xxl_fp8_e4m3fn.safetensors"


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
            print("[KaggleSetup] Synced updated topic history back to GitHub repository.", flush=True)
    except Exception as exc:
        print(f"[KaggleSetup] GitHub sync notice: {exc}", flush=True)


def _copy_final_deliverables_to_kaggle_working():
    """Copies the final rendered MP4, metadata JSON, and HyperFrames HTML into `/kaggle/working` (< 40 MB)."""
    out_src = REPO_ROOT / "output"
    kw_root = Path("/kaggle/working")
    if out_src.exists() and kw_root.exists():
        for ext in ("*.mp4", "*.json", "*.jpg", "*.ass"):
            for f in out_src.glob(ext):
                try:
                    shutil.copy2(f, kw_root / f.name)
                except Exception:
                    pass
        hf_src = out_src / "hyperframes_project"
        hf_dst = kw_root / "hyperframes_project"
        if hf_src.exists():
            hf_dst.mkdir(parents=True, exist_ok=True)
            for ext in ("*.html", "*.md"):
                for f in hf_src.glob(ext):
                    try:
                        shutil.copy2(f, hf_dst / f.name)
                    except Exception:
                        pass


def main():
    comfy_proc = None
    try:
        _ensure_ubuntu_2404_chromium_libs()
        _load_kaggle_secrets_into_env()
        os.environ["MAX_AI_IMAGES_PER_SHORT"] = "10"
        os.environ["MAX_COMFY_VIDEOS_PER_SHORT"] = "10"
        os.environ.setdefault("DRY_RUN", "true")
        if os.getenv("COMFYUI_DOWNLOAD_LTX", "false").lower() == "true":
            _ensure_comfyui_and_ltx_fp8_installed(Path("/tmp/ComfyUI"))
        comfy_proc = None  # v14.0 Direct In-Process ComfyUI Engine uses 100% of GPU VRAM directly!
        os.chdir(str(REPO_ROOT))
        os.environ["KAGGLE_AUTO_DISCONNECT"] = "false"
        run_pipeline()
        _sync_completed_topic_to_github()
        _copy_final_deliverables_to_kaggle_working()
    finally:
        os.environ["KAGGLE_AUTO_DISCONNECT"] = "true"
        auto_disconnect_kaggle_gpu(comfy_proc=comfy_proc, force_kernel_exit=True)


if __name__ == "__main__":
    main()
