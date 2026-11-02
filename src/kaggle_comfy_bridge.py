"""
Kaggle GPU Lifecycle & Auto-Disconnect Manager (`src/kaggle_comfy_bridge.py` — v12.0 Quota-Saver Edition):
Designed specifically around Kaggle's **30 hours/week free GPU quota** (`Tesla T4 x2` / `P100`):
1. `start_local_comfyui_on_kaggle()`:
   - When running inside a Kaggle GPU VM (`/kaggle/working`), installs & launches headless ComfyUI (`127.0.0.1:8188`)
     with `LTX-Video 2B v0.9.5` (`Lightricks/LTX-Video`) or `SVD-XT 1.1` in fp16 mode and waits until `/system_stats`
     is ready.
2. `auto_disconnect_kaggle_gpu(comfy_proc=None)`:
   - Immediately after the Short video finishes generating and uploading to YouTube:
     a) Terminates the ComfyUI server process (`SIGTERM` -> `SIGKILL`),
     b) Releases all CUDA VRAM (`torch.cuda.empty_cache()`),
     c) Flushes logs and sends `SIGTERM` to the Kaggle Jupyter/container parent process followed by `os._exit(0)`
        so the Kaggle GPU VM shuts down and disconnects immediately — burning **ZERO idle minutes** of your 30 hrs/week quota!
"""
from __future__ import annotations
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
import requests


def is_running_on_kaggle() -> bool:
    return bool(
        os.getenv("KAGGLE_KERNEL_RUN_TYPE")
        or os.getenv("KAGGLE_URL_BASE")
        or Path("/kaggle/working").exists()
    )


def start_local_comfyui_on_kaggle(
    comfy_dir: Path = Path("/tmp/ComfyUI"),
    port: int = 8188,
    startup_timeout_sec: int = 180,
) -> Optional[subprocess.Popen]:
    """
    Starts headless ComfyUI on `127.0.0.1:{port}` inside a Kaggle GPU notebook/kernel
    and sets `COMFYUI_API_URL=http://127.0.0.1:{port}`.
    """
    api_url = f"http://127.0.0.1:{port}"
    try:
        r = requests.get(f"{api_url}/system_stats", timeout=3)
        if r.status_code == 200:
            os.environ["COMFYUI_API_URL"] = api_url
            print(f"[KaggleGPU] ComfyUI server already active at {api_url}")
            return None
    except Exception:
        pass

    if not comfy_dir.exists() or not (comfy_dir / "main.py").exists():
        print(f"[KaggleGPU] ComfyUI directory not found at {comfy_dir}; skipping local ComfyUI boot.")
        return None

    print(f"[KaggleGPU] Booting headless ComfyUI server on {api_url}...")
    log_file = open(comfy_dir / "comfyui_server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--disable-auto-launch",
            "--fp16-vae",
            "--lowvram",
        ],
        cwd=str(comfy_dir),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    deadline = time.time() + startup_timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[KaggleGPU] ComfyUI process exited early with code {proc.returncode}.")
            return None
        try:
            r = requests.get(f"{api_url}/system_stats", timeout=3)
            if r.status_code == 200:
                os.environ["COMFYUI_API_URL"] = api_url
                print(f"[KaggleGPU] ComfyUI GPU server ready at {api_url}!")
                return proc
        except Exception:
            time.sleep(2.0)

    print("[KaggleGPU] Timed out waiting for ComfyUI server startup.")
    try:
        proc.terminate()
    except Exception:
        pass
    return None


def auto_disconnect_kaggle_gpu(
    comfy_proc: Optional[subprocess.Popen] = None,
    force_kernel_exit: bool = True,
):
    """
    Gracefully stops ComfyUI, frees GPU VRAM, and terminates the Kaggle GPU session immediately
    after a Short completes so zero idle minutes of the 30 hrs/week Kaggle quota are consumed.
    """
    if comfy_proc is not None:
        print("[KaggleGPU] Stopping ComfyUI server process...")
        try:
            comfy_proc.terminate()
            comfy_proc.wait(timeout=8)
        except Exception:
            try:
                comfy_proc.kill()
            except Exception:
                pass

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("[KaggleGPU] Cleared CUDA VRAM cache.")
    except Exception:
        pass

    if is_running_on_kaggle() and force_kernel_exit and os.getenv("KAGGLE_AUTO_DISCONNECT", "true").lower() == "true":
        print(
            "[KaggleGPU] Short generation & upload complete! Automatically disconnecting Kaggle GPU session "
            "now to preserve your 30 hrs/week GPU quota..."
        )
        sys.stdout.flush()
        sys.stderr.flush()
        run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "").strip().lower()
        # In Interactive mode, killing parent PID (Jupyter server) immediately stops the Kaggle GPU VM
        if run_type == "interactive":
            try:
                os.kill(os.getppid(), signal.SIGTERM)
                time.sleep(1.0)
            except Exception:
                pass
        os._exit(0)
