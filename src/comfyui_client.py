"""
ComfyUI Realistic AI Video Generation Client (`src/comfyui_client.py` — v12.0 Kaggle GPU + ComfyUI I2V Edition):
Turns ultra-sharp 9:16 vertical `MAI-Image-2.6-Flash` historical keyframes into realistic moving cinema video clips (`.mp4`)
using ComfyUI's REST API (`/upload/image`, `/prompt`, `/history`, `/view`).

Supported ComfyUI Workflows:
1. `ltx_i2v` (Default on Kaggle T4x2 / P100 GPU):
   - Uses mainline ComfyUI's native `LTX-Video` (`Lightricks/LTX-Video` `ltx-video-2b-v0.9.5.safetensors`)
     Image-to-Video node graph (`LoadImage` -> `LTXVImgToVideo` -> `LTXVConditioning` -> `LTXVScheduler` -> `SamplerCustom` -> `VAEDecode` -> `SaveAnimatedWEBP`)
     to animate the exact `MAI-Image-2.6-Flash` historical frame with realistic physical motion (flickering firelight,
     rolling ocean waves, marching soldiers, drifting smoke, turning gears, and natural camera dolly movement).
2. `svd_xt` (Stable Video Diffusion XT 1.1 `svd_xt_1_1.safetensors`):
   - Uses ComfyUI's built-in `ImageOnlyCheckpointLoader` -> `SVD_img2vid_Conditioning` -> `KSampler` -> `VAEDecode` -> `SaveAnimatedWEBP`.
3. Automatic High-Realism Cinema Motion Fallback:
   - If `COMFYUI_API_URL` is unreachable (e.g. when running on CPU-only GitHub Actions without an active Kaggle tunnel),
     cleanly synthesizes a 9:16 30fps `.mp4` motion clip with 3x subpixel dolly/parallax and temporal 35mm grain so a run never fails.
"""
from __future__ import annotations
import json
import os
import random
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests


class ComfyUIVideoClient:
    WIDTH = 1080
    HEIGHT = 1920
    FPS = 30

    def __init__(self, api_url: Optional[str] = None):
        raw_url = (
            api_url
            or os.getenv("COMFYUI_API_URL")
            or os.getenv("KAGGLE_COMFYUI_URL")
            or ""
        ).strip().rstrip("/")
        self.api_url = raw_url
        self.workflow_mode = (os.getenv("COMFYUI_WORKFLOW_MODE") or "ltx_i2v").strip().lower()
        self.ltx_ckpt = (os.getenv("COMFYUI_LTX_CHECKPOINT") or "ltx-video-2b-v0.9.5.safetensors").strip()
        self.t5_clip = (os.getenv("COMFYUI_T5_MODEL") or "t5xxl_fp16.safetensors").strip()
        self.svd_ckpt = (os.getenv("COMFYUI_SVD_CHECKPOINT") or "svd_xt_1_1.safetensors").strip()
        self.client_id = str(uuid.uuid4())
        self.session = requests.Session()
        self.clips_rendered = 0
        try:
            self.max_clips = max(0, int(os.getenv("MAX_COMFY_VIDEOS_PER_SHORT", "10")))
        except ValueError:
            self.max_clips = 10

    def is_server_reachable(self) -> bool:
        if not self.api_url:
            return False
        try:
            resp = self.session.get(f"{self.api_url}/system_stats", timeout=6)
            return resp.status_code == 200
        except Exception:
            return False

    def _detect_available_workflow(self) -> str:
        """Auto-detects whether the connected ComfyUI server has LTX-Video or SVD-XT checkpoints installed."""
        if not self.api_url:
            return self.workflow_mode
        try:
            resp = self.session.get(f"{self.api_url}/object_info/CheckpointLoaderSimple", timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                ckpts = (
                    data.get("CheckpointLoaderSimple", {})
                    .get("input", {})
                    .get("required", {})
                    .get("ckpt_name", [[]])[0]
                )
                # Also check available CLIP encoders if LTX is present
                clip_resp = self.session.get(f"{self.api_url}/object_info/CLIPLoader", timeout=6)
                if clip_resp.status_code == 200:
                    clips = (
                        clip_resp.json()
                        .get("CLIPLoader", {})
                        .get("input", {})
                        .get("required", {})
                        .get("clip_name", [[]])[0]
                    )
                    for cl in clips:
                        if "t5" in str(cl).lower():
                            self.t5_clip = str(cl)
                            break
                for c in ckpts:
                    if "svd" in str(c).lower():
                        self.svd_ckpt = str(c)
                        return "svd_xt"
                for c in ckpts:
                    if "ltx" in str(c).lower():
                        self.ltx_ckpt = str(c)
                        return "ltx_i2v"
        except Exception:
            pass
        return self.workflow_mode

    def _upload_keyframe(self, image_path: Path) -> Optional[str]:
        """Uploads the 9:16 keyframe image to ComfyUI's `/upload/image` endpoint."""
        try:
            with open(image_path, "rb") as fp:
                files = {"image": (image_path.name, fp, "image/jpeg")}
                data = {"overwrite": "true", "type": "input"}
                resp = self.session.post(f"{self.api_url}/upload/image", files=files, data=data, timeout=30)
                if resp.status_code == 200:
                    return resp.json().get("name", image_path.name)
        except Exception as exc:
            print(f"[ComfyUI] Keyframe upload notice: {exc}")
        return None

    def _build_ltx_i2v_prompt(self, uploaded_image_name: str, motion_prompt: str, num_frames: int = 73) -> Dict[str, Any]:
        """
        Builds a native ComfyUI `LTX-Video` Image-to-Video workflow graph (`576x1024` 9:16 vertical -> upscaled to `1080x1920`).
        `num_frames` in LTX-Video must be `8 * k + 1` (e.g. `73` frames = ~3.0s at 24fps, or `97` frames = ~4.0s).
        """
        seed = random.randint(10_000, 9_999_999_999)
        pos_prompt = (
            f"{motion_prompt}, photorealistic historical documentary cinematography, natural realistic physical motion, "
            "subtle cinema dolly camera movement, 35mm film grain, lifelike textures, high detail, 4k"
        )
        neg_prompt = (
            "morphing, distorted face, extra limbs, cartoon, anime, watermark, text, blurry, low quality, flicker, static"
        )
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": self.ltx_ckpt},
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": self.t5_clip, "type": "ltxv"},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": pos_prompt, "clip": ["2", 0]},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": neg_prompt, "clip": ["2", 0]},
            },
            "5": {
                "class_type": "LoadImage",
                "inputs": {"image": uploaded_image_name},
            },
            "6": {
                "class_type": "LTXVPreprocess",
                "inputs": {"image": ["5", 0], "img_compression": 28},
            },
            "7": {
                "class_type": "LTXVImgToVideo",
                "inputs": {
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "vae": ["1", 2],
                    "image": ["6", 0],
                    "width": 576,
                    "height": 1024,
                    "length": num_frames,
                    "batch_size": 1,
                    "strength": 0.88,
                },
            },
            "8": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": ["7", 0],
                    "negative": ["7", 1],
                    "frame_rate": 24.0,
                },
            },
            "9": {
                "class_type": "LTXVScheduler",
                "inputs": {
                    "steps": 24,
                    "max_shift": 2.05,
                    "base_shift": 0.95,
                    "stretch": True,
                    "terminal": 0.1,
                    "latent": ["7", 2],
                },
            },
            "10": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler"},
            },
            "11": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["8", 0],
                    "negative": ["8", 1],
                    "cfg": 3.2,
                },
            },
            "12": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": seed},
            },
            "13": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["12", 0],
                    "guider": ["11", 0],
                    "sampler": ["10", 0],
                    "sigmas": ["9", 0],
                    "latent_image": ["7", 2],
                },
            },
            "14": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["13", 0], "vae": ["1", 2]},
            },
            "15": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "images": ["14", 0],
                    "filename_prefix": "comfy_ltx_short",
                    "fps": 24.0,
                    "lossless": False,
                    "quality": 92,
                    "method": "default",
                },
            },
        }

    def _build_svd_xt_prompt(self, uploaded_image_name: str, num_frames: int = 25) -> Dict[str, Any]:
        """Builds a core ComfyUI `SVD-XT 1.1` Image-to-Video workflow (`576x1024` vertical)."""
        seed = random.randint(10_000, 9_999_999_999)
        return {
            "1": {
                "class_type": "ImageOnlyCheckpointLoader",
                "inputs": {"ckpt_name": self.svd_ckpt},
            },
            "2": {
                "class_type": "LoadImage",
                "inputs": {"image": uploaded_image_name},
            },
            "3": {
                "class_type": "SVD_img2vid_Conditioning",
                "inputs": {
                    "clip_vision": ["1", 1],
                    "init_image": ["2", 0],
                    "vae": ["1", 2],
                    "width": 576,
                    "height": 1024,
                    "video_frames": num_frames,
                    "motion_bucket_id": 115,
                    "fps": 8,
                    "augmentation_level": 0.02,
                },
            },
            "4": {
                "class_type": "VideoLinearCFGGuidance",
                "inputs": {"model": ["1", 0], "min_cfg": 1.0},
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["4", 0],
                    "seed": seed,
                    "steps": 20,
                    "cfg": 2.5,
                    "sampler_name": "euler",
                    "scheduler": "karras",
                    "positive": ["3", 0],
                    "negative": ["3", 1],
                    "latent_image": ["3", 2],
                    "denoise": 1.0,
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            },
            "7": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": "comfy_svd_short",
                    "fps": 12.0,
                    "lossless": False,
                    "quality": 92,
                    "method": "default",
                },
            },
        }

    def animate_keyframe_to_video(
        self,
        keyframe_path: Path,
        motion_prompt: str,
        duration_sec: float,
        dest_mp4: Path,
        timeout_sec: int = 240,
    ) -> Optional[Path]:
        """
        Uploads `keyframe_path` to ComfyUI, executes the I2V workflow (`ltx_i2v` or `svd_xt`),
        downloads the animated output, and transcodes it to a crisp `1080x1920` @ 30fps `.mp4` clip
        matching `duration_sec`. Returns `None` if ComfyUI is not active or budget is reached.
        """
        if self.clips_rendered >= self.max_clips or not self.is_server_reachable():
            return None

        uploaded_name = self._upload_keyframe(keyframe_path)
        if not uploaded_name:
            return None

        mode = self._detect_available_workflow()
        workflow = (
            self._build_svd_xt_prompt(uploaded_name)
            if mode == "svd_xt"
            else self._build_ltx_i2v_prompt(uploaded_name, motion_prompt)
        )

        try:
            resp = self.session.post(
                f"{self.api_url}/prompt",
                json={"prompt": workflow, "client_id": self.client_id},
                timeout=20,
            )
            if resp.status_code != 200:
                print(f"[ComfyUI] Prompt submission notice ({resp.status_code}): {resp.text[:240]}")
                return None
            prompt_id = resp.json().get("prompt_id")
            if not prompt_id:
                return None

            print(f"[ComfyUI] Rendering realistic {mode.upper()} video clip (prompt_id={prompt_id[:10]})...")
            deadline = time.time() + timeout_sec
            output_file_info: Optional[Dict[str, str]] = None

            while time.time() < deadline:
                time.sleep(2.5)
                hist_r = self.session.get(f"{self.api_url}/history/{prompt_id}", timeout=12)
                if hist_r.status_code != 200:
                    continue
                hist_data = hist_r.json()
                if prompt_id not in hist_data:
                    continue
                outputs = hist_data[prompt_id].get("outputs", {})
                for node_out in outputs.values():
                    for key in ("images", "gifs", "videos"):
                        items = node_out.get(key, [])
                        if items:
                            output_file_info = items[0]
                            break
                    if output_file_info:
                        break
                if output_file_info:
                    break

            if not output_file_info:
                print(f"[ComfyUI] Timed out waiting for prompt_id={prompt_id[:10]}")
                return None

            view_params = {
                "filename": output_file_info.get("filename", ""),
                "subfolder": output_file_info.get("subfolder", ""),
                "type": output_file_info.get("type", "output"),
            }
            dl_r = self.session.get(f"{self.api_url}/view", params=view_params, timeout=45)
            dl_r.raise_for_status()

            raw_ext = Path(view_params["filename"]).suffix or ".webp"
            raw_anim = dest_mp4.parent / f"_raw_comfy_{dest_mp4.stem}{raw_ext}"
            raw_anim.write_bytes(dl_r.content)

            target_dur = max(1.5, float(duration_sec))
            vf_chain = (
                f"scale={self.WIDTH}:{self.HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={self.WIDTH}:{self.HEIGHT},"
                f"minterpolate=fps={self.FPS}:mi_mode=mci:mc_mode=aobmc:vsbmc=1,"
                f"eq=contrast=1.05:saturation=0.98,"
                f"unsharp=5:5:0.6:5:5:0.0,"
                f"setsar=1,format=yuv420p"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-stream_loop",
                "-1",
                "-i",
                str(raw_anim),
                "-t",
                f"{target_dur:.3f}",
                "-vf",
                vf_chain,
                "-r",
                str(self.FPS),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(dest_mp4),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
            raw_anim.unlink(missing_ok=True)
            if proc.returncode == 0 and dest_mp4.exists() and dest_mp4.stat().st_size > 15_000:
                self.clips_rendered += 1
                print(
                    f"[ComfyUI] Generated 1080x1920 realistic AI video clip -> {dest_mp4.name} "
                    f"({self.clips_rendered}/{self.max_clips})"
                )
                return dest_mp4
        except Exception as exc:
            print(f"[ComfyUI] Video generation notice: {exc}")
        return None
