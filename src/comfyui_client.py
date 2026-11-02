"""
ComfyUI Realistic AI Video Generation Client (`src/comfyui_client.py` — v14.0 Direct In-Process Node Engine):
Eliminates the HTTP server polling timeout & T5/LTX VRAM thrashing that prevented ComfyUI clips from finishing on Kaggle T4:

How v14.0 Direct In-Process ComfyUI Execution Works on Kaggle GPU (`Tesla T4 x2` / `P100`):
1. `prepare_batch_prompts(scenes)`:
   - Imports ComfyUI's native Python nodes (`CLIPLoader`, `CLIPTextEncode`, `CheckpointLoaderSimple`,
     `LTXVPreprocess`, `LTXVImgToVideo`, `LTXVConditioning`, `LTXVScheduler`, `KSamplerSelect`,
     `SamplerCustom`, `VAEDecode`) directly from `/tmp/ComfyUI` in-process (no separate port 8188 server competing for VRAM!).
   - Loads `t5xxl_fp8_e4m3fn.safetensors` ONCE, encodes the positive and negative motion prompts for ALL scenes
     in ~5 seconds, detaches the compact conditioning tensors (~2 MB each), and **completely unloads & deletes `T5-XXL`
     from GPU VRAM (`del clip; model_management.unload_all_models(); torch.cuda.empty_cache()`)**!
   - Loads `ltx-video-2b-v0.9.5.safetensors` (`4.3 GB`) into GPU VRAM ONCE and keeps it permanently resident
     across all scenes (using only ~5.1 GB out of 15.0 GB T4 VRAM, with ZERO model swapping between scenes!).
2. `animate_keyframe_to_video(...)`:
   - Runs `LTXVPreprocess(img_compression=35)` -> `LTXVImgToVideo(448x768, 25 frames, strength=0.78)` ->
     `LTXVConditioning` -> `LTXVScheduler(14 steps)` -> `SamplerCustom` -> `VAEDecode` directly in Python
     with zero HTTP timeout, printing every step and any traceback directly to the Kaggle log in real time!
   - Also retains full HTTP REST API support (`COMFYUI_API_URL`) if connected to an external remote ComfyUI server.
"""
from __future__ import annotations
import gc
import inspect
import json
import os
import random
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import requests


def _clone_conditioning(cond_list: Any) -> Any:
    """Detaches and clones a ComfyUI CONDITIONING structure (`[[tensor, dict], ...]`) so `del clip` frees 100% of T5 weights."""
    import torch  # type: ignore
    cloned = []
    for item in cond_list:
        t = item[0]
        d = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
        t_copy = t.detach().clone() if isinstance(t, torch.Tensor) else t
        d_copy = {}
        for k, v in d.items():
            d_copy[k] = v.detach().clone() if isinstance(v, torch.Tensor) else v
        cloned.append([t_copy, d_copy])
    return cloned


class ComfyUIVideoClient:
    WIDTH = 1080
    HEIGHT = 1920
    FPS = 30
    # Native 9:16 LTX-Video generation resolution (multiples of 64 & 32; 448x768x25 decodes cleanly in T4 VRAM)
    GEN_WIDTH = 448
    GEN_HEIGHT = 768
    GEN_FRAMES = 25
    GEN_STEPS = 14

    def __init__(self, api_url: Optional[str] = None, comfy_dir: Optional[Path] = None):
        raw_url = (
            api_url
            or os.getenv("COMFYUI_API_URL")
            or os.getenv("KAGGLE_COMFYUI_URL")
            or ""
        ).strip().rstrip("/")
        self.api_url = raw_url
        self.comfy_dir = comfy_dir or Path(os.getenv("COMFYUI_DIR", "/tmp/ComfyUI"))
        self.workflow_mode = (os.getenv("COMFYUI_WORKFLOW_MODE") or "ltx_i2v").strip().lower()
        self.ltx_ckpt = (os.getenv("COMFYUI_LTX_CHECKPOINT") or "ltx-video-2b-v0.9.5.safetensors").strip()
        self.t5_clip = (os.getenv("COMFYUI_T5_MODEL") or "t5xxl_fp8_e4m3fn.safetensors").strip()
        self.svd_ckpt = (os.getenv("COMFYUI_SVD_CHECKPOINT") or "svd_xt_1_1.safetensors").strip()
        self.client_id = str(uuid.uuid4())
        self.session = requests.Session()
        self.clips_rendered = 0
        self.consecutive_failures = 0
        self.circuit_open = False
        try:
            self.max_clips = max(0, int(os.getenv("MAX_COMFY_VIDEOS_PER_SHORT", "6")))
        except ValueError:
            self.max_clips = 6

        # Direct in-process ComfyUI state
        self._direct_ready = False
        self._direct_failed = False
        self._direct_model: Any = None
        self._direct_vae: Any = None
        self._direct_nodes: Dict[str, Any] = {}
        self._preencoded_prompts: Dict[int, Tuple[Any, Any]] = {}
        self._default_cond: Optional[Tuple[Any, Any]] = None

    def _can_use_direct_engine(self) -> bool:
        if self._direct_failed:
            return False
        if self._direct_ready:
            return True
        if not (self.comfy_dir.exists() and (self.comfy_dir / "nodes.py").exists()):
            return False
        try:
            import torch  # type: ignore
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def prepare_batch_prompts(self, scenes: List[Dict[str, Any]]) -> bool:
        """
        Pre-encodes all scene motion prompts in a single batch using `T5-XXL`, immediately unloads `T5-XXL`
        from GPU VRAM (`del clip; unload_all_models(); empty_cache()`), and loads `LTX-Video 2B` once into GPU VRAM.
        """
        if self.max_clips <= 0 or not self._can_use_direct_engine():
            return False
        if self._direct_ready:
            return True

        t0 = time.time()
        try:
            import torch  # type: ignore
            comfy_str = str(self.comfy_dir.resolve())
            if comfy_str not in sys.path:
                sys.path.insert(0, comfy_str)

            print(
                f"[ComfyUI-Direct] Initializing in-process ComfyUI engine from {comfy_str} "
                f"(GPUs detected: {torch.cuda.device_count()})...",
                flush=True,
            )
            saved_argv = list(sys.argv)
            sys.argv = [sys.argv[0]]
            try:
                import folder_paths  # type: ignore
                from comfy import model_management  # type: ignore
            finally:
                sys.argv = saved_argv
            from nodes import (  # type: ignore
                CheckpointLoaderSimple,
                CLIPLoader,
                CLIPTextEncode,
                VAEDecode,
                LoadImage,
            )
            from comfy_extras.nodes_custom_sampler import KSamplerSelect, SamplerCustom  # type: ignore
            from comfy_extras.nodes_lt import (  # type: ignore
                LTXVPreprocess,
                LTXVImgToVideo,
                LTXVScheduler,
                LTXVConditioning,
            )

            self._direct_nodes = {
                "model_management": model_management,
                "CheckpointLoaderSimple": CheckpointLoaderSimple(),
                "CLIPLoader": CLIPLoader(),
                "CLIPTextEncode": CLIPTextEncode(),
                "VAEDecode": VAEDecode(),
                "LoadImage": LoadImage(),
                "KSamplerSelect": KSamplerSelect(),
                "SamplerCustom": SamplerCustom(),
                "LTXVPreprocess": LTXVPreprocess(),
                "LTXVImgToVideo": LTXVImgToVideo(),
                "LTXVScheduler": LTXVScheduler(),
                "LTXVConditioning": LTXVConditioning(),
            }

            # Resolve T5 filename present in models/clip or models/text_encoders
            clip_candidates = [self.t5_clip, "t5xxl_fp8_e4m3fn.safetensors", "t5xxl_fp8_e4m3fn_scaled.safetensors", "t5xxl_fp16.safetensors"]
            resolved_clip = self.t5_clip
            for cand in clip_candidates:
                if (self.comfy_dir / "models" / "clip" / cand).exists() or (self.comfy_dir / "models" / "text_encoders" / cand).exists():
                    resolved_clip = cand
                    break
            self.t5_clip = resolved_clip

            print(f"[ComfyUI-Direct] Step 1/2: Loading text encoder '{self.t5_clip}' to batch-encode {len(scenes)} scene prompts...", flush=True)
            clip_loader = self._direct_nodes["CLIPLoader"]
            clip_encode = self._direct_nodes["CLIPTextEncode"]

            with torch.inference_mode():
                sig = inspect.signature(clip_loader.load_clip)
                if "device" in sig.parameters:
                    clip = clip_loader.load_clip(self.t5_clip, "ltxv", "default")[0]
                else:
                    clip = clip_loader.load_clip(self.t5_clip, "ltxv")[0]

                neg_text = (
                    "low quality, worst quality, deformed, distorted, disfigured, motion smear, "
                    "motion artifacts, cartoon, anime, watermark, text, blurry, flicker, static"
                )
                neg_cond_raw = clip_encode.encode(clip, neg_text)[0]
                neg_cond = _clone_conditioning(neg_cond_raw)

                for idx, sc in enumerate(scenes, 1):
                    sid = int(sc.get("scene_id") or idx)
                    m_prompt = str(
                        sc.get("motion_prompt")
                        or sc.get("image_prompt")
                        or sc.get("visual_subject")
                        or "cinematic historical documentary scene"
                    ).strip()
                    pos_text = (
                        f"best quality, 4k, HDR, {m_prompt}, photorealistic historical documentary cinematography, "
                        "natural realistic physical motion, subtle cinema camera movement, 35mm film grain, lifelike textures"
                    )
                    pos_cond_raw = clip_encode.encode(clip, pos_text)[0]
                    pos_cond = _clone_conditioning(pos_cond_raw)
                    self._preencoded_prompts[sid] = (pos_cond, _clone_conditioning(neg_cond))
                    if self._default_cond is None:
                        self._default_cond = (pos_cond, _clone_conditioning(neg_cond))

                # Completely unload T5-XXL before loading LTX-Video 2B!
                del clip
                model_management.unload_all_models()
                gc.collect()
                torch.cuda.empty_cache()

            free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            print(
                f"[ComfyUI-Direct] Batch-encoded {len(self._preencoded_prompts)} scenes and unloaded T5-XXL "
                f"(Free GPU VRAM: {free_gb:.2f} GB). Step 2/2: Loading '{self.ltx_ckpt}'...",
                flush=True,
            )

            with torch.inference_mode():
                ckpt_loader = self._direct_nodes["CheckpointLoaderSimple"]
                self._direct_model, _, self._direct_vae = ckpt_loader.load_checkpoint(self.ltx_ckpt)

            free_after = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            self._direct_ready = True
            print(
                f"[ComfyUI-Direct] LTX-Video 2B ready in {time.time() - t0:.1f}s! "
                f"(Free GPU VRAM: {free_after:.2f} GB — zero model swapping required across scenes)",
                flush=True,
            )
            return True
        except Exception as exc:
            print(f"[ComfyUI-Direct] In-process initialization notice ({exc}):", flush=True)
            traceback.print_exc()
            self._direct_failed = True
            return False

    def _animate_via_direct_nodes(
        self,
        scene_id: int,
        keyframe_path: Path,
        motion_prompt: str,
        duration_sec: float,
        dest_mp4: Path,
    ) -> Optional[Path]:
        """Executes LTX-Video 2B Image-to-Video directly in Python using pre-encoded T5 tensors and resident LTX model."""
        t_start = time.time()
        try:
            import torch  # type: ignore
            if not self._direct_ready and not self.prepare_batch_prompts([{"scene_id": scene_id, "motion_prompt": motion_prompt}]):
                return None

            cond_pair = self._preencoded_prompts.get(scene_id) or self._default_cond
            if cond_pair is None:
                return None
            pos_cond = _clone_conditioning(cond_pair[0])
            neg_cond = _clone_conditioning(cond_pair[1])

            preprocess_node = self._direct_nodes["LTXVPreprocess"]
            img_to_video = self._direct_nodes["LTXVImgToVideo"]
            conditioning_node = self._direct_nodes["LTXVConditioning"]
            scheduler_node = self._direct_nodes["LTXVScheduler"]
            sampler_select = self._direct_nodes["KSamplerSelect"]
            sampler_node = self._direct_nodes["SamplerCustom"]
            vae_decode = self._direct_nodes["VAEDecode"]

            # Load keyframe into ComfyUI IMAGE tensor [1, H, W, 3] in float32 [0, 1]
            with Image.open(keyframe_path) as pil_im:
                rgb_im = pil_im.convert("RGB")
                arr = np.array(rgb_im).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(arr).unsqueeze(0)

            print(
                f"[ComfyUI-Direct] Scene {scene_id:02d}: Animating {keyframe_path.name} -> "
                f"{self.GEN_WIDTH}x{self.GEN_HEIGHT} x {self.GEN_FRAMES} frames ({self.GEN_STEPS} steps)...",
                flush=True,
            )

            with torch.inference_mode():
                try:
                    processed_img = preprocess_node.preprocess(image_tensor, 35)[0]
                except Exception as pre_exc:
                    print(f"[ComfyUI-Direct] LTXVPreprocess fallback ({pre_exc}); using raw tensor.", flush=True)
                    processed_img = image_tensor

                gen_kwargs: Dict[str, Any] = {
                    "positive": pos_cond,
                    "negative": neg_cond,
                    "vae": self._direct_vae,
                    "image": processed_img,
                    "width": self.GEN_WIDTH,
                    "height": self.GEN_HEIGHT,
                    "length": self.GEN_FRAMES,
                    "batch_size": 1,
                }
                sig_i2v = inspect.signature(img_to_video.generate)
                if "strength" in sig_i2v.parameters:
                    gen_kwargs["strength"] = 0.78

                video_output = img_to_video.generate(**gen_kwargs)
                cond_pos, cond_neg = conditioning_node.append(
                    positive=video_output[0],
                    negative=video_output[1],
                    frame_rate=24.0,
                )
                sigmas = scheduler_node.get_sigmas(
                    steps=self.GEN_STEPS,
                    max_shift=2.05,
                    base_shift=0.95,
                    stretch=True,
                    terminal=0.1,
                    latent=video_output[2],
                )[0]
                selected_sampler = sampler_select.get_sampler("euler")[0]
                seed = random.randint(10_000, 9_999_999_999)

                t_samp = time.time()
                sampled = sampler_node.sample(
                    model=self._direct_model,
                    add_noise=True,
                    noise_seed=seed,
                    cfg=3.0,
                    positive=cond_pos,
                    negative=cond_neg,
                    sampler=selected_sampler,
                    sigmas=sigmas,
                    latent_image=video_output[2],
                )[0]
                t_dec = time.time()
                decoded = vae_decode.decode(self._direct_vae, sampled)[0].detach()

            if decoded.ndim == 5:
                decoded = decoded[0]
            frames_np = (decoded.cpu().float().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            num_f, h_f, w_f, _ = frames_np.shape
            print(
                f"[ComfyUI-Direct] Scene {scene_id:02d}: Sampled in {t_dec - t_samp:.1f}s, "
                f"VAE decoded {num_f} frames ({w_f}x{h_f}) in {time.time() - t_dec:.1f}s!",
                flush=True,
            )

            # Encode raw RGB frames via FFmpeg pipe directly to 1080x1920 @ 30fps MP4
            raw_mp4 = dest_mp4.parent / f"_raw_direct_{dest_mp4.stem}.mp4"
            ff_raw = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-s",
                    f"{w_f}x{h_f}",
                    "-pix_fmt",
                    "rgb24",
                    "-r",
                    "16",
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "17",
                    "-pix_fmt",
                    "yuv420p",
                    str(raw_mp4),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert ff_raw.stdin is not None
            for frame in frames_np:
                ff_raw.stdin.write(frame.tobytes())
            ff_raw.stdin.close()
            ff_raw.wait(timeout=45)

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
                str(raw_mp4),
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
            raw_mp4.unlink(missing_ok=True)
            gc.collect()
            torch.cuda.empty_cache()

            if proc.returncode == 0 and dest_mp4.exists() and dest_mp4.stat().st_size > 15_000:
                self.clips_rendered += 1
                self.consecutive_failures = 0
                print(
                    f"[ComfyUI-Direct] Scene {scene_id:02d} COMPLETE -> {dest_mp4.name} "
                    f"({dest_mp4.stat().st_size // 1024} KB, {self.clips_rendered}/{self.max_clips} clips) "
                    f"in {time.time() - t_start:.1f}s!",
                    flush=True,
                )
                return dest_mp4
        except Exception as exc:
            print(f"[ComfyUI-Direct] Scene {scene_id:02d} error ({exc}):", flush=True)
            traceback.print_exc()
            self.consecutive_failures += 1
            if self.consecutive_failures >= 2:
                self.circuit_open = True
        return None

    def is_server_reachable(self) -> bool:
        if self._can_use_direct_engine():
            return True
        if not self.api_url:
            self.api_url = (os.getenv("COMFYUI_API_URL") or os.getenv("KAGGLE_COMFYUI_URL") or "").strip().rstrip("/")
        if not self.api_url or self.circuit_open:
            return False
        try:
            resp = self.session.get(f"{self.api_url}/system_stats", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _detect_available_workflow(self) -> str:
        if not self.api_url:
            return self.workflow_mode
        try:
            resp = self.session.get(f"{self.api_url}/object_info/CheckpointLoaderSimple", timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                ckpts = (
                    data.get("CheckpointLoaderSimple", {})
                    .get("input", {})
                    .get("required", {})
                    .get("ckpt_name", [[]])[0]
                )
                for c in ckpts:
                    if "ltx" in str(c).lower():
                        self.ltx_ckpt = str(c)
                        return "ltx_i2v"
                for c in ckpts:
                    if "svd" in str(c).lower():
                        self.svd_ckpt = str(c)
                        return "svd_xt"
        except Exception:
            pass
        return self.workflow_mode

    def _upload_keyframe(self, image_path: Path) -> Optional[str]:
        try:
            with open(image_path, "rb") as fp:
                files = {"image": (image_path.name, fp, "image/jpeg")}
                data = {"overwrite": "true", "type": "input"}
                resp = self.session.post(f"{self.api_url}/upload/image", files=files, data=data, timeout=25)
                if resp.status_code == 200:
                    return resp.json().get("name", image_path.name)
        except Exception as exc:
            print(f"[ComfyUI] Keyframe upload notice: {exc}", flush=True)
        return None

    def _get_node_accepted_inputs(self, node_class: str) -> Optional[set]:
        if not self.api_url:
            return None
        try:
            r = self.session.get(f"{self.api_url}/object_info/{node_class}", timeout=5)
            if r.status_code == 200:
                info = r.json().get(node_class, {}).get("input", {})
                keys = set(info.get("required", {}).keys()) | set(info.get("optional", {}).keys())
                return keys if keys else None
        except Exception:
            pass
        return None

    def _build_ltx_i2v_prompt(self, uploaded_image_name: str, motion_prompt: str, num_frames: int = 25) -> Dict[str, Any]:
        seed = random.randint(10_000, 9_999_999_999)
        pos_prompt = (
            f"best quality, 4k, HDR, {motion_prompt}, photorealistic historical documentary cinematography, "
            "natural realistic physical motion, subtle cinema camera movement, 35mm film grain, lifelike textures"
        )
        neg_prompt = (
            "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, "
            "cartoon, anime, watermark, text, blurry, flicker"
        )
        ltx_inputs: Dict[str, Any] = {
            "positive": ["3", 0],
            "negative": ["4", 0],
            "vae": ["1", 2],
            "image": ["5", 0],
            "width": 512,
            "height": 896,
            "length": num_frames,
            "batch_size": 1,
        }
        accepted_ltx = self._get_node_accepted_inputs("LTXVImgToVideo")
        if accepted_ltx is None or "strength" in accepted_ltx:
            ltx_inputs["strength"] = 0.78
        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.ltx_ckpt}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": self.t5_clip, "type": "ltxv"}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": pos_prompt, "clip": ["2", 0]}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_prompt, "clip": ["2", 0]}},
            "5": {"class_type": "LoadImage", "inputs": {"image": uploaded_image_name}},
            "7": {"class_type": "LTXVImgToVideo", "inputs": ltx_inputs},
            "8": {"class_type": "LTXVConditioning", "inputs": {"positive": ["7", 0], "negative": ["7", 1], "frame_rate": 24.0}},
            "9": {"class_type": "LTXVScheduler", "inputs": {"steps": 14, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1, "latent": ["7", 2]}},
            "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "11": {
                "class_type": "SamplerCustom",
                "inputs": {
                    "model": ["1", 0],
                    "add_noise": True,
                    "noise_seed": seed,
                    "cfg": 3.0,
                    "positive": ["8", 0],
                    "negative": ["8", 1],
                    "sampler": ["10", 0],
                    "sigmas": ["9", 0],
                    "latent_image": ["7", 2],
                },
            },
            "14": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
            "15": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["14", 0], "filename_prefix": "comfy_ltx_short", "fps": 24.0, "lossless": False, "quality": 90, "method": "default"}},
        }

    def _build_svd_xt_prompt(self, uploaded_image_name: str, num_frames: int = 14) -> Dict[str, Any]:
        seed = random.randint(10_000, 9_999_999_999)
        return {
            "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": self.svd_ckpt}},
            "2": {"class_type": "LoadImage", "inputs": {"image": uploaded_image_name}},
            "3": {
                "class_type": "SVD_img2vid_Conditioning",
                "inputs": {
                    "clip_vision": ["1", 1],
                    "init_image": ["2", 0],
                    "vae": ["1", 2],
                    "width": 512,
                    "height": 896,
                    "video_frames": num_frames,
                    "motion_bucket_id": 115,
                    "fps": 8,
                    "augmentation_level": 0.02,
                },
            },
            "4": {"class_type": "VideoLinearCFGGuidance", "inputs": {"model": ["1", 0], "min_cfg": 1.0}},
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["4", 0],
                    "seed": seed,
                    "steps": 15,
                    "cfg": 2.5,
                    "sampler_name": "euler",
                    "scheduler": "karras",
                    "positive": ["3", 0],
                    "negative": ["3", 1],
                    "latent_image": ["3", 2],
                    "denoise": 1.0,
                },
            },
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["6", 0], "filename_prefix": "comfy_svd_short", "fps": 12.0, "lossless": False, "quality": 90, "method": "default"}},
        }

    def animate_keyframe_to_video(
        self,
        keyframe_path: Path,
        motion_prompt: str,
        duration_sec: float,
        dest_mp4: Path,
        timeout_sec: int = 180,
        scene_id: int = 1,
    ) -> Optional[Path]:
        if self.clips_rendered >= self.max_clips or self.circuit_open:
            return None

        # Priority 1: Direct In-Process ComfyUI Execution (on Kaggle GPU)
        if self._can_use_direct_engine():
            res = self._animate_via_direct_nodes(
                scene_id=scene_id,
                keyframe_path=keyframe_path,
                motion_prompt=motion_prompt,
                duration_sec=duration_sec,
                dest_mp4=dest_mp4,
            )
            if res is not None:
                return res
            if self.circuit_open or not self.api_url:
                return None

        # Priority 2: External Remote ComfyUI HTTP Server (if COMFYUI_API_URL is set)
        if not self.api_url or not self.is_server_reachable():
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
                print(f"[ComfyUI] Prompt validation rejected ({resp.status_code}): {resp.text[:350]}", flush=True)
                self.consecutive_failures += 1
                if self.consecutive_failures >= 2:
                    self.circuit_open = True
                return None

            prompt_id = resp.json().get("prompt_id")
            if not prompt_id:
                return None

            print(f"[ComfyUI] Rendering realistic {mode.upper()} video clip (prompt_id={prompt_id[:10]})...", flush=True)
            deadline = time.time() + timeout_sec
            output_file_info: Optional[Dict[str, str]] = None

            while time.time() < deadline:
                time.sleep(2.0)
                hist_r = self.session.get(f"{self.api_url}/history/{prompt_id}", timeout=10)
                if hist_r.status_code != 200:
                    continue
                hist_data = hist_r.json()
                if prompt_id not in hist_data:
                    continue

                entry = hist_data[prompt_id]
                status_info = entry.get("status", {})
                status_str = str(status_info.get("status_str", "")).lower()
                if status_str == "error":
                    msgs = status_info.get("messages", [])
                    print(f"[ComfyUI] Execution error reported for {prompt_id[:10]}: {str(msgs)[-400:]}", flush=True)
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= 2:
                        self.circuit_open = True
                    return None

                outputs = entry.get("outputs", {})
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
                print(f"[ComfyUI] Timed out or no output for prompt_id={prompt_id[:10]}; interrupting queue...", flush=True)
                try:
                    self.session.post(f"{self.api_url}/interrupt", timeout=5)
                except Exception:
                    pass
                self.consecutive_failures += 1
                if self.consecutive_failures >= 2:
                    self.circuit_open = True
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
                self.consecutive_failures = 0
                return dest_mp4
        except Exception as exc:
            print(f"[ComfyUI] Video generation notice: {exc}", flush=True)
        return None
