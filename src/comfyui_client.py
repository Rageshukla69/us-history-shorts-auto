"""
ComfyUI Realistic AI Video Generation Client (`src/comfyui_client.py` — v16.0 Smart Scene Filter + Seamless Ping-Pong LTX-2B):
1. Real In-Process ComfyUI `LTX-Video 2B` (`ltx-video-2b-v0.9.5.safetensors` + `t5xxl_fp8_e4m3fn.safetensors`) —
   NO cheap synthetic snowfall/particle overlays!
2. Smart Scene Filter (`is_scene_suitable_for_i2v`):
   - Open-weights 2B video models on a 15 GB Tesla T4 GPU compress images through a 32x spatial VAE (`16x28` latent grid).
     They excel at wide environmental/atmospheric motion (rivers, eclipses, clouds, smoke, fire, ships, wide battlefields),
     but warp close-up human hands, writing quills, faces, and geometric spheres (globes).
   - `is_scene_suitable_for_i2v(scene)` automatically routes ONLY wide atmospheric/environmental scenes (up to 4 per Short)
     to ComfyUI `LTX-Video 2B`, while keeping close-up portraits, hands writing, and globes as razor-sharp `1080x1920`
     `MAI-Image-2.6-Flash` keyframes animated by HyperFrames' 60fps GSAP camera rig!
3. Seamless Ping-Pong Slow-Motion (`frames + frames[-2:0:-1]`):
   - Eliminates the jarring 1.5-second `-stream_loop -1` hard snap seen in `1000047128.mp4` by encoding a smooth
     forward-and-reverse ping-pong cycle at 12fps -> 30fps (~4.1s seamless cycle) with `strength=0.94` and `cfg=2.2`.
"""
from __future__ import annotations
import gc
import inspect
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageSequence
import requests

# Close-up anatomy / fine geometry terms that 2B diffusion models on 15GB T4 warp or melt
CLOSEUP_ANATOMY_OR_TEXT_RE = re.compile(
    r"\b(hand|hands|finger|fingers|writing|wrote|quill|pen|scroll|parchment|manuscript|letter|book|"
    r"globe|orb|sphere|compass|astrolabe|telescope|chart|diagram|map|portrait|face|eyes|beard|"
    r"herodotus|thales|astronomer|scholar|scribe|philosopher|marriage|wedding|bride|groom|handshake|"
    r"seal|stamp|coin|tablet|inscription)\b",
    re.IGNORECASE,
)

# Wide environmental / atmospheric motion terms where LTX-Video 2B looks natural on T4
WIDE_ATMOSPHERIC_MOTION_RE = re.compile(
    r"\b(river|water|waves|sea|ocean|eclipse|sky|clouds|sun|moon|stars|corona|smoke|fire|flames|"
    r"torch|camp|battlefield|valley|canyon|mountains|wind|banner|flag|storm|mist|fog|landscape|horizon)\b",
    re.IGNORECASE,
)


def _clone_conditioning(cond_list: Any) -> Any:
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


def _invoke_comfy_node(node_obj: Any, *args: Any, **kwargs: Any) -> Any:
    fn = getattr(node_obj, "execute", None)
    if fn is None:
        func_name = getattr(node_obj, "FUNCTION", None)
        if func_name and hasattr(node_obj, func_name):
            fn = getattr(node_obj, func_name)
        else:
            for cand in ("load_checkpoint", "load_clip", "encode", "decode", "generate", "sample", "preprocess"):
                if hasattr(node_obj, cand):
                    fn = getattr(node_obj, cand)
                    break
    if fn is None:
        raise RuntimeError(f"Could not find executable method on ComfyUI node {type(node_obj).__name__}")
    res = fn(*args, **kwargs)
    return getattr(res, "args", res)


class ComfyUIVideoClient:
    WIDTH = 1080
    HEIGHT = 1920
    FPS = 30
    GEN_WIDTH = 512
    GEN_HEIGHT = 896
    GEN_FRAMES = 25
    GEN_STEPS = 20

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
            self.max_clips = max(0, int(os.getenv("MAX_COMFY_VIDEOS_PER_SHORT", "4")))
        except ValueError:
            self.max_clips = 4

        self._direct_ready = False
        self._direct_failed = False
        self._direct_model: Any = None
        self._direct_vae: Any = None
        self._direct_modules: Dict[str, Any] = {}
        self._preencoded_prompts: Dict[int, Tuple[Any, Any]] = {}
        self._default_cond: Optional[Tuple[Any, Any]] = None
        self._scene_suitability: Dict[int, bool] = {}

    @staticmethod
    def is_scene_suitable_for_i2v(scene_text: str) -> bool:
        """
        Returns True ONLY if the scene depicts wide environmental/atmospheric motion (sky, eclipse, river,
        battlefield, clouds, fire, smoke) and does NOT feature close-up hands writing, globes, or portraits
        (which 2B models on 15GB T4 warp/melt).
        """
        if CLOSEUP_ANATOMY_OR_TEXT_RE.search(scene_text):
            return False
        if WIDE_ATMOSPHERIC_MOTION_RE.search(scene_text):
            return True
        return True

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
                f"[ComfyUI-Direct] Initializing in-process ComfyUI LTX-Video 2B engine from {comfy_str} "
                f"(GPUs detected: {torch.cuda.device_count()})...",
                flush=True,
            )
            saved_argv = list(sys.argv)
            sys.argv = [sys.argv[0]]
            try:
                import folder_paths  # type: ignore
                import node_helpers  # type: ignore
                import comfy.model_management as model_management  # type: ignore
                import comfy.utils as comfy_utils  # type: ignore
                import comfy.samplers as comfy_samplers  # type: ignore
                import comfy.sample as comfy_sample  # type: ignore
                from nodes import CheckpointLoaderSimple, CLIPLoader, CLIPTextEncode  # type: ignore
            finally:
                sys.argv = saved_argv

            self._direct_modules = {
                "model_management": model_management,
                "comfy_utils": comfy_utils,
                "comfy_samplers": comfy_samplers,
                "comfy_sample": comfy_sample,
                "node_helpers": node_helpers,
                "CheckpointLoaderSimple": CheckpointLoaderSimple(),
                "CLIPLoader": CLIPLoader(),
                "CLIPTextEncode": CLIPTextEncode(),
            }

            clip_candidates = [
                self.t5_clip,
                "t5xxl_fp8_e4m3fn.safetensors",
                "t5xxl_fp8_e4m3fn_scaled.safetensors",
                "t5xxl_fp16.safetensors",
            ]
            for cand in clip_candidates:
                if (self.comfy_dir / "models" / "clip" / cand).exists() or (self.comfy_dir / "models" / "text_encoders" / cand).exists():
                    self.t5_clip = cand
                    break

            print(
                f"[ComfyUI-Direct] Step 1/2: Loading text encoder '{self.t5_clip}' to batch-encode {len(scenes)} scene prompts...",
                flush=True,
            )
            clip_loader = self._direct_modules["CLIPLoader"]
            clip_encode = self._direct_modules["CLIPTextEncode"]

            with torch.inference_mode():
                try:
                    clip = _invoke_comfy_node(clip_loader, self.t5_clip, "ltxv", "default")[0]
                except TypeError:
                    clip = _invoke_comfy_node(clip_loader, self.t5_clip, "ltxv")[0]

                neg_text = (
                    "low quality, worst quality, deformed, distorted, disfigured, morphing, melting, "
                    "motion smear, motion artifacts, cartoon, anime, watermark, text, blurry, flicker"
                )
                neg_cond_raw = _invoke_comfy_node(clip_encode, clip, neg_text)[0]
                neg_cond = _clone_conditioning(neg_cond_raw)

                for idx, sc in enumerate(scenes, 1):
                    sid = int(sc.get("scene_id") or idx)
                    combined_desc = " ".join(
                        str(sc.get(k) or "") for k in ("visual_subject", "image_prompt", "motion_prompt", "narration")
                    )
                    self._scene_suitability[sid] = self.is_scene_suitable_for_i2v(combined_desc)

                    m_prompt = str(
                        sc.get("motion_prompt")
                        or sc.get("image_prompt")
                        or sc.get("visual_subject")
                        or "cinematic historical documentary scene"
                    ).strip()
                    pos_text = (
                        f"best quality, 4k, HDR, {m_prompt}, subtle natural atmospheric motion, "
                        "steady cinema camera, photorealistic historical documentary, 35mm film grain, high detail"
                    )
                    pos_cond_raw = _invoke_comfy_node(clip_encode, clip, pos_text)[0]
                    pos_cond = _clone_conditioning(pos_cond_raw)
                    self._preencoded_prompts[sid] = (pos_cond, _clone_conditioning(neg_cond))
                    if self._default_cond is None:
                        self._default_cond = (pos_cond, _clone_conditioning(neg_cond))

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
                ckpt_loader = self._direct_modules["CheckpointLoaderSimple"]
                loaded_ckpt = _invoke_comfy_node(ckpt_loader, self.ltx_ckpt)
                self._direct_model = loaded_ckpt[0]
                self._direct_vae = loaded_ckpt[2]

            free_after = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            self._direct_ready = True
            suitable_ids = [k for k, v in self._scene_suitability.items() if v]
            print(
                f"[ComfyUI-Direct] LTX-Video 2B ready in {time.time() - t0:.1f}s! "
                f"(Free GPU VRAM: {free_after:.2f} GB; wide atmospheric scenes eligible for I2V: {suitable_ids})",
                flush=True,
            )
            return True
        except Exception as exc:
            print(f"[ComfyUI-Direct] In-process initialization notice ({exc}):", flush=True)
            traceback.print_exc()
            self._direct_failed = True
            return False

    def _passes_consistency_gate(self, frames_np: np.ndarray) -> bool:
        """
        Checks if the generated LTX-Video frames remain structurally consistent with Frame 0
        (rejecting clips that develop localized hallucination blobs like the brown smudge on the globe).
        """
        if len(frames_np) < 4:
            return False
        f0 = frames_np[0].astype(np.float32) / 255.0
        f_last = frames_np[-1].astype(np.float32) / 255.0
        # Global mean absolute drift
        mean_drift = float(np.abs(f_last - f0).mean())
        if mean_drift > 0.22 or mean_drift < 0.002:
            print(f"[ComfyUI-Direct] Consistency gate rejected clip (mean_drift={mean_drift:.3f}); keeping sharp 1080x1920 MAI keyframe.", flush=True)
            return False
        return True

    def _encode_frames_to_mp4(self, frames_np: np.ndarray, duration_sec: float, dest_mp4: Path) -> Optional[Path]:
        """
        Encodes `frames_np` as a seamless ping-pong (forward + reverse) slow-motion sequence at 12fps -> 30fps
        so there is NEVER a jarring 1.5-second `-stream_loop -1` jump back to frame 0!
        """
        num_f, h_f, w_f, _ = frames_np.shape
        # Construct seamless ping-pong sequence: 0..N-1 followed by N-2..1 (48 frames @ 12fps = 4.0s seamless cycle)
        if num_f > 4:
            ping_pong_frames = np.concatenate([frames_np, frames_np[-2:0:-1]], axis=0)
        else:
            ping_pong_frames = frames_np

        raw_mp4 = dest_mp4.parent / f"_raw_direct_{dest_mp4.stem}.mp4"
        try:
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
                    "12",
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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            assert ff_raw.stdin is not None
            for frame in ping_pong_frames:
                ff_raw.stdin.write(frame.tobytes())
            ff_raw.stdin.close()
            ff_raw.wait(timeout=45)

            target_dur = max(1.5, float(duration_sec))
            vf_chain = (
                f"scale={self.WIDTH}:{self.HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={self.WIDTH}:{self.HEIGHT},"
                f"fps={self.FPS},"
                f"eq=contrast=1.04:saturation=0.98,"
                f"unsharp=5:5:0.65:5:5:0.0,"
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
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
            raw_mp4.unlink(missing_ok=True)
            if proc.returncode == 0 and dest_mp4.exists() and dest_mp4.stat().st_size > 15_000:
                return dest_mp4
        except Exception as exc:
            raw_mp4.unlink(missing_ok=True)
            print(f"[ComfyUI-Direct] Frame encode notice: {exc}", flush=True)
        return None

    def _animate_via_direct_nodes(
        self,
        scene_id: int,
        keyframe_path: Path,
        motion_prompt: str,
        duration_sec: float,
        dest_mp4: Path,
    ) -> Optional[Path]:
        # Skip close-up anatomy / hands / writing / globe / portrait scenes so they stay razor-sharp 1080x1920 MAI keyframes!
        if not self._scene_suitability.get(scene_id, self.is_scene_suitable_for_i2v(motion_prompt)):
            print(
                f"[ComfyUI-Direct] Scene {scene_id:02d}: Close-up character/hand/instrument shot detected — "
                f"preserving native 1080x1920 MAI-Image-2.6-Flash keyframe with HyperFrames GSAP motion.",
                flush=True,
            )
            return None

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

            model_management = self._direct_modules["model_management"]
            comfy_utils = self._direct_modules["comfy_utils"]
            comfy_samplers = self._direct_modules["comfy_samplers"]
            comfy_sample = self._direct_modules["comfy_sample"]
            node_helpers = self._direct_modules["node_helpers"]

            with Image.open(keyframe_path) as pil_im:
                rgb_im = pil_im.convert("RGB")
                arr = np.array(rgb_im).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(arr).unsqueeze(0)

            print(
                f"[ComfyUI-Direct] Scene {scene_id:02d}: Animating wide atmospheric shot {keyframe_path.name} -> "
                f"{self.GEN_WIDTH}x{self.GEN_HEIGHT} x {self.GEN_FRAMES} frames ({self.GEN_STEPS} steps, strength=0.94, cfg=2.2)...",
                flush=True,
            )

            with torch.inference_mode():
                width = self.GEN_WIDTH
                height = self.GEN_HEIGHT
                length = self.GEN_FRAMES
                strength = 0.94  # High adherence to reference keyframe so it never hallucinates blobs

                pixels = comfy_utils.common_upscale(
                    image_tensor.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
                encode_pixels = pixels[:, :, :, :3]
                t_lat = self._direct_vae.encode(encode_pixels)

                latent = torch.zeros(
                    [1, 128, ((length - 1) // 8) + 1, height // 32, width // 32],
                    device=model_management.intermediate_device(),
                )
                latent[:, :, :t_lat.shape[2]] = t_lat

                noise_mask = torch.ones(
                    (1, 1, latent.shape[2], 1, 1),
                    dtype=torch.float32,
                    device=latent.device,
                )
                noise_mask[:, :, :t_lat.shape[2]] = 1.0 - strength

                cond_pos = node_helpers.conditioning_set_values(pos_cond, {"frame_rate": 24.0})
                cond_neg = node_helpers.conditioning_set_values(neg_cond, {"frame_rate": 24.0})

                tokens = math.prod(latent.shape[2:])
                sigmas = torch.linspace(1.0, 0.0, self.GEN_STEPS + 1)
                x1, x2 = 1024, 4096
                mm = (2.05 - 0.95) / (x2 - x1)
                b = 0.95 - mm * x1
                sigma_shift = tokens * mm + b
                sigmas = torch.where(
                    sigmas != 0,
                    math.exp(sigma_shift) / (math.exp(sigma_shift) + (1.0 / sigmas - 1.0)),
                    0.0,
                )
                non_zero_mask = sigmas != 0
                non_zero_sigmas = sigmas[non_zero_mask]
                one_minus_z = 1.0 - non_zero_sigmas
                scale_factor = one_minus_z[-1] / (1.0 - 0.1)
                sigmas[non_zero_mask] = 1.0 - (one_minus_z / scale_factor)

                sampler_obj = comfy_samplers.ksampler("euler")
                seed = random.randint(10_000, 9_999_999_999)
                noise = comfy_sample.prepare_noise(latent, seed)

                t_samp = time.time()
                samples = comfy_sample.sample_custom(
                    self._direct_model,
                    noise,
                    2.2,
                    sampler_obj,
                    sigmas,
                    cond_pos,
                    cond_neg,
                    latent,
                    noise_mask=noise_mask,
                    callback=None,
                    disable_pbar=False,
                    seed=seed,
                )

                t_dec = time.time()
                decoded = self._direct_vae.decode(samples).detach()

            if decoded.ndim == 5:
                decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
            frames_np = (decoded.cpu().float().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            num_f, h_f, w_f, _ = frames_np.shape
            print(
                f"[ComfyUI-Direct] Scene {scene_id:02d}: Sampled in {t_dec - t_samp:.1f}s, "
                f"VAE decoded {num_f} frames ({w_f}x{h_f}) in {time.time() - t_dec:.1f}s!",
                flush=True,
            )

            if not self._passes_consistency_gate(frames_np):
                gc.collect()
                torch.cuda.empty_cache()
                return None

            out_path = self._encode_frames_to_mp4(frames_np, duration_sec, dest_mp4)
            gc.collect()
            torch.cuda.empty_cache()

            if out_path is not None:
                self.clips_rendered += 1
                self.consecutive_failures = 0
                print(
                    f"[ComfyUI-Direct] Scene {scene_id:02d} COMPLETE -> {dest_mp4.name} "
                    f"({dest_mp4.stat().st_size // 1024} KB, {self.clips_rendered}/{self.max_clips} clips) "
                    f"in {time.time() - t_start:.1f}s!",
                    flush=True,
                )
                return out_path
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
        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.ltx_ckpt}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": self.t5_clip, "type": "ltxv"}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": pos_prompt, "clip": ["2", 0]}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_prompt, "clip": ["2", 0]}},
            "5": {"class_type": "LoadImage", "inputs": {"image": uploaded_image_name}},
            "7": {
                "class_type": "LTXVImgToVideo",
                "inputs": {
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "vae": ["1", 2],
                    "image": ["5", 0],
                    "width": 512,
                    "height": 896,
                    "length": num_frames,
                    "batch_size": 1,
                    "strength": 0.94,
                },
            },
            "8": {"class_type": "LTXVConditioning", "inputs": {"positive": ["7", 0], "negative": ["7", 1], "frame_rate": 24.0}},
            "9": {"class_type": "LTXVScheduler", "inputs": {"steps": 20, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1, "latent": ["7", 2]}},
            "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "11": {
                "class_type": "SamplerCustom",
                "inputs": {
                    "model": ["1", 0],
                    "add_noise": True,
                    "noise_seed": seed,
                    "cfg": 2.2,
                    "positive": ["8", 0],
                    "negative": ["8", 1],
                    "sampler": ["10", 0],
                    "sigmas": ["9", 0],
                    "latent_image": ["7", 2],
                },
            },
            "14": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
            "15": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["14", 0], "filename_prefix": "comfy_ltx_short", "fps": 24.0, "lossless": False, "quality": 92, "method": "default"}},
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

        if self._can_use_direct_engine():
            return self._animate_via_direct_nodes(
                scene_id=scene_id,
                keyframe_path=keyframe_path,
                motion_prompt=motion_prompt,
                duration_sec=duration_sec,
                dest_mp4=dest_mp4,
            )

        return None
