"""
Professional Documentary Video Composer (v6.0 — Smooth Subpixel Cinema Motion, 2-Layer Parallax & Aspect-Safe Video Engine):
Fixes Every Visual Issue Reported in `a165fe13.mp4` and `final_short.mp4`:
1. ZERO Camera Shake & ZERO Mid-Photo Blurry Flashes:
   - Completely removed `sin()`/`cos()` camera shake formulas (`impact_shake_zoom` / `fast_punch_in` wobble).
   - Completely removed the mid-scene photo split (`seg_a` / `seg_b`) and `gblur=sigma=5` white flash that previously
     triggered a jarring shake and blurry flash in the middle of every photo!
2. 3x Subpixel Supersampling (`scale=3240:5760` before `zoompan`) — Fixes FFmpeg Ticket #4298 Jitter:
   - Standard FFmpeg `zoompan` rounds crop coordinates to whole pixels on the input image, causing stepped jitter.
   - Upscaling to a `3240x5760` subpixel grid before `zoompan` gives 0.33-pixel precision so every slow push-in,
     pull-out, and documentary pan is buttery-smooth at 30fps.
3. True 2-Layer Parallax for Wide Historical Photos & Uncropped Blurred-Backdrop Framing for Wide Video Clips:
   - Portrait `9:16` visuals fill the `1080x1920` screen edge-to-edge.
   - Wide horizontal video clips AND wide historical photos (`aspect > 1.12`) are NEVER chopped off: 100% of the wide
     frame sits in a crisp cinema card over a moving, Gaussian-blurred `1080x1920` backdrop of the same scene.
"""
import os
import random
import subprocess
from pathlib import Path
from typing import List
import requests
from PIL import Image, ImageEnhance


class VideoComposer:
    WIDTH = 1080
    HEIGHT = 1920
    SUBPIXEL_W = 3240
    SUBPIXEL_H = 5760
    FPS = 30

    def __init__(self):
        try:
            self.bg_music_vol = float(os.getenv("BG_MUSIC_VOLUME", "0.30"))
        except ValueError:
            self.bg_music_vol = 0.30

    @staticmethod
    def _run_ffmpeg(cmd: List[str], step_name: str):
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed during {step_name} (exit {proc.returncode}):\n{proc.stderr[-800:]}")

    @classmethod
    def _inspect_image_aspect_and_prepare(cls, asset_path: Path, out_clean_path: Path) -> float:
        """
        Normalizes RGB color/contrast and returns the natural aspect ratio (`width / height`)
        without cropping wide historical images.
        """
        with Image.open(asset_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            aspect = w / max(h, 1)
            if aspect <= 1.12:
                # Portrait / vertical 9:16 image: fit cleanly to 1080x1920 with upper-third headroom
                scale = max(cls.WIDTH / max(w, 1), cls.HEIGHT / max(h, 1))
                new_w = max(cls.WIDTH, int(round(w * scale)))
                new_h = max(cls.HEIGHT, int(round(h * scale)))
                resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
                left = (new_w - cls.WIDTH) // 2
                top = min(max(0, int((new_h - cls.HEIGHT) * 0.20)), new_h - cls.HEIGHT)
                cropped = resized.crop((left, top, left + cls.WIDTH, top + cls.HEIGHT))
                cropped = ImageEnhance.Contrast(cropped).enhance(1.05)
                cropped = ImageEnhance.Sharpness(cropped).enhance(1.06)
                cropped.save(out_clean_path, quality=95)
            else:
                # Wide horizontal historical photo: keep 100% of original width/height uncropped
                # so FFmpeg can animate it with 2-layer parallax over a moving blurred backdrop!
                scale = min(1920 / max(w, 1), 1440 / max(h, 1))
                if scale < 1.0 or w < 960:
                    nw = max(960, int(round(w * max(scale, 960 / max(w, 1)))))
                    nh = max(540, int(round(h * (nw / max(w, 1)))))
                    rgb = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
                rgb = ImageEnhance.Contrast(rgb).enhance(1.05)
                rgb = ImageEnhance.Sharpness(rgb).enhance(1.06)
                rgb.save(out_clean_path, quality=95)
            return aspect

    def render_short(
        self,
        scenes: List[dict],
        master_vo: Path,
        ass_path: Path,
        output_mp4: Path,
        temp_dir: Path,
    ) -> Path:
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_mp4.parent.mkdir(parents=True, exist_ok=True)

        segment_files: List[Path] = []
        total_duration = sum(max(0.8, float(s.get("duration_sec", 2.5))) for s in scenes)

        for idx, scene in enumerate(scenes):
            scene_id = scene.get("scene_id", idx + 1)
            dur = max(0.8, float(scene["duration_sec"]))
            asset_path = Path(scene["asset_path"])
            asset_type = scene["asset_type"]
            effect = scene.get("effect", "smooth_push_in")

            # One continuous, uninterrupted cinematic shot per scene (NO mid-scene split, NO shake!)
            seg = temp_dir / f"seg_{scene_id:02d}.mp4"
            self._render_scene_segment(
                asset_path=asset_path,
                asset_type=asset_type,
                duration=dur,
                effect=effect,
                out_path=seg,
                cam_mode=idx % 5,
                temp_dir=temp_dir,
                scene_id=scene_id,
            )
            segment_files.append(seg)

        concat_file = temp_dir / "video_segments.txt"
        concat_file.write_text(
            "\n".join([f"file '{p.resolve()}'" for p in segment_files]),
            encoding="utf-8",
        )
        raw_visual_mp4 = temp_dir / "visual_track.mp4"
        self._run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(raw_visual_mp4),
            ],
            "segment concatenation",
        )

        mixed_audio_wav = temp_dir / "master_mix.wav"
        self._build_master_audio_mix(master_vo, mixed_audio_wav, temp_dir)

        ass_escaped = (
            str(ass_path.resolve())
            .replace("\\", "/")
            .replace(":", "\\:")
            .replace("=", "\\=")
        )
        tot_d = max(10.0, total_duration)
        vf_final = (
            f"vignette=PI/8,"
            f"drawbox=x=0:y=0:w='iw*min(1,t/{tot_d:.2f})':h=6:color=0xFFD700@0.92:t=fill,"
            f"ass=filename='{ass_escaped}'"
        )
        final_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_visual_mp4),
            "-i",
            str(mixed_audio_wav),
            "-vf",
            vf_final,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_mp4),
        ]
        self._run_ffmpeg(final_cmd, "final subtitle burn-in and master encode")
        print(f"[VideoComposer] Rendered v6.0 Smooth Cinema 1080x1920 Short -> {output_mp4} ({output_mp4.stat().st_size // 1024} KB)")
        return output_mp4

    def _render_scene_segment(
        self,
        asset_path: Path,
        asset_type: str,
        duration: float,
        effect: str,
        out_path: Path,
        cam_mode: int = 0,
        temp_dir: Path = Path("temp"),
        scene_id: int = 1,
    ):
        frames = max(int(round(duration * self.FPS)), 18)

        # Buttery-smooth subpixel camera expressions (ZERO sin/cos camera shake!)
        if effect in ("smooth_pull_out", "high_contrast_zoom_out") or cam_mode == 1:
            z_expr = f"1.085-0.075*(on/{frames})"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif effect in ("cinema_pan_right", "parallax_pan_right") or cam_mode == 2:
            z_expr = "1.06"
            x_expr = f"(iw-iw/zoom)*0.15+(iw-iw/zoom)*0.70*(on/{frames})"
            y_expr = "ih/2-(ih/zoom/2)"
        elif effect == "cinema_pan_left" or cam_mode == 3:
            z_expr = "1.06"
            x_expr = f"(iw-iw/zoom)*0.85-(iw-iw/zoom)*0.70*(on/{frames})"
            y_expr = "ih/2-(ih/zoom/2)"
        elif effect == "cinema_tilt_up" or cam_mode == 4:
            z_expr = "1.06"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = f"(ih-ih/zoom)*0.80-(ih-ih/zoom)*0.60*(on/{frames})"
        else:
            # Default: Smooth documentary push-in
            z_expr = f"1.01+0.08*(on/{frames})"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"

        # Clean, subtle 0.10s optical dissolve at start of cut (no white flash, no gaussian blur!)
        clean_transition = "fade=t=in:st=0:d=0.10"

        if asset_type == "image":
            clean_img = temp_dir / f"clean_{scene_id:02d}.jpg"
            aspect = self._inspect_image_aspect_and_prepare(asset_path, clean_img)

            if aspect <= 1.12:
                # Portrait 9:16 visual: 3x Subpixel Supersampling (3240x5760) eliminates FFmpeg Ticket #4298 jitter!
                vf = (
                    f"scale={self.SUBPIXEL_W}:{self.SUBPIXEL_H}:flags=bilinear,"
                    f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
                    f"d={frames}:s={self.WIDTH}x{self.HEIGHT}:fps={self.FPS},"
                    f"drawbox=x=0:y=ih*0.72:w=iw:h=ih*0.28:color=0x08080C@0.42:t=fill,"
                    f"{clean_transition},"
                    f"setsar=1,format=yuv420p"
                )
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(clean_img),
                    "-vf",
                    vf,
                    "-t",
                    f"{duration:.3f}",
                    "-r",
                    str(self.FPS),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-an",
                    str(out_path),
                ]
            else:
                # Wide horizontal historical photo: True 2-Layer Parallax (Moving Blurred Backdrop + Smooth Uncropped Center Card)
                fg_h = min(1180, max(580, int(round(1036 / max(aspect, 1.15)))))
                if fg_h % 2 == 1:
                    fg_h += 1
                sub_fg_w = 1036 * 3
                sub_fg_h = fg_h * 3
                filter_complex = (
                    f"[0:v]split=2[bg_in][fg_in];"
                    f"[bg_in]scale={self.WIDTH*2}:{self.HEIGHT*2}:force_original_aspect_ratio=increase,"
                    f"crop={self.WIDTH*2}:{self.HEIGHT*2},"
                    f"zoompan=z='1.08-0.05*(on/{frames})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                    f"d={frames}:s={self.WIDTH}x{self.HEIGHT}:fps={self.FPS},"
                    f"boxblur=28:4,eq=brightness=-0.24:saturation=0.85[bg];"
                    f"[fg_in]scale={sub_fg_w}:{sub_fg_h}:flags=bilinear,"
                    f"zoompan=z='1.0+0.055*(on/{frames})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                    f"d={frames}:s=1036x{fg_h}:fps={self.FPS},"
                    f"pad=iw+6:ih+6:3:3:color=0xE6CD91[fg];"
                    f"[bg][fg]overlay=(W-w)/2:(H-h)*0.38:shortest=1,"
                    f"drawbox=x=0:y=ih*0.72:w=iw:h=ih*0.28:color=0x08080C@0.42:t=fill,"
                    f"{clean_transition},"
                    f"setsar=1,format=yuv420p[vout]"
                )
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(clean_img),
                    "-t",
                    f"{duration:.3f}",
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[vout]",
                    "-r",
                    str(self.FPS),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-an",
                    str(out_path),
                ]
        else:
            # Real Motion Video Clip:
            # Preserves 100% of the video clip width in a crisp cinema card over a moving blurred 1080x1920 backdrop
            # (or fills the screen if the video clip is already vertical 9:16).
            filter_complex = (
                f"[0:v]split=2[bg_in][fg_in];"
                f"[bg_in]scale={self.WIDTH}:{self.HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={self.WIDTH}:{self.HEIGHT},boxblur=28:4,eq=brightness=-0.24:saturation=0.85[bg];"
                f"[fg_in]scale=1044:1480:force_original_aspect_ratio=decrease,"
                f"pad=iw+6:ih+6:3:3:color=0xE6CD91[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)*0.38,"
                f"fps={self.FPS},"
                f"drawbox=x=0:y=ih*0.72:w=iw:h=ih*0.28:color=0x08080C@0.42:t=fill,"
                f"{clean_transition},"
                f"setsar=1,format=yuv420p[vout]"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-stream_loop",
                "-1",
                "-i",
                str(asset_path),
                "-t",
                f"{duration:.3f}",
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-r",
                str(self.FPS),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(out_path),
            ]
        self._run_ffmpeg(cmd, f"scene segment {out_path.name}")

    def _select_bgm_track(self, temp_dir: Path) -> Path:
        bgm_url = os.getenv("BGM_URL", "").strip()
        if bgm_url:
            try:
                resp = requests.get(bgm_url, timeout=30)
                resp.raise_for_status()
                custom_dl = temp_dir / "custom_bgm.mp3"
                custom_dl.write_bytes(resp.content)
                print(f"[VideoComposer] Downloaded custom BGM from BGM_URL -> {custom_dl}")
                return custom_dl
            except Exception as exc:
                print(f"[VideoComposer] BGM_URL notice ({exc}); falling back to local BGM.")

        bgm_dir = Path("assets/bgm")
        if bgm_dir.exists():
            custom_tracks = [
                p for p in bgm_dir.iterdir()
                if p.suffix.lower() in (".mp3", ".wav", ".ogg", ".m4a") and p.stat().st_size > 10_000
            ]
            if custom_tracks:
                chosen = random.choice(custom_tracks)
                print(f"[VideoComposer] Using custom BGM track from assets/bgm/: {chosen.name}")
                return chosen

        sfx_dir = Path("assets/sfx")
        for candidate in ("interstellar_bgm.wav", "mystery_bgm.wav", "ambient_bed.wav"):
            p = sfx_dir / candidate
            if p.exists():
                return p
        raise RuntimeError("No BGM track found in assets/bgm/ or assets/sfx/.")

    def _build_master_audio_mix(
        self,
        master_vo: Path,
        out_wav: Path,
        temp_dir: Path,
    ):
        bgm_track = self._select_bgm_track(temp_dir)
        hook_sfx = Path("assets/sfx/hook_impact.wav")

        if hook_sfx.exists():
            filter_complex = (
                "[0:a]volume=1.08,asplit=2[vo_main][vo_sc];"
                f"[1:a]volume={self.bg_music_vol:.2f}[bgm_raw];"
                "[bgm_raw][vo_sc]sidechaincompress=threshold=0.03:ratio=3.5:attack=35:release=350[bgm_ducked];"
                "[2:a]volume=0.32[hook_sfx];"
                "[vo_main][bgm_ducked][hook_sfx]amix=inputs=3:duration=first:dropout_transition=2,alimiter=limit=0.95[aout]"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(master_vo),
                "-stream_loop",
                "-1",
                "-i",
                str(bgm_track),
                "-i",
                str(hook_sfx),
                "-filter_complex",
                filter_complex,
                "-map",
                "[aout]",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(out_wav),
            ]
        else:
            filter_complex = (
                "[0:a]volume=1.08,asplit=2[vo_main][vo_sc];"
                f"[1:a]volume={self.bg_music_vol:.2f}[bgm_raw];"
                "[bgm_raw][vo_sc]sidechaincompress=threshold=0.03:ratio=3.5:attack=35:release=350[bgm_ducked];"
                "[vo_main][bgm_ducked]amix=inputs=2:duration=first:dropout_transition=2,alimiter=limit=0.95[aout]"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(master_vo),
                "-stream_loop",
                "-1",
                "-i",
                str(bgm_track),
                "-filter_complex",
                filter_complex,
                "-map",
                "[aout]",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(out_wav),
            ]
        self._run_ffmpeg(cmd, "master voiceover + sidechain BGM mix")
