"""
Professional Documentary Video Composer (1080x1920 @ 30fps vertical 9:16).
Key Upgrades:
- Frosted Cinematic Background Canvas: Landscape or square archival photographs are never awkwardly
  cropped into unrecognizable blobs (like a giant patch or cut-off soldiers); instead, images with
  aspect ratios wider than 0.68 are framed over a dark blurred atmospheric 9:16 background with soft
  vignette, while vertical/near-vertical plates fill the 9:16 frame with smooth Ken Burns motion.
- Bottom & Top Gradient Readability Vignette: Ensures captions and callout badges are 100% legible
  against any background while staying strictly within the safe boundary.
- Mysterious Background Music with Automatic Voiceover Ducking (`sidechaincompress`):
  Plays ONLY the dark mysterious documentary score (`mystery_bgm.wav`) ducked smoothly under the narrator,
  with ZERO cartoon transition sound effects (no zoop, whoosh, pop, or shutter).
"""
import os
import subprocess
from pathlib import Path
from typing import List
from PIL import Image, ImageEnhance, ImageFilter


class VideoComposer:
    WIDTH = 1080
    HEIGHT = 1920
    OVERSAMPLE_W = 1260
    OVERSAMPLE_H = 2240
    FPS = 30

    def __init__(self):
        try:
            self.bg_music_vol = float(os.getenv("BG_MUSIC_VOLUME", "0.28"))
        except ValueError:
            self.bg_music_vol = 0.28

    @staticmethod
    def _run_ffmpeg(cmd: List[str], step_name: str):
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed during {step_name} (exit {proc.returncode}):\n{proc.stderr[-800:]}")

    @classmethod
    def _prepare_vertical_archival_plate(cls, asset_path: Path, out_plate_path: Path) -> Path:
        """
        Pre-composes non-vertical archival photos (landscape battle photos, square unit patches, portraits)
        onto a 1260x2240 vertical canvas with a dark frosted atmospheric background so `zoompan` never
        chops off heads, tanks, or historical subjects.
        """
        with Image.open(asset_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            aspect = w / max(h, 1)

            target_w, target_h = cls.OVERSAMPLE_W, cls.OVERSAMPLE_H

            if aspect <= 0.66:
                scale = max(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
                left = (new_w - target_w) // 2
                top = (new_h - target_h) // 2
                plate = resized.crop((left, top, left + target_w, top + target_h))
            else:
                bg_scale = max(target_w / w, target_h / h)
                bg_w, bg_h = int(w * bg_scale), int(h * bg_scale)
                bg = rgb.resize((bg_w, bg_h), Image.Resampling.BICUBIC)
                bg_left = (bg_w - target_w) // 2
                bg_top = (bg_h - target_h) // 2
                bg = bg.crop((bg_left, bg_top, bg_left + target_w, bg_top + target_h))
                bg = bg.filter(ImageFilter.GaussianBlur(radius=36))
                bg = ImageEnhance.Brightness(bg).enhance(0.32)
                bg = ImageEnhance.Color(bg).enhance(0.65)

                fg_max_w = int(target_w * 0.94)
                fg_max_h = int(target_h * 0.62)
                fg_scale = min(fg_max_w / w, fg_max_h / h)
                fg_w, fg_h = int(w * fg_scale), int(h * fg_scale)
                fg = rgb.resize((fg_w, fg_h), Image.Resampling.LANCZOS)
                fg = ImageEnhance.Contrast(fg).enhance(1.06)

                paste_x = (target_w - fg_w) // 2
                paste_y = int((target_h - fg_h) * 0.43)
                bg.paste(fg, (paste_x, paste_y))
                plate = bg

            plate_rgba = plate.convert("RGBA")
            grad = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
            grad_px = grad.load()
            for y in range(target_h):
                alpha = 0
                if y < int(target_h * 0.14):
                    alpha = int(115 * (1.0 - y / (target_h * 0.14)))
                elif y > int(target_h * 0.70):
                    prog = (y - target_h * 0.70) / (target_h * 0.30)
                    alpha = int(175 * (prog ** 1.3))
                if alpha > 0:
                    for x in range(target_w):
                        grad_px[x, y] = (8, 8, 10, alpha)

            composited = Image.alpha_composite(plate_rgba, grad).convert("RGB")
            composited.save(out_plate_path, quality=94)
            return out_plate_path

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

        for idx, scene in enumerate(scenes):
            scene_id = scene.get("scene_id", idx + 1)
            dur = max(0.8, float(scene["duration_sec"]))
            asset_path = Path(scene["asset_path"])
            asset_type = scene["asset_type"]
            effect = scene.get("effect", "blur_snap_zoom_in")

            if asset_type == "image":
                plate_path = temp_dir / f"plate_{scene_id:02d}.jpg"
                asset_path = self._prepare_vertical_archival_plate(asset_path, plate_path)

            if dur > 3.8:
                half1 = round(dur * 0.52, 3)
                half2 = round(dur - half1, 3)
                seg1 = temp_dir / f"seg_{scene_id:02d}_a.mp4"
                seg2 = temp_dir / f"seg_{scene_id:02d}_b.mp4"
                self._render_scene_segment(asset_path, asset_type, half1, effect, seg1, cam_mode=idx % 4, tight_crop=False)
                self._render_scene_segment(asset_path, asset_type, half2, "fast_punch_in", seg2, cam_mode=(idx + 1) % 4, tight_crop=True)
                segment_files.extend([seg1, seg2])
            else:
                seg = temp_dir / f"seg_{scene_id:02d}.mp4"
                self._render_scene_segment(asset_path, asset_type, dur, effect, seg, cam_mode=idx % 4, tight_crop=False)
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
        self._build_master_audio_mix(master_vo, mixed_audio_wav)

        ass_escaped = (
            str(ass_path.resolve())
            .replace("\\", "/")
            .replace(":", "\\:")
            .replace("=", "\\=")
        )
        final_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_visual_mp4),
            "-i",
            str(mixed_audio_wav),
            "-vf",
            f"vignette=PI/6,ass=filename='{ass_escaped}'",
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
        print(f"[VideoComposer] Rendered final 1080x1920 Short -> {output_mp4} ({output_mp4.stat().st_size // 1024} KB)")
        return output_mp4

    def _render_scene_segment(
        self,
        asset_path: Path,
        asset_type: str,
        duration: float,
        effect: str,
        out_path: Path,
        cam_mode: int = 0,
        tight_crop: bool = False,
    ):
        frames = max(int(round(duration * self.FPS)), 20)

        if tight_crop or effect == "fast_punch_in":
            z_expr = f"min(1.12+0.12*(on/{frames}),1.24)"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif effect == "high_contrast_zoom_out" or cam_mode == 1:
            z_expr = f"max(1.14-0.12*(on/{frames}),1.02)"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif cam_mode == 2:
            z_expr = "1.10"
            x_expr = f"(iw-iw/zoom)*(on/{frames})"
            y_expr = "ih/2-(ih/zoom/2)"
        elif cam_mode == 3:
            z_expr = "1.10"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = f"(ih-ih/zoom)*(on/{frames})"
        else:
            z_expr = f"min(1.01+0.12*(on/{frames}),1.13)"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"

        color_grade = "eq=contrast=1.06:saturation=0.94:brightness=-0.01"
        if effect == "vintage_sepia_zoom":
            color_grade = "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,eq=contrast=1.10"

        fade_out_st = max(0.10, duration - 0.14)

        if asset_type == "image":
            vf = (
                f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
                f"d={frames}:s={self.WIDTH}x{self.HEIGHT}:fps={self.FPS},"
                f"{color_grade},"
                f"gblur=sigma=8:enable='lt(t,0.10)',"
                f"fade=t=in:st=0:d=0.10,fade=t=out:st={fade_out_st:.2f}:d=0.12,"
                f"setsar=1,format=yuv420p"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(asset_path),
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
            scale_w = 1380 if tight_crop else 1140
            scale_h = 2450 if tight_crop else 2020
            vf = (
                f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase,"
                f"crop={self.WIDTH}:{self.HEIGHT},"
                f"fps={self.FPS},"
                f"{color_grade},"
                f"gblur=sigma=8:enable='lt(t,0.10)',"
                f"fade=t=in:st=0:d=0.10,fade=t=out:st={fade_out_st:.2f}:d=0.12,"
                f"setsar=1,format=yuv420p"
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
                "-vf",
                vf,
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

    def _build_master_audio_mix(
        self,
        master_vo: Path,
        out_wav: Path,
    ):
        """
        Mixes the voiceover with the dark mysterious documentary background music (`mystery_bgm.wav`)
        using smooth sidechain compression (auto-ducking) so the mystery music swells during pauses
        and sits cleanly underneath the narrator — with ZERO whoosh/zoop/pop sound effects.
        """
        sfx_dir = Path("assets/sfx")
        mystery_bgm = sfx_dir / "mystery_bgm.wav"
        if not mystery_bgm.exists():
            mystery_bgm = sfx_dir / "ambient_bed.wav"

        filter_complex = (
            "[0:a]volume=1.05,asplit=2[vo_main][vo_sc];"
            f"[1:a]volume={self.bg_music_vol:.2f}[bgm_raw];"
            "[bgm_raw][vo_sc]sidechaincompress=threshold=0.03:ratio=4:attack=35:release=320[bgm_ducked];"
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
            str(mystery_bgm),
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
        self._run_ffmpeg(cmd, "master audio mix with mysterious BGM ducking")
