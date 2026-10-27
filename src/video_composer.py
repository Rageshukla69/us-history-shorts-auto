"""
Professional Documentary Video Composer (v4.0 — ShortGPT & MoneyPrinterTurbo Inspired).
Key Upgrades:
1. Full-Bleed Vertical Framing (`_prepare_vertical_archival_plate`):
   - Portrait and near-square photos (`aspect <= 1.18`) fill the entire 9:16 vertical screen
     (smartly biased toward the upper-third so faces and rockets are never cut off and there are
     NO ugly black boxes around portrait photos!).
   - Wide landscape archival diagrams/photos (`aspect > 1.18`) are scaled generously (`96%` width)
     over a vibrant softly blurred glass backdrop with a subtle 3px warm cinema frame.
2. Multi-Source Background Music Selector (`_select_bgm_track`):
   - Just like `MoneyPrinterTurbo`, if you drop any `.mp3`, `.wav`, or `.ogg` track into `assets/bgm/`
     (or set `BGM_URL` in GitHub Secrets), it automatically uses your custom track!
   - Otherwise uses the built-in uplifting Hans Zimmer "Interstellar" organ & piano arpeggio score
     (`assets/sfx/interstellar_bgm.wav`) with smooth sidechain voiceover ducking.
"""
import os
import random
import subprocess
from pathlib import Path
from typing import List
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


class VideoComposer:
    WIDTH = 1080
    HEIGHT = 1920
    OVERSAMPLE_W = 1260
    OVERSAMPLE_H = 2240
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
    def _prepare_vertical_archival_plate(cls, asset_path: Path, out_plate_path: Path) -> Path:
        """
        Prepares a 1260x2240 vertical 9:16 plate:
        - Portrait and near-square images (`aspect <= 1.18`) fill the entire 9:16 frame edge-to-edge
          with upper-center bias so no ugly letterbox/pillarbox bars appear.
        - Wide horizontal images (`aspect > 1.18`) sit over a softly blurred atmospheric backdrop
          with a crisp 3px archival border.
        """
        with Image.open(asset_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            aspect = w / max(h, 1)

            target_w, target_h = cls.OVERSAMPLE_W, cls.OVERSAMPLE_H

            if aspect <= 1.18:
                # Full-bleed 9:16 cover (upper-center biased so heads/rockets stay framed)
                scale = max(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
                left = (new_w - target_w) // 2
                top = int((new_h - target_h) * 0.28)
                plate = resized.crop((left, top, left + target_w, top + target_h))
            else:
                # Wide landscape photo or diagram: rich blurred glass backdrop + large framed card
                bg_scale = max(target_w / w, target_h / h)
                bg_w, bg_h = int(w * bg_scale), int(h * bg_scale)
                bg = rgb.resize((bg_w, bg_h), Image.Resampling.BICUBIC)
                bg_left = (bg_w - target_w) // 2
                bg_top = (bg_h - target_h) // 2
                bg = bg.crop((bg_left, bg_top, bg_left + target_w, bg_top + target_h))
                bg = bg.filter(ImageFilter.GaussianBlur(radius=28))
                bg = ImageEnhance.Brightness(bg).enhance(0.45)
                bg = ImageEnhance.Color(bg).enhance(0.80)

                fg_max_w = int(target_w * 0.96)
                fg_max_h = int(target_h * 0.68)
                fg_scale = min(fg_max_w / w, fg_max_h / h)
                fg_w, fg_h = int(w * fg_scale), int(h * fg_scale)
                fg = rgb.resize((fg_w, fg_h), Image.Resampling.LANCZOS)
                fg = ImageEnhance.Contrast(fg).enhance(1.06)

                paste_x = (target_w - fg_w) // 2
                paste_y = int((target_h - fg_h) * 0.42)
                bg.paste(fg, (paste_x, paste_y))

                # Subtle warm archival border around wide landscape plate
                draw = ImageDraw.Draw(bg)
                draw.rectangle(
                    [paste_x - 2, paste_y - 2, paste_x + fg_w + 2, paste_y + fg_h + 2],
                    outline=(220, 200, 150),
                    width=3,
                )
                plate = bg

            # Smooth bottom & top gradient vignette so captions and badges pop cleanly
            plate_rgba = plate.convert("RGBA")
            grad = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
            grad_px = grad.load()
            for y in range(target_h):
                alpha = 0
                if y < int(target_h * 0.13):
                    alpha = int(105 * (1.0 - y / (target_h * 0.13)))
                elif y > int(target_h * 0.72):
                    prog = (y - target_h * 0.72) / (target_h * 0.28)
                    alpha = int(170 * (prog ** 1.3))
                if alpha > 0:
                    for x in range(target_w):
                        grad_px[x, y] = (8, 8, 12, alpha)

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
        self._build_master_audio_mix(master_vo, mixed_audio_wav, temp_dir)

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
            z_expr = f"min(1.10+0.12*(on/{frames}),1.22)"
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

        color_grade = "eq=contrast=1.06:saturation=0.96:brightness=-0.01"
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

    def _select_bgm_track(self, temp_dir: Path) -> Path:
        """
        MoneyPrinterTurbo-style BGM selector:
        1. If `BGM_URL` environment variable is set, downloads that audio track.
        2. If user placed any `.mp3`, `.wav`, or `.ogg` file inside `assets/bgm/`, randomly picks one.
        3. Otherwise uses the synthesized uplifting Interstellar organ & piano arpeggio score
           (`assets/sfx/interstellar_bgm.wav`).
        """
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

        filter_complex = (
            "[0:a]volume=1.05,asplit=2[vo_main][vo_sc];"
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
        self._run_ffmpeg(cmd, "master audio mix with Interstellar BGM ducking")
