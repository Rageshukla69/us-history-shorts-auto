"""
Editor V1 Video Composer (1080x1920 @ 30fps vertical 9:16).
Honors your `BG_MUSIC_VOLUME` Heroku Config Var (default `0.25`) for mixing the background bed.
"""
import os
import subprocess
from pathlib import Path
from typing import List


class VideoComposer:
    WIDTH = 1080
    HEIGHT = 1920
    FPS = 30

    def __init__(self):
        try:
            self.bg_music_vol = float(os.getenv("BG_MUSIC_VOLUME", "0.25"))
        except ValueError:
            self.bg_music_vol = 0.25

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
        sfx_cue_points: List[tuple] = []

        for scene in scenes:
            scene_id = scene["scene_id"]
            dur = float(scene["duration_sec"])
            start_sec = float(scene["start_sec"])
            asset_path = Path(scene["asset_path"])
            asset_type = scene["asset_type"]
            effect = scene.get("effect", "blur_snap_zoom_in")
            sfx_name = scene.get("sfx", "whoosh")

            sfx_cue_points.append((start_sec, sfx_name))

            # EDITOR V1 GUARDRAIL: Never allow a single visual shot to stay unchanged >= 3.6 seconds
            if dur > 3.6:
                half1 = round(dur * 0.52, 3)
                half2 = round(dur - half1, 3)
                seg1 = temp_dir / f"seg_{scene_id:02d}_a.mp4"
                seg2 = temp_dir / f"seg_{scene_id:02d}_b.mp4"
                self._render_scene_segment(asset_path, asset_type, half1, effect, seg1, tight_crop=False)
                self._render_scene_segment(asset_path, asset_type, half2, "fast_punch_in", seg2, tight_crop=True)
                segment_files.extend([seg1, seg2])
                sfx_cue_points.append((start_sec + half1, "pop"))
            else:
                seg = temp_dir / f"seg_{scene_id:02d}.mp4"
                self._render_scene_segment(asset_path, asset_type, dur, effect, seg, tight_crop=False)
                segment_files.append(seg)

        concat_file = temp_dir / "video_segments.txt"
        concat_file.write_text(
            "\n".join([f"file '{p.resolve()}'" for p in segment_files]),
            encoding="utf-8",
        )
        raw_visual_mp4 = temp_dir / "visual_track.mp4"
        subprocess.run(
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
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        mixed_audio_wav = temp_dir / "master_mix.wav"
        self._build_master_audio_mix(master_vo, sfx_cue_points, mixed_audio_wav)

        ass_escaped = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_visual_mp4),
            "-i",
            str(mixed_audio_wav),
            "-vf",
            f"vignette=PI/5,ass='{ass_escaped}'",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_mp4),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[VideoComposer] Rendered final 1080x1920 Short -> {output_mp4}")
        return output_mp4

    def _render_scene_segment(
        self,
        asset_path: Path,
        asset_type: str,
        duration: float,
        effect: str,
        out_path: Path,
        tight_crop: bool = False,
    ):
        frames = max(int(round(duration * self.FPS)), 15)

        if tight_crop or effect == "fast_punch_in":
            zoom_expr = "min(1.28+0.0025*on,1.45)"
        elif effect == "high_contrast_zoom_out":
            zoom_expr = "max(1.26-0.0020*on,1.05)"
        else:
            zoom_expr = "min(1.04+0.0018*on,1.22)"

        color_grade = "eq=contrast=1.08:saturation=1.12:brightness=-0.02"
        if effect == "vintage_sepia_zoom":
            color_grade = "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,eq=contrast=1.12"

        if asset_type == "image":
            vf = (
                f"scale=1350:2400:force_original_aspect_ratio=increase,"
                f"crop=1350:2400,"
                f"zoompan=z='{zoom_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={self.WIDTH}x{self.HEIGHT}:fps={self.FPS},"
                f"{color_grade},"
                f"boxblur=luma_radius='if(lt(t,0.16),10*(1-t/0.16),0)':luma_power=1"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
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
                "19",
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
                f"boxblur=luma_radius='if(lt(t,0.16),10*(1-t/0.16),0)':luma_power=1"
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
                "19",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(out_path),
            ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _build_master_audio_mix(
        self,
        master_vo: Path,
        sfx_cue_points: List[tuple],
        out_wav: Path,
    ):
        sfx_dir = Path("assets/sfx")
        ambient_bed = sfx_dir / "ambient_bed.wav"

        inputs = ["-i", str(master_vo), "-i", str(ambient_bed)]
        filter_parts = [
            "[0:a]volume=1.0[vo]",
            f"[1:a]volume={self.bg_music_vol:.2f}[bed]",
        ]
        mix_labels = ["[vo]", "[bed]"]

        input_idx = 2
        for cue_time, sfx_name in sfx_cue_points[:28]:
            sfx_file = sfx_dir / f"{sfx_name}.wav"
            if not sfx_file.exists():
                sfx_file = sfx_dir / "whoosh.wav"
            if not sfx_file.exists():
                continue
            delay_ms = max(int(cue_time * 1000), 0)
            inputs.extend(["-i", str(sfx_file)])
            lbl = f"sfx{input_idx}"
            filter_parts.append(f"[{input_idx}:a]volume=0.28,adelay={delay_ms}|{delay_ms}[{lbl}]")
            mix_labels.append(f"[{lbl}]")
            input_idx += 1

        amix_str = "".join(mix_labels) + f"amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0,alimiter=limit=0.95[aout]"
        filter_complex = ";".join(filter_parts + [amix_str])

        cmd = [
            "ffmpeg",
            "-y",
            *inputs,
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
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
