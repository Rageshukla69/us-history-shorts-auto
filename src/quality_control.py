"""Pre-upload quality gate for rendered Shorts. Uses ffprobe/ffmpeg already installed in Actions."""
import json, subprocess
from pathlib import Path

def _run(args):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
    if p.returncode: raise RuntimeError(p.stderr[-1200:])
    return p.stdout.strip()

def validate_rendered_short(video_path: Path, scenes: list, report_path: Path) -> dict:
    if not video_path.exists() or video_path.stat().st_size < 250_000:
        raise RuntimeError("Quality gate: rendered MP4 is missing or implausibly small.")
    raw=_run(["ffprobe","-v","error","-show_entries","format=duration:stream=codec_type,width,height","-of","json",str(video_path)])
    data=json.loads(raw); duration=float(data.get("format",{}).get("duration",0))
    v=next((x for x in data.get("streams",[]) if x.get("codec_type")=="video"),{})
    a=next((x for x in data.get("streams",[]) if x.get("codec_type")=="audio"),None)
    if (v.get("width"),v.get("height")) != (1080,1920): raise RuntimeError(f"Quality gate: expected 1080x1920, got {v.get('width')}x{v.get('height')}.")
    if not a: raise RuntimeError("Quality gate: no audio stream.")
    if not 15 <= duration <= 65: raise RuntimeError(f"Quality gate: duration {duration:.1f}s is outside 15-65s.")
    kinds=[str(s.get("asset_type","")) for s in scenes]
    fallbacks=sum(1 for s in scenes if "EDITORIAL GRAPHIC" in str(s.get("asset_kind","")) or s.get("asset_type")=="editorial_card")
    if scenes and fallbacks/len(scenes) > 0.30:
        raise RuntimeError(f"Quality gate: {fallbacks}/{len(scenes)} scenes have no matching source media (>30%); refusing upload.")
    # Detect long black frames, a known failure mode in the previous output.
    black_proc=subprocess.run(["ffmpeg","-hide_banner","-i",str(video_path),"-vf","blackdetect=d=1.0:pix_th=0.10:pic_th=0.98","-an","-f","null","-"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=90)
    if black_proc.returncode not in (0, 1): raise RuntimeError("Quality gate: FFmpeg black-frame analysis failed.")
    if "black_start:" in black_proc.stderr:
        raise RuntimeError("Quality gate: sustained black frames detected; refusing upload.")
    report={"passed":True,"duration_seconds":round(duration,2),"width":1080,"height":1920,"has_audio":True,"scene_count":len(scenes),"fallback_scene_count":fallbacks,"notes":["Automated technical checks passed; a human visual review is still required."]}
    report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
