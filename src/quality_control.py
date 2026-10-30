"""Technical and documentary-source QC for vertical Shorts."""
import json, os, subprocess
from pathlib import Path


def _run(args, timeout=90):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
    if p.returncode: raise RuntimeError(p.stderr[-1600:])
    return p.stdout.strip()


def validate_rendered_short(video_path: Path, scenes: list, report_path: Path) -> dict:
    if not video_path.exists() or video_path.stat().st_size < 250_000:
        raise RuntimeError("QC: rendered MP4 is missing or implausibly small.")
    raw=_run(["ffprobe","-v","error","-show_entries","format=duration:stream=codec_type,width,height,avg_frame_rate","-of","json",str(video_path)])
    data=json.loads(raw); duration=float(data.get("format",{}).get("duration",0))
    v=next((x for x in data.get("streams",[]) if x.get("codec_type")=="video"),{})
    a=next((x for x in data.get("streams",[]) if x.get("codec_type")=="audio"),None)
    if (v.get("width"),v.get("height")) != (1080,1920): raise RuntimeError(f"QC: expected 1080x1920, got {v.get('width')}x{v.get('height')}.")
    if not a: raise RuntimeError("QC: no audio stream.")
    if not 15 <= duration <= 65: raise RuntimeError(f"QC: duration {duration:.1f}s is outside 15-65s.")
    if not scenes: raise RuntimeError("QC: no scenes were rendered.")
    # Authentic footage and rights-cleared archival stills count as source media. AI is labelled separately.
    kinds=[str(s.get("asset_kind", "")).upper() for s in scenes]
    authentic=sum(1 for k in kinds if "AUTHENTIC VIDEO" in k or "ARCHIVAL IMAGE" in k)
    ai=sum(1 for k in kinds if "AI RECONSTRUCTION" in k)
    graphics=sum(1 for k in kinds if "EDITORIAL GRAPHIC" in k)
    source_ratio=authentic/max(1,len(scenes))
    dry_run=os.getenv("DRY_RUN","true").strip().lower()=="true"
    min_ratio=float(os.getenv("MIN_AUTHENTIC_MEDIA_RATIO","0.30"))
    upload_eligible=source_ratio >= min_ratio
    # Detect sustained black frames. Do not confuse intentionally dark imagery with a fully black scene.
    black=subprocess.run(["ffmpeg","-hide_banner","-i",str(video_path),"-vf","blackdetect=d=1.2:pix_th=0.04:pic_th=0.98","-an","-f","null","-"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120)
    if black.returncode not in (0,1): raise RuntimeError("QC: FFmpeg black-frame analysis failed.")
    if "black_start:" in black.stderr: raise RuntimeError("QC: sustained black frames detected.")
    report={"passed":True,"duration_seconds":round(duration,2),"width":1080,"height":1920,"has_audio":True,
            "scene_count":len(scenes),"authentic_source_scene_count":authentic,"authentic_source_ratio":round(source_ratio,3),
            "ai_reconstruction_scene_count":ai,"editorial_graphic_scene_count":graphics,
            "minimum_authentic_media_ratio":min_ratio,"upload_eligible":upload_eligible,
            "dry_run":dry_run,"notes":[]}
    if not upload_eligible:
        report["notes"].append(f"Only {authentic}/{len(scenes)} scenes use rights-cleared archival media. Preview is allowed; automatic YouTube upload is blocked until source coverage reaches {min_ratio:.0%}.")
        if not dry_run:
            report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
            raise RuntimeError(report["notes"][0])
    else:
        report["notes"].append("Technical checks passed and archival-source coverage meets the configured threshold. Human review remains recommended.")
    report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
