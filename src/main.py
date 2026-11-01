"""
Main End-to-End Orchestrator for the Automated US History Shorts Channel (v9.0 — HeyGen HyperFrames + Up to 10 MAI-Image-2.6-Flash Cinema Edition).
1. Selects next unused US/World history topic (`src/topic_manager.py`)
2. Generates 8-12 rapid story beats via `gpt-6-luna` (`src/foundry_llm.py`) with up to 10 `MAI-Image-2.6-Flash` shots + archival footage
3. Synthesizes young US English voiceover (`en-US-BrianMultilingualNeural`) with exact millisecond word timestamps (`src/voice_engine.py`)
4. Fetches authentic archival video clips, archival photos, and up to 10 bespoke 9:16 `MAI-Image-2.6-Flash` visuals (`src/media_fetcher.py`)
5. Builds the complete **HeyGen HyperFrames HTML + CSS + GSAP + HyperShader** project (`output/hyperframes_project/index.html`, `preview.html`, `DESIGN.md`)
6. Renders the final `1080x1920` MP4 via `npx hyperframes render` (with automatic hybrid FFmpeg dual-camera fallback so CI never fails)
7. Runs technical & visual Quality Control (`src/quality_control.py`), publishes to YouTube (`src/youtube_uploader.py`), and logs a full audit table to `$GITHUB_STEP_SUMMARY`.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from sfx_generator import generate_all_sfx
from topic_manager import select_next_topic, mark_topic_completed
from foundry_llm import FoundryLLMClient
from voice_engine import VoiceEngine
from media_fetcher import MediaFetcher
from caption_builder import generate_synced_ass_captions
from video_composer import VideoComposer
from hyperframes_composer import HyperFramesComposer
from youtube_uploader import upload_short_to_youtube
from quality_control import validate_rendered_short

HISTORY_FILE = Path("data/upload_history.json")


def _should_skip_backup_cron_slot(min_hours_between_uploads: float = 8.0) -> bool:
    event_name = os.getenv("GITHUB_EVENT_NAME", "").strip()
    if event_name != "schedule":
        return False

    if not HISTORY_FILE.exists():
        return False

    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        uploads = data.get("uploads", [])
        if not uploads:
            return False
        last_ts_raw = uploads[-1].get("timestamp_utc", "")
        if not last_ts_raw:
            return False
        last_dt = datetime.fromisoformat(last_ts_raw)
        elapsed_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600.0
        if elapsed_hours < min_hours_between_uploads:
            print(
                f"[Pipeline] Backup cron slot guard: Last Short was uploaded {elapsed_hours:.2f}h ago "
                f"(< {min_hours_between_uploads}h threshold). Skipping duplicate backup slot cleanly."
            )
            return True
    except Exception as exc:
        print(f"[Pipeline] Timestamp check notice: {exc}")
    return False


def _write_github_step_summary(
    topic: str,
    title: str,
    fetcher: MediaFetcher,
    video_id: str,
    rendered_via_hyperframes_cli: bool,
):
    engine_label = "HeyGen HyperFrames CLI (`npx hyperframes render`)" if rendered_via_hyperframes_cli else "HyperFrames HTML Bundle + Hybrid FFmpeg Cinema Engine"
    lines = [
        f"## 🎬 US History Short Run Summary (v9.0 HyperFrames Edition) — `{title}`",
        f"- **Historical Topic**: {topic}",
        f"- **Render Engine**: {engine_label}",
        f"- **HyperFrames Composition Bundle**: `output/hyperframes_project/index.html` (`preview.html`, `DESIGN.md`)",
        f"- **YouTube Video ID**: `{video_id}`",
        "",
        "### 📊 Visual & Cost Breakdown",
        "| Metric | Count | Cost Impact |",
        "| :--- | :--- | :--- |",
        f"| **🎨 Paid AI Images (`MAI-Image-2.6-Flash`)** | **`{fetcher.ai_images_used} / {fetcher.max_ai_images}` (Hard Cap)** | Capped at `{fetcher.max_ai_images}` |",
        f"| **🎬 Real Moving Video Clips** | **`{fetcher.video_clips_used}` clips** | **$0.00 (Free)** |",
        f"| **🏛️ Verified Archival Photos** | **`{fetcher.archival_photos_used}` photos** | **$0.00 (Free)** |",
        f"| **📰 Designed Editorial Graphics** | **`{fetcher.reframed_cuts_used}` cuts** | **$0.00 (Free)** |",
        "",
        "### 🎞️ Scene-by-Scene Visual Audit",
        "| Scene | Media Type Used | Source / Asset | Spoken Narration |",
        "| :---: | :--- | :--- | :--- |",
    ]
    for item in fetcher.scene_audit_log:
        lines.append(
            f"| **{item['scene_id']:02d}** | {item['kind']} | `{item['source']}` | {item['narration']} |"
        )
    summary_md = "\n".join(lines) + "\n"
    print("\n" + "=" * 78)
    print(summary_md)
    print("=" * 78 + "\n")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY", "").strip()
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as sf:
                sf.write(summary_md)
        except Exception as exc:
            print(f"[Pipeline] GITHUB_STEP_SUMMARY write notice: {exc}")


def run_pipeline():
    load_dotenv()
    if _should_skip_backup_cron_slot():
        return

    temp_dir = Path("temp")
    output_dir = Path("output")
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (Path("assets/sfx/interstellar_bgm.wav")).exists() or not (Path("assets/sfx/hook_impact.wav")).exists():
        generate_all_sfx()

    llm = FoundryLLMClient()
    custom_topic = os.getenv("CUSTOM_TOPIC", "").strip()
    topic = custom_topic if custom_topic else select_next_topic(llm_client=llm)

    # Step 1: Storyboard via `FOUNDRY_TEXT_MODEL` (`gpt-6-luna`)
    storyboard = llm.generate_storyboard(topic)
    scenes = storyboard.get("scenes", [])
    if not scenes:
        raise RuntimeError("Storyboard contained no scenes.")

    # Step 2: Young US English Voiceover (`en-US-BrianMultilingualNeural`) + Word Timings
    voice_engine = VoiceEngine()
    master_vo, word_timings, enriched_scenes = voice_engine.synthesize_scenes(
        scenes, temp_dir / "audio"
    )

    # Step 3: Fetch Verified Video Clips + Archival Photos + Up to 10 MAI-Image-2.6-Flash Visuals
    fetcher = MediaFetcher(
        topic=topic,
        era_keywords=storyboard.get("historical_era_keywords", []),
        wikipedia_articles=storyboard.get("wikipedia_articles", []),
        video_pool_queries=storyboard.get("video_pool_queries", []),
        required_visual_keywords=storyboard.get("required_visual_keywords", []),
        is_space_or_science=bool(storyboard.get("is_space_or_science", False)),
        llm_client=llm,
    )
    for scene in enriched_scenes:
        asset_path, asset_type = fetcher.fetch_scene_asset(scene, temp_dir / "media")
        scene["asset_path"] = str(asset_path)
        scene["asset_type"] = asset_type
        scene["asset_kind"] = fetcher.scene_audit_log[-1].get("kind", "UNKNOWN") if fetcher.scene_audit_log else "UNKNOWN"

    # Step 4: Generate Word-Synced Captions (.ass) & Broadcast -14 LUFS Master Audio Mix
    ass_path = output_dir / "subtitles.ass"
    generate_synced_ass_captions(word_timings, enriched_scenes, ass_path)

    composer = VideoComposer()
    total_duration = sum(max(0.8, float(s.get("duration_sec", 2.5))) for s in enriched_scenes)
    mixed_audio_wav = temp_dir / "master_mix.wav"
    composer._build_master_audio_mix(
        master_vo, mixed_audio_wav, temp_dir, scenes=enriched_scenes, total_duration=total_duration
    )

    # Step 5: Author Complete HeyGen HyperFrames Project (`output/hyperframes_project/`) & Render MP4
    hf_composer = HyperFramesComposer(project_dir=output_dir / "hyperframes_project")
    hf_project_dir = hf_composer.build_project(
        topic=topic,
        title=storyboard.get("title", topic),
        scenes=enriched_scenes,
        word_timings=word_timings,
        master_audio_wav=mixed_audio_wav,
    )

    final_mp4 = output_dir / "final_short.mp4"
    rendered_via_hf_cli = hf_composer.try_cli_render(final_mp4)
    if not rendered_via_hf_cli:
        composer.render_short(enriched_scenes, master_vo, ass_path, final_mp4, temp_dir / "segments")

    # Step 6: Technical & Documentary QC Gate
    qc = validate_rendered_short(final_mp4, enriched_scenes, output_dir / "quality_report.json")
    print(f"[QualityControl] PASS: {qc}")

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "topic": topic,
                "title": storyboard.get("title"),
                "description": storyboard.get("description"),
                "render_engine": "hyperframes_cli" if rendered_via_hf_cli else "hyperframes_hybrid_ffmpeg",
                "hyperframes_project_dir": str(hf_project_dir),
                "ai_images_used": fetcher.ai_images_used,
                "max_ai_images_cap": fetcher.max_ai_images,
                "video_clips_used": fetcher.video_clips_used,
                "archival_photos_used": fetcher.archival_photos_used,
                "reframed_cuts_used": fetcher.reframed_cuts_used,
                "media_source_manifest": fetcher.scene_audit_log,
                "scene_audit_log": fetcher.scene_audit_log,
                "scenes": enriched_scenes,
                "attributions": fetcher.attributions,
                "sources": storyboard.get("sources", []),
                "quality_control": qc,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Step 7: Upload to YouTube
    video_id = upload_short_to_youtube(
        video_path=final_mp4,
        title=storyboard.get("title", topic),
        description=storyboard.get("description", topic),
        tags=storyboard.get("tags", ["history", "shorts", "ushistory"]),
        attributions=fetcher.attributions,
        sources=storyboard.get("sources", []),
    )

    # Step 8: Write Visual & Cost Audit Table to GitHub Actions Run Summary (`$GITHUB_STEP_SUMMARY`)
    _write_github_step_summary(
        topic, storyboard.get("title", topic), fetcher, video_id, rendered_via_hf_cli
    )

    # Step 9: Mark topic as used and push immediately to GitHub
    if video_id != "dry_run_id":
        mark_topic_completed(topic, video_id=video_id, title=storyboard.get("title", topic))
    print(f"[Pipeline] Completed v9.0 HyperFrames run for topic: '{topic}' (video_id={video_id})")


if __name__ == "__main__":
    run_pipeline()
