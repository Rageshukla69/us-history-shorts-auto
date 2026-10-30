"""
Main End-to-End Orchestrator for the Automated US History Shorts Channel (v7.5 — Hybrid Video Clip + Cost-Capped 4 AI Image Edition).
Writes a complete **Visual & Cost Breakdown Table** directly onto your GitHub Actions Run Summary page (`$GITHUB_STEP_SUMMARY`)
after every single run so you can see at a glance:
- Exact number of AI images generated (e.g. `4 / 4 cap`)
- Exact number of Real Video Clips used
- Exact number of Archival Photos used
- Scene-by-scene table of every visual used in the Short!
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
from youtube_uploader import upload_short_to_youtube

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


def _write_github_step_summary(topic: str, title: str, fetcher: MediaFetcher, video_id: str):
    """
    Writes a clean Markdown table to `$GITHUB_STEP_SUMMARY` (displayed right on the GitHub Actions run page)
    and prints it to the console so you always know the exact count of AI images, video clips, and photos used!
    """
    lines = [
        f"## 🎬 US History Short Run Summary — `{title}`",
        f"- **Historical Topic**: {topic}",
        f"- **YouTube Video ID**: `{video_id}`",
        "",
        "### 📊 Visual & Cost Breakdown",
        "| Metric | Count | Cost Impact |",
        "| :--- | :--- | :--- |",
        f"| **🎨 Paid AI Images (`MAI-Image-2.6-Flash`)** | **`{fetcher.ai_images_used} / {fetcher.max_ai_images}` (Hard Cap)** | Capped at `{fetcher.max_ai_images}` |",
        f"| **🎬 Real Moving Video Clips** | **`{fetcher.video_clips_used}` clips** | **$0.00 (Free)** |",
        f"| **🏛️ Verified Archival Photos** | **`{fetcher.archival_photos_used}` photos** | **$0.00 (Free)** |",
        f"| **🎞️ Noir Storyboard Macro Cuts** | **`{fetcher.reframed_cuts_used}` cuts** | **$0.00 (Free)** |",
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
    topic = select_next_topic(llm_client=llm)

    # Step 1: Storyboard & Physical Video Pool Queries via `FOUNDRY_TEXT_MODEL` (`gpt-6-luna`)
    storyboard = llm.generate_storyboard(topic)
    scenes = storyboard.get("scenes", [])
    if not scenes:
        raise RuntimeError("Storyboard contained no scenes.")

    # Step 2: Young US English Voiceover (`en-US-BrianMultilingualNeural`) + Word Timings
    voice_engine = VoiceEngine()
    master_vo, word_timings, enriched_scenes = voice_engine.synthesize_scenes(
        scenes, temp_dir / "audio"
    )

    # Step 3: Fetch Verified Video Clips + Verified Archival Photos + Cost-Capped MAI-Image Visuals
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

    # Step 4: Generate Pro Creator Multi-Color Word-Synced Captions + Subscribe CTA (.ass)
    ass_path = output_dir / "subtitles.ass"
    generate_synced_ass_captions(word_timings, enriched_scenes, ass_path)

    # Step 5: Render 1080x1920 MP4 with Dual-Camera Axial Cuts, IMAX Video Framing & -14 LUFS Audio
    composer = VideoComposer()
    final_mp4 = output_dir / "final_short.mp4"
    composer.render_short(enriched_scenes, master_vo, ass_path, final_mp4, temp_dir / "segments")

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "topic": topic,
                "title": storyboard.get("title"),
                "description": storyboard.get("description"),
                "ai_images_used": fetcher.ai_images_used,
                "max_ai_images_cap": fetcher.max_ai_images,
                "video_clips_used": fetcher.video_clips_used,
                "archival_photos_used": fetcher.archival_photos_used,
                "reframed_cuts_used": fetcher.reframed_cuts_used,
                "scene_audit_log": fetcher.scene_audit_log,
                "scenes": enriched_scenes,
                "attributions": fetcher.attributions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Step 6: Upload to YouTube via `YOUTUBE_TOKEN_JSON` (skipped automatically when DRY_RUN=true)
    video_id = upload_short_to_youtube(
        video_path=final_mp4,
        title=storyboard.get("title", topic),
        description=storyboard.get("description", topic),
        tags=storyboard.get("tags", ["history", "shorts", "ushistory"]),
        attributions=fetcher.attributions,
    )

    # Step 7: Write Visual & Cost Audit Table to GitHub Actions Run Summary (`$GITHUB_STEP_SUMMARY`)
    _write_github_step_summary(topic, storyboard.get("title", topic), fetcher, video_id)

    # Step 8: Mark topic as used and push immediately to GitHub
    if video_id != "dry_run_id":
        mark_topic_completed(topic, video_id=video_id, title=storyboard.get("title", topic))
    print(f"[Pipeline] Completed v7.5 run for topic: '{topic}' (video_id={video_id})")


if __name__ == "__main__":
    run_pipeline()
