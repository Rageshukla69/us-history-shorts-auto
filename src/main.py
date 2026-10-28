"""
Main End-to-End Orchestrator for the Automated US History Shorts Channel (v4.2 — 9:16 Full-Bleed & Pro Creator Captions Edition).
Configured to drop-in match your exact Heroku Config Vars (`AZURE_SPEECH_KEY`, `FOUNDRY_TEXT_ENDPOINT`,
`FOUNDRY_ENDPOINT`, `MAI-Image-2.6-Flash`, `gpt-6-luna`, `YOUTUBE_TOKEN_JSON`, etc.).

Upgrades in v4.2:
1. Strict 100% Full-Screen 9:16 (`1080x1920`) Visual Normalization (`src/media_fetcher.py` & `src/video_composer.py`) —
   every single scene fills the vertical 9:16 frame edge-to-edge without ever shrinking into a floating box.
2. Pro Creator 82pt Montserrat Bold Word-Synced Captions (`src/caption_builder.py`) with 75ms elastic active-word pop,
   8px outline, 5px shadow, and `MarginV: 530` thumb-stop positioning.
3. Reliable 12-Hour Schedule & Backup Cron Slot Guard (`_should_skip_backup_cron_slot`) so GitHub Actions never drops
   a 12-hour window and never double-uploads.
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
    """
    Because GitHub Actions occasionally drops peak-hour cron triggers under high load,
    our workflow schedules a primary slot (`01:25` & `13:25` UTC) plus a backup slot
    (`04:25` & `16:25` UTC). If this is a scheduled cron run (not a manual `workflow_dispatch`
    run) and a Short was already uploaded within the last `min_hours_between_uploads` hours,
    we exit cleanly so the channel strictly uploads once every ~12 hours.
    """
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


def run_pipeline():
    load_dotenv()
    if _should_skip_backup_cron_slot():
        return

    temp_dir = Path("temp")
    output_dir = Path("output")
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always ensure the uplifting Interstellar documentary score exists
    if not (Path("assets/sfx/interstellar_bgm.wav")).exists():
        generate_all_sfx()

    llm = FoundryLLMClient()
    topic = select_next_topic(llm_client=llm)

    # Step 1: Storyboard & Script via `FOUNDRY_TEXT_MODEL` (`gpt-6-luna`)
    storyboard = llm.generate_storyboard(topic)
    scenes = storyboard.get("scenes", [])
    if not scenes:
        raise RuntimeError("Storyboard contained no scenes.")

    # Step 2: Young US English Voiceover (`en-US-BrianMultilingualNeural`) + Word Timings
    voice_engine = VoiceEngine()
    master_vo, word_timings, enriched_scenes = voice_engine.synthesize_scenes(
        scenes, temp_dir / "audio"
    )

    # Step 3: Fetch Verified 9:16 Archival Photos (Wikipedia REST Media-List + NASA API + LLM Verifier) + Capped `MAI-Image-2.6-Flash`
    fetcher = MediaFetcher(
        topic=topic,
        era_keywords=storyboard.get("historical_era_keywords", []),
        wikipedia_articles=storyboard.get("wikipedia_articles", []),
        required_visual_keywords=storyboard.get("required_visual_keywords", []),
        is_space_or_science=bool(storyboard.get("is_space_or_science", False)),
        llm_client=llm,
    )
    for scene in enriched_scenes:
        asset_path, asset_type = fetcher.fetch_scene_asset(scene, temp_dir / "media")
        scene["asset_path"] = str(asset_path)
        scene["asset_type"] = asset_type

    print(
        f"[Pipeline] Visuals ready: used {fetcher.ai_images_used}/{fetcher.max_ai_images} MAI-Image credits "
        f"and {len(fetcher.used_urls)} verified 9:16 archival photos."
    )

    # Step 4: Generate Pro Creator 82pt Word-Synced Captions (.ass)
    ass_path = output_dir / "subtitles.ass"
    generate_synced_ass_captions(word_timings, enriched_scenes, ass_path)

    # Step 5: Render 1080x1920 MP4 with 100% Full-Bleed 9:16 Framing & Custom/Interstellar BGM Ducking
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
                "verified_pool_size": len(fetcher.verified_pool),
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

    # Step 7: Only mark topic as used if not in dry_run mode
    if video_id != "dry_run_id":
        mark_topic_completed(topic, video_id=video_id, title=storyboard.get("title", topic))
    print(f"[Pipeline] Completed run for topic: '{topic}' (video_id={video_id})")


if __name__ == "__main__":
    run_pipeline()
