"""
Main End-to-End Orchestrator for the Automated US History Shorts Channel (v6.0 — Smooth Cinema & Verified Video Pool Edition).
Configured to drop-in match your exact Heroku Config Vars (`AZURE_SPEECH_KEY`, `FOUNDRY_TEXT_ENDPOINT`,
`FOUNDRY_ENDPOINT`, `MAI-Image-2.6-Flash`, `gpt-6-luna`, `YOUTUBE_TOKEN_JSON`, etc.).

Upgrades in v6.0:
1. Keyless LLM-Verified Video Clip Pool (`src/media_fetcher.py` & `src/foundry_llm.py`) — pre-fetches candidate videos
   from Wikipedia REST Video, Wikimedia Commons Video (`filetype:video` streamable derivatives), NASA Video Archive,
   and US National Archives / Prelinger using concise `video_pool_queries`, verifies every clip with `gpt-6-luna`
   (`verify_video_candidates`), and validates 5 frames per clip to block black screens and text slates.
2. Zero Camera Shake & 3x Subpixel Smooth Cinema Motion (`src/video_composer.py`) — eliminates all `sin()`/`cos()`
   shake and mid-scene white flashes, upscales plates to `3240x5760` before `zoompan` to fix FFmpeg Ticket #4298
   pixel jitter, and animates wide photos/clips with 2-layer parallax over a moving blurred backdrop.
3. Strict 0% Visual Repetition Guarantee — never reuses an image or clip across scenes in the same Short.
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

    # Step 1: Storyboard & Concise Video Pool Queries via `FOUNDRY_TEXT_MODEL` (`gpt-6-luna`)
    storyboard = llm.generate_storyboard(topic)
    scenes = storyboard.get("scenes", [])
    if not scenes:
        raise RuntimeError("Storyboard contained no scenes.")

    # Step 2: Young US English Voiceover (`en-US-BrianMultilingualNeural`) + Word Timings
    voice_engine = VoiceEngine()
    master_vo, word_timings, enriched_scenes = voice_engine.synthesize_scenes(
        scenes, temp_dir / "audio"
    )

    # Step 3: Fetch LLM+CV-Verified Video Clips + Unique Archival Photos + Unique MAI-Image Visuals
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

    print(
        f"[Pipeline] v6.0 Visuals ready: {fetcher.video_clips_used} verified VIDEO CLIPS, "
        f"{fetcher.ai_images_used} unique MAI-Image scenes, "
        f"and {len(fetcher.used_urls)} total verified archival assets (0 repeats)."
    )

    # Step 4: Generate Pro Creator Multi-Color Word-Synced Captions + Subscribe CTA (.ass)
    ass_path = output_dir / "subtitles.ass"
    generate_synced_ass_captions(word_timings, enriched_scenes, ass_path)

    # Step 5: Render 1080x1920 MP4 with Smooth 3x Subpixel Motion, 2-Layer Parallax & Sidechain BGM
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
                "video_clips_used": fetcher.video_clips_used,
                "ai_images_used": fetcher.ai_images_used,
                "verified_video_pool_size": len(fetcher.verified_video_pool),
                "verified_photo_pool_size": len(fetcher.verified_pool),
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

    # Step 7: Mark topic as used and push immediately to GitHub
    if video_id != "dry_run_id":
        mark_topic_completed(topic, video_id=video_id, title=storyboard.get("title", topic))
    print(f"[Pipeline] Completed v6.0 run for topic: '{topic}' (video_id={video_id})")


if __name__ == "__main__":
    run_pipeline()
