"""
Main End-to-End Orchestrator for the 12-Hour Automated US History Shorts Channel.
Configured to drop-in match your exact Heroku Config Vars (`AZURE_SPEECH_KEY`, `FOUNDRY_TEXT_ENDPOINT`,
`FOUNDRY_ENDPOINT`, `MAI-Image-2.6-Flash`, `gpt-6-luna`, `YOUTUBE_TOKEN_JSON`, etc.).

Upgrades in v3.0:
- Boundary-safe 1-to-2 line word-synced captions with attached punctuation (`src/caption_builder.py`)
- Strict MAI-Image credit cap (`MAX_AI_IMAGES_PER_SHORT`, default `2`) + topic-anchored Wikipedia/Wikimedia archival pool (`src/media_fetcher.py`)
- Dark mysterious documentary background score (`mystery_bgm.wav`) with automatic voiceover ducking and zero whoosh/zoop SFX (`src/sfx_generator.py` & `src/video_composer.py`)
"""
import json
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


def run_pipeline():
    load_dotenv()
    temp_dir = Path("temp")
    output_dir = Path("output")
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always ensure the dark mysterious documentary score exists
    if not (Path("assets/sfx/mystery_bgm.wav")).exists():
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

    # Step 3: Fetch Verified Archival Photos (Wikipedia/Wikimedia) + Capped `MAI-Image-2.6-Flash`
    fetcher = MediaFetcher(
        topic=topic,
        era_keywords=storyboard.get("historical_era_keywords", []),
    )
    for scene in enriched_scenes:
        asset_path, asset_type = fetcher.fetch_scene_asset(scene, temp_dir / "media")
        scene["asset_path"] = str(asset_path)
        scene["asset_type"] = asset_type

    print(
        f"[Pipeline] Visuals ready: used {fetcher.ai_images_used}/{fetcher.max_ai_images} MAI-Image credits "
        f"and {len(fetcher.used_urls)} verified archival photos."
    )

    # Step 4: Generate Boundary-Safe Word-Synced Captions (.ass)
    ass_path = output_dir / "subtitles.ass"
    generate_synced_ass_captions(word_timings, enriched_scenes, ass_path)

    # Step 5: Render 1080x1920 MP4 with Frosted Archival Framing & Mysterious BGM Ducking
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
                "scenes": enriched_scenes,
                "attributions": fetcher.attributions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Step 6: Upload to YouTube via `YOUTUBE_TOKEN_JSON`
    video_id = upload_short_to_youtube(
        video_path=final_mp4,
        title=storyboard.get("title", topic),
        description=storyboard.get("description", topic),
        tags=storyboard.get("tags", ["history", "shorts", "ushistory"]),
        attributions=fetcher.attributions,
    )

    # Step 7: Mark topic as used
    mark_topic_completed(topic, video_id=video_id, title=storyboard.get("title", topic))
    print(f"[Pipeline] Completed run for topic: '{topic}'")


if __name__ == "__main__":
    run_pipeline()
