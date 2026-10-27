"""
Manages daily US/World history topic selection so the bot never repeats a topic.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

SEED_FILE = Path("data/topics_seed.json")
USED_FILE = Path("data/used_topics.json")
HISTORY_FILE = Path("data/upload_history.json")


def load_used_topics() -> List[str]:
    if not USED_FILE.exists():
        return []
    try:
        data = json.loads(USED_FILE.read_text(encoding="utf-8"))
        return data.get("used", [])
    except Exception:
        return []


def select_next_topic(llm_client=None) -> str:
    custom = os.getenv("CUSTOM_TOPIC", "").strip()
    if custom:
        print(f"[TopicManager] Using custom topic from workflow input: {custom}")
        return custom

    used = set(load_used_topics())
    seeds = []
    if SEED_FILE.exists():
        seeds = json.loads(SEED_FILE.read_text(encoding="utf-8"))

    for topic in seeds:
        if topic not in used:
            print(f"[TopicManager] Selected next unused seed topic: {topic}")
            return topic

    if llm_client is not None:
        print("[TopicManager] Seed list exhausted; generating fresh US/World history topic via GPT-6 Luna...")
        return llm_client.generate_fresh_topic(list(used)[-50:])

    return f"Untold Historical Mystery #{len(used) + 1}"


def mark_topic_completed(topic: str, video_id: Optional[str] = None, title: str = ""):
    USED_FILE.parent.mkdir(parents=True, exist_ok=True)
    used = load_used_topics()
    if topic not in used:
        used.append(topic)
    USED_FILE.write_text(json.dumps({"used": used}, indent=2), encoding="utf-8")

    uploads = []
    if HISTORY_FILE.exists():
        try:
            uploads = json.loads(HISTORY_FILE.read_text(encoding="utf-8")).get("uploads", [])
        except Exception:
            uploads = []

    uploads.append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "title": title or topic,
        "youtube_video_id": video_id or "dry_run",
        "youtube_url": f"https://youtube.com/shorts/{video_id}" if video_id else None,
    })
    HISTORY_FILE.write_text(json.dumps({"uploads": uploads}, indent=2), encoding="utf-8")
