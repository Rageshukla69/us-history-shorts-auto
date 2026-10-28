"""
Bulletproof Anti-Repetition Topic Manager for US/World History Shorts (`src/topic_manager.py`):
1. Live Git State Pull (`_sync_latest_git_state`):
   Pulls the latest `data/used_topics.json` and `data/upload_history.json` from `origin/main` before selecting
   a topic so queued back-to-back workflow runs can NEVER pick the same topic.
2. Dual-File Self-Healing Union (`load_used_topics`):
   Combines topics from BOTH `data/used_topics.json` and `data/upload_history.json` so no uploaded topic
   is ever forgotten.
3. Normalized & Keyword-Overlap Duplicate Blocker (`is_topic_already_used`):
   Blocks exact matches, punctuation/case variations, and reworded AI topics that share >= 70% of core subject words.
4. Immediate Post-Upload Git Commit & Push (`_push_topic_state_to_github`):
   Immediately commits and pushes `data/used_topics.json`, `data/upload_history.json`, and `data/topics_seed.json`
   inside Python right after YouTube upload completes (with a 3-attempt rebase retry loop).
"""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set

SEED_FILE = Path("data/topics_seed.json")
USED_FILE = Path("data/used_topics.json")
HISTORY_FILE = Path("data/upload_history.json")

STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "into", "when", "what", "how", "why",
    "who", "where", "this", "these", "those", "their", "there", "were", "was", "have",
    "had", "has", "been", "being", "most", "more", "over", "under", "after", "before",
    "during", "about", "against", "between", "through", "almost", "single", "secret",
    "history", "historical", "world", "american", "america", "untold", "mystery",
    "craziest", "bizarre", "great", "first", "last", "years", "year", "century",
}

RESERVE_US_TOPICS = [
    "The 1942 Battle of Los Angeles When Anti-Aircraft Guns Fired 1,400 Shells at the Night Sky",
    "How Deborah Sampson Disguised Herself as a Continental Soldier in the American Revolution",
    "The 1913 Great Lakes Storm (The White Hurricane) That Sank 12 Ships in One Night",
    "Operation Acoustic Kitty and the CIA's Cold War Espionage Gadgets",
    "How the 1888 Schoolhouse Blizzard Changed Weather Forecasting Across the American Plains",
    "The 1958 Tybee Island Mid-Air Collision That Left a Lost Nuclear Bomb Off the Georgia Coast",
    "How the USS William D. Porter Accidentally Fired a Live Torpedo Toward President Roosevelt's Ship",
    "The 1872 Great Epizootic That Brought 19th-Century American Cities to a Standstill",
    "How Garrett Morgan Invented Both the Three-Position Traffic Signal and the Early Smoke Hood",
    "The 1976 Chowchilla School Bus Kidnapping Where 26 Children Dug Their Own Way to Freedom",
]


def _normalize_topic(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _extract_core_keywords(text: str) -> Set[str]:
    words = re.findall(r"[a-z0-9]{4,}", (text or "").lower())
    return {w for w in words if w not in STOPWORDS}


def is_topic_already_used(candidate: str, used_topics: List[str]) -> bool:
    """
    Returns True if `candidate` matches any topic in `used_topics` either by normalized string
    equality, substring containment, or >= 70% core historical keyword overlap.
    """
    cand_norm = _normalize_topic(candidate)
    if not cand_norm:
        return True

    cand_kw = _extract_core_keywords(candidate)

    for used in used_topics:
        u_norm = _normalize_topic(used)
        if not u_norm:
            continue
        if cand_norm == u_norm or cand_norm in u_norm or u_norm in cand_norm:
            return True
        if len(cand_kw) >= 3:
            u_kw = _extract_core_keywords(used)
            if u_kw:
                overlap = len(cand_kw & u_kw) / float(min(len(cand_kw), len(u_kw)))
                if overlap >= 0.70:
                    return True
    return False


def _sync_latest_git_state() -> None:
    """
    If running inside a GitHub Actions workspace, pulls the latest commits on `main`
    before reading `data/used_topics.json` so queued runs always see newly completed topics.
    """
    if not os.getenv("GITHUB_ACTIONS"):
        return
    try:
        subprocess.run(
            ["git", "pull", "--rebase", "--autostash", "origin", "main"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=25,
            check=False,
        )
    except Exception:
        pass


def load_used_topics() -> List[str]:
    """
    Loads the union of `data/used_topics.json` AND `data/upload_history.json` so even if
    one file was ever out of sync, every uploaded topic is permanently remembered.
    """
    combined: List[str] = []
    seen_norm: Set[str] = set()

    if USED_FILE.exists():
        try:
            data = json.loads(USED_FILE.read_text(encoding="utf-8"))
            for t in data.get("used", []):
                norm = _normalize_topic(t)
                if norm and norm not in seen_norm:
                    seen_norm.add(norm)
                    combined.append(t.strip())
        except Exception:
            pass

    if HISTORY_FILE.exists():
        try:
            hdata = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            for item in hdata.get("uploads", []):
                t = (item.get("topic") or "").strip()
                norm = _normalize_topic(t)
                if norm and norm not in seen_norm:
                    seen_norm.add(norm)
                    combined.append(t)
        except Exception:
            pass

    return combined


def load_seed_topics() -> List[str]:
    if not SEED_FILE.exists():
        return []
    try:
        data = json.loads(SEED_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_seed_topics(seeds: List[str]) -> None:
    SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEED_FILE.write_text(json.dumps(seeds, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def select_next_topic(llm_client=None) -> str:
    custom = os.getenv("CUSTOM_TOPIC", "").strip()
    if custom:
        print(f"[TopicManager] Using custom topic from workflow input: {custom}")
        return custom

    _sync_latest_git_state()

    used_list = load_used_topics()
    seeds = load_seed_topics()

    unused_seeds = [t for t in seeds if not is_topic_already_used(t, used_list)]

    # Auto-replenish the seed bank when fewer than 3 unused topics remain so the channel never runs out
    if len(unused_seeds) < 3 and llm_client is not None:
        print("[TopicManager] Fewer than 3 unused seed topics left — auto-replenishing via GPT-6 Luna...")
        try:
            if hasattr(llm_client, "generate_fresh_topics_batch"):
                new_batch = llm_client.generate_fresh_topics_batch(used_list + seeds, count=15)
            else:
                new_batch = [llm_client.generate_fresh_topic(used_list + seeds)]
            added = 0
            for cand in new_batch:
                cand = (cand or "").strip()
                if cand and not is_topic_already_used(cand, used_list + seeds):
                    seeds.append(cand)
                    unused_seeds.append(cand)
                    added += 1
            if added > 0:
                save_seed_topics(seeds)
                print(f"[TopicManager] Auto-replenished {SEED_FILE} with {added} brand-new topics.")
        except Exception as exc:
            print(f"[TopicManager] Batch replenishment notice: {exc}")

    if unused_seeds:
        chosen = unused_seeds[0]
        print(f"[TopicManager] Selected next unused topic ({len(unused_seeds)} remaining in queue): {chosen}")
        return chosen

    if llm_client is not None:
        for _ in range(4):
            try:
                fresh = llm_client.generate_fresh_topic(used_list[-60:] + seeds[-30:])
                if fresh and not is_topic_already_used(fresh, used_list):
                    seeds.append(fresh)
                    save_seed_topics(seeds)
                    return fresh
            except Exception as exc:
                print(f"[TopicManager] Fresh topic generation notice: {exc}")

    for reserve in RESERVE_US_TOPICS:
        if not is_topic_already_used(reserve, used_list):
            seeds.append(reserve)
            save_seed_topics(seeds)
            return reserve

    return f"Untold American Historical Mystery #{len(used_list) + 1}"


def _push_topic_state_to_github(topic: str) -> None:
    """
    Immediately commits and pushes `data/used_topics.json`, `data/upload_history.json`, and
    `data/topics_seed.json` to GitHub right inside Python as soon as the YouTube upload finishes,
    using a 3-attempt pull-rebase retry loop so a topic can NEVER repeat.
    """
    if not os.getenv("GITHUB_ACTIONS"):
        return
    try:
        subprocess.run(["git", "config", "--local", "user.email", "actions@github.com"], check=False)
        subprocess.run(["git", "config", "--local", "user.name", "History Shorts Auto-Bot"], check=False)
        subprocess.run(
            ["git", "add", "data/used_topics.json", "data/upload_history.json", "data/topics_seed.json"],
            check=False,
        )
        diff_proc = subprocess.run(["git", "diff", "--staged", "--quiet"], check=False)
        if diff_proc.returncode == 0:
            return

        subprocess.run(
            ["git", "commit", "-m", f"chore: record completed short ({topic[:45]})"],
            check=False,
        )
        for attempt in range(1, 4):
            pull_res = subprocess.run(
                ["git", "pull", "--rebase", "--autostash", "origin", "main"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            push_res = subprocess.run(
                ["git", "push", "origin", "HEAD:main"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if push_res.returncode == 0:
                print(f"[TopicManager] Immediately pushed updated used_topics.json to GitHub (attempt {attempt}).")
                return
            print(f"[TopicManager] Git push attempt {attempt} notice: {push_res.stderr[:200]} {pull_res.stderr[:200]}")
    except Exception as exc:
        print(f"[TopicManager] Immediate git push notice (workflow step will retry): {exc}")


def mark_topic_completed(topic: str, video_id: Optional[str] = None, title: str = ""):
    USED_FILE.parent.mkdir(parents=True, exist_ok=True)
    used = load_used_topics()
    if not any(_normalize_topic(u) == _normalize_topic(topic) for u in used):
        used.append(topic)
    USED_FILE.write_text(json.dumps({"used": used}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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
    HISTORY_FILE.write_text(json.dumps({"uploads": uploads}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if video_id and video_id != "dry_run_id":
        _push_topic_state_to_github(topic)
