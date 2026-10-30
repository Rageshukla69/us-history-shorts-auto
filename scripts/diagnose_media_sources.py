"""Read-only diagnostic: test archival source discovery without downloading media."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.archive_media import ArchiveMediaFinder, rank_candidates

QUERIES = [
    ("World War II", {"visual_subject": "B-17 bomber aircraft in flight", "visual_keywords": ["B-17", "bomber", "aircraft"], "prefer_media": "video_clip"}),
    ("Cold War", {"visual_subject": "Cold War radar station operators", "visual_keywords": ["radar", "antenna", "operators"], "prefer_media": "archival"}),
]

def main():
    finder = ArchiveMediaFinder()
    for label, scene in QUERIES:
        query = scene["visual_subject"]
        print(f"\n=== {label}: {query} ===")
        for kind in ("video", "image"):
            ranked = rank_candidates(finder.search(query, kind, limit=8), scene)
            print(f"{kind}: {len(ranked)} rights-eligible candidates")
            for item in ranked[:5]:
                print(json.dumps({"score":item.get("relevance_score"),"source":item.get("source"),"title":item.get("title"),"license":item.get("license"),"source_url":item.get("source_url")},ensure_ascii=False))

if __name__ == "__main__": main()
