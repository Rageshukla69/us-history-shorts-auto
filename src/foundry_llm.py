"""
Microsoft Foundry LLM Client (v6.5 — High-Retention Viral Architect, Cost-Capped 3 AI Images & Deep Free Archival Sourcing):
- Mapped directly to your Heroku / GitHub Secrets (`FOUNDRY_TEXT_ENDPOINT`, `FOUNDRY_TEXT_API_KEY`, `gpt-6-luna`).
- Strictly caps `MAI-Image-2.6-Flash` to 3 images per Short (`MAX_AI_IMAGES_PER_SHORT = 3`, cutting AI image cost by ~75%!)
  and places those 3 AI images on the 3 highest-retention beats:
  1. Scene 1 (0.0s Thumb-Stopping Hook — guarantees a jaw-dropping vertical 9:16 opener on Frame 1!)
  2. Scene 7 (Midpoint Twist ~20s)
  3. Scene 13 (Climax ~40s)
- Expands `wikipedia_articles` to 8–10 visually rich historical articles and generates specific multi-word
  historical `video_pool_queries` and `wikimedia_query` strings (banning generic 1-word homonyms like "cat" or "antenna")
  so the other 13–14 scenes are 100% free, authentic historical video clips and photographs with ZERO repeats!
"""
import json
import os
import re
from typing import Any, Dict, List
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM_PROMPT = """You are an elite viral US YouTube Shorts documentary director ("History Bypass / Johnny Harris / Fern / Cleo Abram" style).
Your goal is 85%+ "Stayed to Watch" retention, maximum viewer engagement, and low API cost.
Write a gripping, fast-paced 46-to-52 second spoken script (~118 to 130 words total) broken into 15 to 16 cinematic scenes.

STRICT RETENTION, COST-CAPPED AI VISUALS & FREE ARCHIVAL SOURCING RULES:
1. 0.0s THUMB-STOPPING HOOK (Scene 1):
   - Scene 1 MUST open with a shocking, high-stakes 5-to-7 word hook containing a specific year, number, or secret (e.g., "In 1967, the CIA turned a live cat into a $20 million spy.", "In 1932, Australia lost a military war... to birds.").
   - ALWAYS set `"prefer_media": "ai_image"` on **Scene 1** so the very first frame at `00:00` is a breathtaking, 100% subject-accurate vertical 9:16 cinema poster that stops viewers from scrolling!
2. STRICT 3-IMAGE AI COST CAP (`prefer_media: "ai_image"` on ONLY 3 SCENES TOTAL!):
   - To keep Foundry image generation costs ultra-low, set `"prefer_media": "ai_image"` on **AT MOST 3 SCENES TOTAL**:
     * **Scene 1** (The 0.0s Hook)
     * **Scene 7** (The Midpoint Twist)
     * **Scene 12 or 13** (The Climax)
   - All other 12 to 13 scenes MUST use `"prefer_media": "video_clip"` (6 to 8 scenes) or `"prefer_media": "archival"` (5 to 6 scenes), which are 100% FREE!
3. BAN GENERIC 1-WORD HOMONYMS IN `video_pool_queries`, `video_query`, AND `wikimedia_query`:
   - NEVER use a bare generic word like `"cat"`, `"dog"`, `"antenna"`, `"car"`, or `"bird"` by itself — on Wikimedia Commons, `"cat"` returns amateur home videos of kittens on a bedsheet, and `"antenna"` returns modern desert radio telescopes!
   - Every entry in `video_pool_queries` (8 to 10 strings), `video_query`, and `wikimedia_query` MUST be a **2-to-4 word historical/documentary search phrase** that includes the era, agency, country, or historical context (e.g., `"Cold War espionage"`, `"1960s Washington DC"`, `"Central Intelligence Agency"`, `"Soviet Union Moscow"`, `"vintage radio receiver"`, `"White House 1960s"`, `"Capitol Building Washington"`, `"KGB headquarters"`).
4. 8 TO 10 VISUALLY RICH WIKIPEDIA ARTICLES (`wikipedia_articles`):
   - Provide **8 to 10 exact English Wikipedia article titles** covering not just the narrow event, but also the **historical era, key leaders, buildings/locations, agencies, and hardware** (e.g., for Acoustic Kitty: `["Acoustic Kitty", "Central Intelligence Agency", "Cold War", "Embassy of Russia, Washington, D.C.", "Richard Helms", "KGB", "Espionage", "Kremlin", "Washington, D.C.", "Signals intelligence"]`).
   - This gives our free archival fetcher 30+ real historical photographs so NO image ever repeats!
5. SMOOTH CINEMA CAMERA MOTION (`effect` per scene — NO JARRING SHAKE):
   - Choose from: `"smooth_push_in"`, `"smooth_pull_out"`, `"cinema_pan_right"`, `"cinema_pan_left"`, `"cinema_tilt_up"`.
6. PUNCHY CALLOUT BADGES & DEBATE CTA:
   - `on_screen_callout`: Short 1-3 word uppercase badge under 18 chars (e.g. `"1967: $20M SPY"`, `"SOVIET EMBASSY"`, `"DECLASSIFIED"`).
   - End the final scene's narration with a controversial 5-word question that triggers comments + `"Subscribe for daily history."`

Return ONLY valid JSON matching this schema:
{
  "title": "Viral US YouTube Short title under 62 chars (include #shorts)",
  "description": "Gripping 2-sentence hook + historical context + #shorts #history #ushistory #documentary",
  "tags": ["history", "shorts", "ushistory", "documentary", "viralhistory"],
  "is_space_or_science": false,
  "wikipedia_articles": [
    "Acoustic Kitty",
    "Central Intelligence Agency",
    "Cold War",
    "Embassy of Russia, Washington, D.C.",
    "Richard Helms",
    "KGB",
    "Espionage",
    "Kremlin",
    "Washington, D.C."
  ],
  "video_pool_queries": [
    "Cold War",
    "Washington D.C. 1960s",
    "Soviet Union military",
    "Central Intelligence Agency",
    "Moscow Kremlin",
    "Vintage radio equipment",
    "1960s automobile traffic",
    "United States Capitol"
  ],
  "required_visual_keywords": ["cia", "cold war", "soviet", "washington", "1960s", "espionage", "intelligence", "kremlin", "moscow", "radio", "embassy", "president"],
  "scenes": [
    {
      "scene_id": 1,
      "narration": "In 1967, the CIA spent twenty million dollars turning a cat into a spy.",
      "video_query": "Cold War",
      "pexels_query": "dark vintage spy room reel tape",
      "wikimedia_query": "Central Intelligence Agency headquarters 1960s historical",
      "nasa_query": "",
      "wikipedia_article": "Central Intelligence Agency",
      "image_prompt": "Cinematic 1967 Cold War spy thriller photograph of a shadowy CIA laboratory in Washington D.C. with reel-to-reel tape recorders and a sleek cat on a dossier table, dramatic noir lighting, 35mm film grain, vertical 9:16, no text",
      "prefer_media": "ai_image",
      "effect": "smooth_push_in",
      "on_screen_callout": "1967: $20M SPY"
    }
  ]
}
"""


class FoundryLLMClient:
    def __init__(self):
        self.endpoint = (
            os.getenv("FOUNDRY_TEXT_ENDPOINT")
            or os.getenv("FOUNDRY_ENDPOINT")
            or ""
        ).strip().rstrip("/")
        self.api_key = (
            os.getenv("FOUNDRY_TEXT_API_KEY")
            or os.getenv("FOUNDRY_API_KEY")
            or ""
        ).strip()
        self.model = (
            os.getenv("FOUNDRY_TEXT_MODEL")
            or os.getenv("FOUNDRY_LLM_MODEL")
            or "gpt-6-luna"
        ).strip()

    def _candidate_urls(self) -> List[str]:
        base = self.endpoint
        if "/chat/completions" in base:
            root = base.split("/models/")[0].split("/openai/")[0].split("/chat/completions")[0].rstrip("/")
            candidates = [base]
        else:
            root = base.split("/models")[0].split("/openai")[0].rstrip("/")
            candidates = []

        candidates.extend([
            f"{root}/openai/deployments/{self.model}/chat/completions?api-version=2024-12-01-preview",
            f"{root}/openai/deployments/{self.model}/chat/completions?api-version=2024-10-21",
            f"{root}/models/chat/completions?api-version=2024-05-01-preview",
            f"{root}/openai/v1/chat/completions",
            f"{root}/v1/chat/completions",
            f"{root}/chat/completions",
        ])
        return list(dict.fromkeys(candidates))

    def _call_chat_completions(self, messages: List[Dict[str, str]]) -> str:
        if not self.endpoint or not self.api_key:
            raise RuntimeError(
                "Missing FOUNDRY_TEXT_ENDPOINT / FOUNDRY_TEXT_API_KEY in GitHub Secrets."
            )

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }

        payload_variants = [
            {"model": self.model, "messages": messages},
            {"messages": messages},
        ]

        last_err_detail = ""
        for url in self._candidate_urls():
            for payload in payload_variants:
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=90)
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            if content:
                                return content
                    else:
                        last_err_detail = f"[{resp.status_code} on {url.split('//')[-1][:45]}]: {resp.text[:300]}"
                except Exception as exc:
                    last_err_detail = str(exc)

        raise RuntimeError(f"All Microsoft Foundry endpoint variants failed. Last response: {last_err_detail}")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=2, max=6))
    def generate_storyboard(self, topic: str) -> Dict[str, Any]:
        max_ai = int(os.getenv("MAX_AI_IMAGES_PER_SHORT", "3"))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Create a high-retention 52-second US History YouTube Short storyboard on: '{topic}'. "
                    "Ensure 15 to 16 rapid cinematic scenes (7 to 8 words per scene, ~120 words total). "
                    f"CRITICAL COST & RETENTION RULE: Set `prefer_media` to `'ai_image'` on AT MOST {max_ai} scenes total "
                    "(ALWAYS including Scene 1 for a jaw-dropping 0.0s hook, plus Scene 7 and Scene 13)! "
                    "Set `prefer_media` to `'video_clip'` on 7 to 8 scenes and `'archival'` on 5 to 6 scenes. "
                    "Provide 8 to 10 visually rich `wikipedia_articles` (including era, leaders, locations, and hardware) "
                    "and specific 2-to-4 word historical `video_pool_queries` and `wikimedia_query` strings (NEVER use bare 1-word homonyms like 'cat' or 'antenna'). "
                    "Output ONLY valid JSON."
                ),
            },
        ]
        content = self._call_chat_completions(messages)
        match = re.search(r"\{[\s\S]*\}", content)
        raw_json = match.group(0) if match else content
        storyboard = json.loads(raw_json)
        storyboard["topic"] = topic
        print(
            f"[FoundryLLM] Generated v6.5 High-Retention Storyboard '{storyboard.get('title')}' "
            f"({len(storyboard.get('scenes', []))} scenes, max_ai={max_ai}, wiki_articles={len(storyboard.get('wikipedia_articles', []))}) via {self.model}."
        )
        return storyboard

    def verify_video_candidates(
        self, topic: str, candidates: List[Dict[str, str]]
    ) -> List[str]:
        if not candidates:
            return []
        try:
            compact = [
                {
                    "id": c["id"],
                    "title": c.get("title", "")[:95],
                    "desc": c.get("desc", "")[:140],
                }
                for c in candidates[:35]
            ]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict documentary video editor. Given a historical topic and a list of candidate video clips "
                        "(with id, title, desc), return a JSON object `{\"approved_ids\": [...]}` containing ONLY the IDs of video clips "
                        "that genuinely show authentic archival historical footage, military/government/city footage of the era, "
                        "or cinematic documentary footage central to the topic.\n"
                        "STRICTLY REJECT:\n"
                        "- Amateur home videos of pets on beds/floors/porches\n"
                        "- Video games, LEGO, Sonic, cartoons, animations, comics\n"
                        "- Unrelated modern telescopes/observatories on non-astronomy topics\n"
                        "- Unrelated modern foreign politicians/generals, talking-head interviews, or lectures."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Historical Topic: {topic}\nVideo Candidates: {json.dumps(compact)}\nReturn ONLY valid JSON: {{\"approved_ids\": [...]}}",
                },
            ]
            content = self._call_chat_completions(messages)
            match = re.search(r"\{[\s\S]*\}", content)
            data = json.loads(match.group(0) if match else content)
            approved = data.get("approved_ids", [])
            return [str(x) for x in approved]
        except Exception as exc:
            print(f"[FoundryLLM] Video verification fallback notice: {exc}")
            return []

    def verify_archival_candidates(
        self, topic: str, candidates: List[Dict[str, str]]
    ) -> List[str]:
        if not candidates:
            return []
        try:
            compact = [
                {
                    "id": c["id"],
                    "title": c.get("title", "")[:90],
                    "desc": c.get("desc", "")[:120],
                }
                for c in candidates[:40]
            ]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict archival documentary photo editor. Given a historical topic and a list of candidate images "
                        "(with id, title, desc), return a JSON object `{\"approved_ids\": [...]}` containing ONLY the IDs of items "
                        "that are genuine historical photographs, period locations, historical leaders, military/espionage hardware, or era photography.\n"
                        "STRICTLY REJECT:\n"
                        "- Flat vector logos, agency seals, coats of arms, or flags\n"
                        "- Plain black-and-white typed text pages or document scans with no photograph\n"
                        "- Modern astronomical radio telescopes on non-space topics\n"
                        "- Amateur home pet snapshots, modern celebrities, athletes, or cartoons."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Historical Topic: {topic}\nCandidates: {json.dumps(compact)}\nReturn ONLY valid JSON: {{\"approved_ids\": [...]}}",
                },
            ]
            content = self._call_chat_completions(messages)
            match = re.search(r"\{[\s\S]*\}", content)
            data = json.loads(match.group(0) if match else content)
            approved = data.get("approved_ids", [])
            return [str(x) for x in approved]
        except Exception as exc:
            print(f"[FoundryLLM] Candidate verification fallback notice: {exc}")
            return [c["id"] for c in candidates]

    def generate_fresh_topic(self, recent_topics: List[str]) -> str:
        messages = [
            {
                "role": "system",
                "content": "Return ONLY a single line with one fascinating, true US or World historical event suitable for a viral 60-second YouTube Short.",
            },
            {
                "role": "user",
                "content": f"Suggest 1 new history topic not in this list: {', '.join(recent_topics[-30:])}",
            },
        ]
        content = self._call_chat_completions(messages)
        return content.strip().strip('"')

    def generate_fresh_topics_batch(self, recent_topics: List[str], count: int = 15) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a viral US History & Science documentary producer. "
                    "Return ONLY a JSON array of strings (`[\"Topic 1\", \"Topic 2\", ...]`) containing fascinating, "
                    "true, verifiable US or World historical events suitable for 60-second YouTube Shorts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate {count} brand-new, high-retention US/World history topics that do NOT overlap with any of these: "
                    f"{json.dumps(recent_topics[-60:])}. Return ONLY a valid JSON array of strings."
                ),
            },
        ]
        content = self._call_chat_completions(messages)
        match = re.search(r"\[[\s\S]*\]", content)
        arr = json.loads(match.group(0) if match else content)
        return [str(x).strip() for x in arr if str(x).strip()]
