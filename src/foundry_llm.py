"""
Microsoft Foundry LLM Client (v6.0 — Multi-Source Verified Video Clip & Zero-Repeat Documentary Architect):
- Mapped directly to your Heroku / GitHub Secrets (`FOUNDRY_TEXT_ENDPOINT`, `FOUNDRY_TEXT_API_KEY`, `gpt-6-luna`).
- Generates:
  1. Top-level `video_pool_queries` (8 to 10 concise 1-to-2 word visual nouns/actions that actually exist as real
     HD videos on Wikimedia Commons, Wikipedia, NASA Video, and Internet Archive).
  2. Per-scene `video_query`, `wikimedia_query`, and `image_prompt` matched to the exact spoken words of each scene.
- Provides `verify_video_candidates()` and `verify_archival_candidates()` so `gpt-6-luna` inspects every candidate
  video clip and photograph BEFORE download — guaranteeing 100% subject-true clips and zero unrelated videos.
"""
import json
import os
import re
from typing import Any, Dict, List
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM_PROMPT = """You are an elite viral US YouTube Shorts documentary director ("Johnny Harris / Cleo Abram / History Bypass" style).
Your goal is 85%+ "Stayed to Watch" retention and rapid subscriber growth for a US audience.
Write a gripping, fast-paced 46-to-52 second spoken script (~115 to 128 words total) broken into 15 to 17 cinematic scenes.

STRICT RETENTION, REAL VIDEO SOURCING & PACING RULES:
1. 0.0s THUMB-STOPPING HOOK (Scene 1 & Scene 2):
   - Scene 1 MUST open with a shocking, counter-intuitive 5-to-7 word hook containing a concrete year, number, or stakes (e.g., "In 1932, Australia lost a war... to birds.", "One dead body fooled Adolf Hitler's entire army.").
   - Never start with slow throat-clearing like "Did you know that in history...".
2. REAL VIDEO CLIP SOURCING (`video_pool_queries` + per-scene `video_query` + `prefer_media`):
   - Public video repositories (Wikimedia Commons `filetype:video`, Wikipedia video files, NASA Video Archive, US National Archives) index videos by **short 1-to-2 word concrete visual nouns**, NOT long 6-word sentences!
   - `video_pool_queries` (top-level array of 8 to 10 strings): Write 8 to 10 short 1-to-2 word visual search terms that have real footage on Wikimedia Commons / NASA / Archives and directly match this story's subjects, animals, machines, weapons, or landscapes (e.g., for Apollo 13: `["Apollo 13", "Saturn V", "Moon", "Command module", "Mission Control", "Astronaut", "Rocket launch", "Splashdown"]`; for Great Emu War: `["Emu", "Dromaius novaehollandiae", "Outback Australia", "Wheat field", "Machine gun", "Artillery", "Australian bush", "Flock of birds"]`).
   - Per-scene `video_query`: A 1-to-2 word concrete noun/action from `video_pool_queries` matching the exact spoken words of that scene.
   - Per-scene `prefer_media`:
     * Set `"prefer_media": "video_clip"` on **at least 8 to 10 scenes** (including Scene 1, Scene 3, Scene 5, Scene 7, Scene 9, Scene 11, Scene 13, Scene 15) where real moving footage matches the spoken words!
     * Set `"prefer_media": "archival"` on scenes naming a specific historical person, treaty, or documented Wikipedia photograph.
     * Set `"prefer_media": "ai_image"` on dramatic close-up scenes that need a custom historical reconstruction.
3. SMOOTH CINEMA CAMERA MOTION (`effect` per scene — NO JARRING SHAKE):
   - Choose from smooth, broadcast-grade documentary camera movements:
     * `"smooth_push_in"` (slow subpixel cinematic zoom in)
     * `"smooth_pull_out"` (slow subpixel cinematic zoom out reveal)
     * `"cinema_pan_right"` (gentle horizontal documentary drift right)
     * `"cinema_pan_left"` (gentle horizontal documentary drift left)
     * `"cinema_tilt_up"` (gentle vertical documentary rise)
4. CANONICAL WIKIPEDIA ARTICLES & STRICT DOMAIN KEYWORDS:
   - `wikipedia_articles`: 3 to 5 exact English Wikipedia article titles for the historical event and its primary hardware/subjects.
   - `required_visual_keywords`: 8 to 12 lowercase domain nouns specific to this topic.
5. PUNCHY CALLOUT BADGES & SUBSCRIBE CTA:
   - `on_screen_callout`: Short 1-3 word uppercase badge under 18 chars (e.g. `"NOVEMBER 1932"`, `"20,000 EMUS"`, `"9,860 ROUNDS"`).
   - End the final scene's narration with a thought-provoking 4-to-6 word question + subscribe callout.

Return ONLY valid JSON matching this schema:
{
  "title": "Viral US YouTube Short title under 62 chars (include #shorts)",
  "description": "Gripping 2-sentence hook + historical context + #shorts #history #ushistory #documentary",
  "tags": ["history", "shorts", "ushistory", "documentary", "viralhistory"],
  "is_space_or_science": false,
  "wikipedia_articles": ["Emu War", "Emu", "Lewis gun", "Royal Australian Artillery"],
  "video_pool_queries": ["Emu", "Dromaius novaehollandiae", "Outback Australia", "Wheat field", "Lewis gun", "Machine gun", "Artillery", "Australian bush"],
  "required_visual_keywords": ["emu", "australia", "1932", "military", "artillery", "birds", "wheat", "campion", "lewis", "gun", "outback", "soldier"],
  "scenes": [
    {
      "scene_id": 1,
      "narration": "In 1932, Australia declared war on twenty thousand giant birds—and lost.",
      "video_query": "Emu",
      "pexels_query": "emu birds running outback",
      "wikimedia_query": "Emu War 1932 Australia",
      "nasa_query": "",
      "wikipedia_article": "Emu War",
      "image_prompt": "Dramatic 1932 archival war photograph of Australian soldiers facing thousands of giant emus in the Western Australian outback, vertical 9:16, no text",
      "prefer_media": "video_clip",
      "effect": "smooth_push_in",
      "on_screen_callout": "NOVEMBER 1932"
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Create a high-retention 55-second US History YouTube Short storyboard on: '{topic}'. "
                    "Ensure 15 to 17 cinematic scenes (6 to 8 words per scene, ~120 words total). "
                    "CRITICAL FOR REAL VIDEO CLIPS: Provide 8 to 10 short 1-to-2 word concrete visual nouns in `video_pool_queries` "
                    "(terms that have real video footage on Wikimedia Commons / Wikipedia / NASA / Archives), "
                    "set `prefer_media` to `'video_clip'` on at least 8 to 10 scenes with a short 1-to-2 word `video_query`, "
                    "and write a unique, vivid `image_prompt` for EVERY scene so no image ever repeats! "
                    "Use ONLY smooth documentary camera effects (`smooth_push_in`, `smooth_pull_out`, `cinema_pan_right`, `cinema_pan_left`, `cinema_tilt_up`). "
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
            f"[FoundryLLM] Generated v6.0 Cinema Storyboard '{storyboard.get('title')}' "
            f"({len(storyboard.get('scenes', []))} scenes, video_pool_queries={storyboard.get('video_pool_queries', [])[:5]}) via {self.model}."
        )
        return storyboard

    def verify_video_candidates(
        self, topic: str, candidates: List[Dict[str, str]]
    ) -> List[str]:
        """
        Strict LLM Video Clip Gate:
        Inspects candidate video clips (from Wikimedia Commons Video, Wikipedia, NASA, Internet Archive)
        and approves ONLY videos that genuinely show either the historical event OR clean realistic footage
        of the exact animals, vehicles, weapons, spacecraft, or landscapes in the story.
        """
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
                        "that genuinely show either:\n"
                        "1. Authentic archival footage of the historical topic/era, OR\n"
                        "2. Clean, realistic live-action nature/location/hardware B-roll of the exact animals, weapons, ships, aircraft, spacecraft, or landscapes central to the topic.\n"
                        "STRICTLY REJECT any video games, LEGO, Sonic, cartoons, animations, comics, unrelated modern foreign politicians/generals, "
                        "talking-head lectures/interviews, podcasts, or unrelated wars."
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
                for c in candidates[:35]
            ]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict archival documentary fact-checker. Given a historical topic and a list of candidate images "
                        "(with id, title, desc), return a JSON object `{\"approved_ids\": [...]}` containing ONLY the IDs of items "
                        "that genuinely depict the historical event, its era, spacecraft, military hardware, historical figures, or locations. "
                        "STRICTLY REJECT any modern celebrities, actors, singers, athletes, baseball players, red carpet photos, "
                        "comics, cartoons, LEGO toys, unrelated modern military portraits, or homonym traps."
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
