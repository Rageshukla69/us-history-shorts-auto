"""
Microsoft Foundry LLM Client (v5.0 — AI Script-to-Clip Semantic Matcher & High-Retention Hook Engine):
- Mapped directly to your Heroku / GitHub Secrets (`FOUNDRY_TEXT_ENDPOINT`, `FOUNDRY_TEXT_API_KEY`, `gpt-6-luna`).
- Generates word-matched `clip_queries` (`archival_clip_query` + `pexels_query`) for EVERY scene so `MediaFetcher`
  can pull real moving video clips from Internet Archive (Prelinger/NARA/Newsreels), NASA Video Archive,
  Wikimedia Commons Video, and Pexels HD that match the exact spoken words on screen!
- Prioritizes `"prefer_media": "video_clip"` on Scene 1 (0–2.5s Hook), Scene 2, Scene 3, and high-action beats
  (~60%+ of scenes) while using verified Wikipedia/NASA archival photos for specific historical figures/artifacts
  and capped `MAI-Image-2.6-Flash` (`MAX_AI_IMAGES_PER_SHORT`) for conceptual close-ups.
"""
import json
import os
import re
from typing import Any, Dict, List
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM_PROMPT = """You are an elite viral US YouTube Shorts director ("History Bypass / Johnny Harris / Cleo Abram" style) and AI Script-to-Clip Visual Architect.
Your goal is 80%+ "Stayed to Watch" retention and high subscriber conversion for a US audience.
Write a gripping, fast-paced 46-to-52 second spoken script (~115 to 128 words total) broken into 16 to 18 rapid visual scenes.

STRICT RETENTION, SCRIPT-TO-CLIP MATCHING & PACING RULES:
1. 0.0s THUMB-STOPPING HOOK (Scene 1 & Scene 2):
   - Scene 1 MUST start with a shocking, counter-intuitive 5-to-7 word hook containing at least one high-impact power word or number (e.g., "In 1932, Australia lost a war... to birds.", "One dead body fooled Adolf Hitler's entire army.").
   - Never start with slow throat-clearing like "Did you know that in history...".
2. AI WORD-TO-CLIP MATCHING (`archival_clip_query` + `pexels_query` + `prefer_media` on EVERY scene):
   - Look at the exact action verbs and nouns spoken in each scene's `narration`.
   - `archival_clip_query`: A 2-to-4 word search query for public-domain historical motion picture / newsreel footage on Internet Archive, NASA Video, or Wikimedia Video matching the spoken words (e.g., `"WWII tank battle"`, `"rocket launch explosion"`, `"1920s city street"`, `"battleship firing ocean"`, `"vintage factory workers"`, `"atomic bomb test"`).
   - `pexels_query`: A 2-to-4 word cinematic motion B-roll query matching the physical action/mood of the spoken words (e.g., `"stormy ocean waves"`, `"fire explosion sparks"`, `"running marathon crowd"`, `"ancient stone ruins"`, `"secret top secret documents"`).
   - `prefer_media`:
     * Set `"prefer_media": "video_clip"` on **Scene 1, Scene 2, Scene 3**, and on **at least 9 to 11 scenes total** where motion/action matches the spoken words!
     * Set `"prefer_media": "archival"` on scenes naming a specific historical person, specific named ship/artifact, or map documented on Wikipedia.
     * Set `"prefer_media": "ai_image"` on at most `MAX_AI_IMAGES_PER_SHORT` (2–3) ultra-specific conceptual scenes that have no real footage.
3. FLASHY CAMERA FX (`effect` per scene):
   - Choose from: `"impact_shake_zoom"` (use on Scene 1 and major plot twists!), `"whip_flash_zoom"`, `"fast_punch_in"`, `"parallax_pan_right"`, `"high_contrast_zoom_out"`.
4. CANONICAL WIKIPEDIA ARTICLES & STRICT DOMAIN KEYWORDS:
   - `wikipedia_articles`: 3 to 5 exact English Wikipedia article titles for the historical event.
   - `required_visual_keywords`: 8 to 12 lowercase domain nouns so unrelated modern celebrities or sports homonyms are rejected.
5. PUNCHY CALLOUT BADGES & SUBSCRIBE CTA:
   - `on_screen_callout`: Short 1-3 word uppercase badge under 18 chars (e.g. `"1932 AUSTRALIA"`, `"20,000 EMUS"`, `"TOP SECRET"`).
   - End the final scene's narration with a punchy cliffhanger twist or 4-word question that makes viewers comment and subscribe.

Return ONLY valid JSON matching this schema:
{
  "title": "Viral US YouTube Short title under 62 chars (include #shorts)",
  "description": "Gripping 2-sentence hook + historical context + #shorts #history #ushistory #documentary",
  "tags": ["history", "shorts", "ushistory", "documentary", "viralhistory"],
  "is_space_or_science": false,
  "wikipedia_articles": ["Emu War", "Emu", "Royal Australian Artillery"],
  "required_visual_keywords": ["emu", "australia", "1932", "military", "artillery", "birds", "wheat", "campion", "lewis gun"],
  "scenes": [
    {
      "scene_id": 1,
      "narration": "In 1932, Australia declared war on twenty thousand giant birds—and lost.",
      "archival_clip_query": "1930s army soldiers truck",
      "pexels_query": "flock of emus running desert",
      "wikimedia_query": "Emu War 1932 Australia historical",
      "nasa_query": "",
      "wikipedia_article": "Emu War",
      "image_prompt": "Dramatic 1932 archival war photograph of Australian soldiers facing thousands of giant emus in the Western Australian outback, vertical 9:16, no text",
      "prefer_media": "video_clip",
      "effect": "impact_shake_zoom",
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
        max_ai = int(os.getenv("MAX_AI_IMAGES_PER_SHORT", "3"))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Create a high-retention 60-second US History YouTube Short storyboard on: '{topic}'. "
                    "Ensure 16 to 18 rapid scenes (6 to 8 words per scene, ~120 words total). "
                    "CRITICAL FOR RETENTION: Set `prefer_media` to `'video_clip'` on Scene 1, Scene 2, Scene 3, and at least 9 to 11 action scenes total, "
                    "providing vivid `archival_clip_query` and `pexels_query` phrases that match the exact spoken words of each scene! "
                    f"Use `'archival'` for specific historical portraits/artifacts and `'ai_image'` on at most {max_ai} conceptual scenes. "
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
            f"[FoundryLLM] Generated v5.0 Clip-Synced Storyboard '{storyboard.get('title')}' "
            f"({len(storyboard.get('scenes', []))} scenes, max_ai={max_ai}) via {self.model}."
        )
        return storyboard

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
                        "You are a strict archival documentary fact-checker. Given a historical topic and a list of candidate images/clips "
                        "(with id, title, desc), return a JSON object `{\"approved_ids\": [...]}` containing ONLY the IDs of items "
                        "that genuinely depict the historical event, its era, spacecraft, military hardware, historical figures, or locations. "
                        "STRICTLY REJECT any modern celebrities, actors, singers, athletes, baseball players, red carpet photos, "
                        "unrelated modern military portraits, or homonym traps."
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
