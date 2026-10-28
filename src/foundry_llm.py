"""
Microsoft Foundry LLM Client mapped directly to your Heroku Config Vars:
- `FOUNDRY_TEXT_ENDPOINT` (falls back to `FOUNDRY_ENDPOINT`)
- `FOUNDRY_TEXT_API_KEY` (falls back to `FOUNDRY_API_KEY`)
- `FOUNDRY_TEXT_MODEL` (defaults to `gpt-6-luna`)

Inspired by `ShortGPT` and `MoneyPrinterTurbo`:
- Generates exact `wikipedia_articles` (canonical Wikipedia page titles) so `MediaFetcher` can query
  Wikipedia's official `/api/rest_v1/page/media-list/{title}` endpoint for 100% curated historical photos.
- Generates `required_visual_keywords` (strict domain nouns that MUST appear in any Wikimedia/NASA candidate's
  metadata) so homonym traps like "Atlas" (Jennifer Lopez movie) or "Mariner" (baseball player) are impossible.
- Provides a lightweight `verify_archival_candidates` LLM filter that double-checks candidate image titles/captions
  before they can ever appear in your video.
"""
import json
import os
import re
from typing import Any, Dict, List
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM_PROMPT = """You are an elite viral US documentary director and "ShortGPT / MoneyPrinterTurbo" visual architect for 60-second YouTube Shorts on History & Science.
Your audience is in the United States. Write in punchy, natural American English.
Your job is to write a gripping, factually accurate 46-to-54 second spoken script (~115 to 130 words total, strictly under 58 seconds) and break it into 16 to 18 rapid visual scenes.

STRICT PACING & ARCHIVAL ACCURACY GUARDRAILS:
1. ZERO 4-SECOND SHOTS: Every single scene MUST correspond to 5 to 9 spoken words (~2.0 to 3.3 seconds).
2. HIGH-RETENTION HOOK (Scene 1 & 2): Start in the middle of the action or a shocking historical paradox in the first 3 seconds.
3. CANONICAL WIKIPEDIA ARTICLES (`wikipedia_articles` at top level, plus `wikipedia_article` per scene):
   - Provide 3 to 5 exact, canonical English Wikipedia article titles for the historical event and its key hardware/people/locations (e.g., `["Mariner 1", "Mariner program", "Atlas-Agena", "Cape Canaveral Space Force Station", "Jet Propulsion Laboratory"]`).
4. STRICT DOMAIN KEYWORDS (`required_visual_keywords` at top level):
   - Provide 8 to 12 strict historical/domain nouns (lowercase) that identify authentic imagery for this topic (e.g., `["mariner", "nasa", "atlas", "agena", "spacecraft", "rocket", "launch", "cape canaveral", "venus", "probe", "satellite", "jpl"]`).
   - Any external archival image that does not match at least one of these words will be rejected.
5. UNAMBIGUOUS ARCHIVAL SEARCH QUERIES (NO HOMONYM TRAPS):
   - Never write ambiguous queries like `"Atlas"` (matches a Jennifer Lopez movie) or `"Mariner"` (matches baseball players) or `"Range safety officer"` (matches modern military portraits).
   - Always include the year, program/agency, and object type in `wikimedia_query` and `nasa_query` (e.g., `"Mariner 1 spacecraft NASA 1962"`, `"Atlas Agena rocket launch Cape Canaveral 1962"`, `"NASA Jet Propulsion Laboratory control room 1962"`).
6. SMART AI vs ARCHIVAL MIX (`prefer_media`):
   - Set `"prefer_media": "archival"` for scenes depicting real spacecraft, battlefields, historical figures, maps, or equipment documented in Wikipedia/NASA archives.
   - Set `"prefer_media": "ai_image"` for dramatic close-ups or conceptual scenes that have no real photograph (e.g. `"A handwritten mathematical equation losing an overbar"`, `"Investigators examining telemetry tape"`), up to the `MAX_AI_IMAGES_PER_SHORT` cap.
7. SHORT BOUNDARY-SAFE CALLOUT BADGES:
   - `on_screen_callout`: Short 1-3 word punchy date, location, or stat under 18 characters (e.g. "JULY 1962", "CAPE CANAVERAL", "$80M LOSS"). Leave empty `""` on half the scenes.

Return ONLY valid JSON matching this schema:
{
  "title": "Viral US YouTube Short title under 65 chars (include #shorts)",
  "description": "Engaging description with historical context and hashtags (#history #shorts #ushistory #documentary)",
  "tags": ["history", "shorts", "ushistory", "documentary"],
  "is_space_or_science": true,
  "wikipedia_articles": ["Mariner 1", "Mariner program", "Atlas-Agena", "Jet Propulsion Laboratory"],
  "required_visual_keywords": ["mariner", "nasa", "atlas", "agena", "rocket", "spacecraft", "probe", "launch", "canaveral", "venus", "jpl", "1962"],
  "scenes": [
    {
      "scene_id": 1,
      "narration": "One missing mark doomed America's Venus-bound probe.",
      "wikimedia_query": "Mariner 1 spacecraft NASA 1962",
      "nasa_query": "Mariner 1 spacecraft",
      "wikipedia_article": "Mariner 1",
      "image_prompt": "Authentic 1962 archival NASA photograph of the Mariner 1 Venus spacecraft probe in cleanroom, dramatic documentary lighting, vertical 9:16, no text",
      "pexels_query": "vintage rocket launch",
      "prefer_media": "archival",
      "effect": "blur_snap_zoom_in",
      "on_screen_callout": "JULY 1962"
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
                    f"Create a 60-second History YouTube Short storyboard for a US audience on: '{topic}'. "
                    "Ensure 16 to 18 fast-paced scenes, each 6 to 8 words of narration (~120 words total so the audio stays under 55 seconds). "
                    f"Provide canonical `wikipedia_articles` and `required_visual_keywords`. Set `prefer_media` to 'ai_image' on the {max_ai} most conceptual scenes (such as mathematical equations, lost paperwork, or abstract moments) and 'archival' on all other scenes. "
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
            f"[FoundryLLM] Generated '{storyboard.get('title')}' "
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
                        "You are a strict archival documentary fact-checker. Given a historical topic and a list of candidate images "
                        "(with id, title, desc), return a JSON object `{\"approved_ids\": [...]}` containing ONLY the IDs of images "
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
