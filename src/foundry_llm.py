"""
Microsoft Foundry LLM Client mapped directly to your Heroku Config Vars:
- `FOUNDRY_TEXT_ENDPOINT` (falls back to `FOUNDRY_ENDPOINT`)
- `FOUNDRY_TEXT_API_KEY` (falls back to `FOUNDRY_API_KEY`)
- `FOUNDRY_TEXT_MODEL` (defaults to `gpt-6-luna`)

Includes:
- Automatic URL & parameter format negotiation across all Microsoft Foundry endpoint styles.
- Strict topic-anchored archival query rules so Wikimedia never returns random modern celebrities
  or irrelevant Metaphor/Pun matches (like "rubber" -> Prince Harry).
- Strict AI image credit conservation (`MAX_AI_IMAGES_PER_SHORT`, default `2`).
"""
import json
import os
import re
from typing import Any, Dict, List
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM_PROMPT = """You are an elite viral US documentary director and "Editor V1" pacing architect for 60-second YouTube Shorts on History.
Your audience is in the United States. Write in punchy, natural American English.
Your job is to write a gripping, factually accurate 46-to-54 second spoken script (~115 to 130 words total, strictly under 58 seconds) and break it into 16 to 18 rapid visual scenes.

STRICT EDITOR V1 PACING & ARCHIVAL ACCURACY GUARDRAILS:
1. ZERO 4-SECOND SHOTS: Every single scene MUST correspond to 5 to 9 spoken words (~2.0 to 3.3 seconds).
2. HIGH-RETENTION HOOK (Scene 1 & 2): Start in the middle of the action or a shocking historical paradox in the first 3 seconds.
3. TOPIC-ANCHORED ARCHIVAL SEARCH QUERIES (CRITICAL — NO RANDOM MODERN PEOPLE OR METAPHORS):
   - Every `wikimedia_query` MUST be a concrete, historically specific noun phrase anchored to the main historical topic, era, and entity (e.g., "23rd Headquarters Special Troops WWII dummy tank", "M4 Sherman tank Western Front 1944", "US Army Signal Corps radio operator WWII", "9th US Army Rhine crossing 1945").
   - NEVER use isolated generic adjectives or materials in `wikimedia_query` (like "rubber", "looks real", "illusion", "actors", "secret") that could match modern celebrities, sports events, or tourist plaques.
   - Also provide `wikipedia_article`: the exact title of a relevant historical Wikipedia article for that scene (e.g., "Ghost Army", "Operation Viersen", "Dummy tank", "M4 Sherman", "Operation Plunder").
4. CONSERVE AI IMAGE CREDITS (`prefer_media`):
   - Default to `"wikimedia"` for at least 85-90% of scenes so authentic archival photographs, battle paintings, maps, and historical portraits are used for free.
   - Set `"prefer_media": "ai_image"` on AT MOST 1 or 2 scenes per entire Short (specifically Scene 1 hook or one dramatic close-up where no historical photo exists).
5. SHORT BOUNDARY-SAFE CALLOUT BADGES:
   - `on_screen_callout`: Short 1-3 word punchy historical date, location, or code name under 18 characters (e.g. "FRANCE, 1944", "TOP SECRET", "MARCH 1945"). Leave empty `""` on half the scenes so badges feel selective and impactful.
6. CINEMATIC CAMERA & MOOD:
   - `effect`: Choose from ["blur_snap_zoom_in", "fast_punch_in", "pan_right_sharp", "vintage_sepia_zoom", "high_contrast_zoom_out"].

Return ONLY valid JSON matching this schema:
{
  "title": "Viral US YouTube Short title under 65 chars (include #shorts)",
  "description": "Engaging description with historical context and hashtags (#history #shorts #ushistory #documentary)",
  "tags": ["history", "shorts", "ushistory", "documentary"],
  "historical_era_keywords": ["WWII", "1944", "1945", "US Army", "Ghost Army"],
  "scenes": [
    {
      "scene_id": 1,
      "narration": "In 1944, an entire American army appeared from nowhere.",
      "wikimedia_query": "Ghost Army 23rd Headquarters Special Troops inflatable tank 1944",
      "wikipedia_article": "Ghost Army",
      "image_prompt": "Photorealistic archival 1944 WWII vertical photograph of US soldiers positioning an inflatable dummy Sherman tank in misty French countryside, dramatic documentary lighting, 9:16",
      "pexels_query": "vintage military film",
      "prefer_media": "wikimedia",
      "effect": "blur_snap_zoom_in",
      "on_screen_callout": "FRANCE, 1944"
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
        max_ai = int(os.getenv("MAX_AI_IMAGES_PER_SHORT", "2"))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Create a 60-second History YouTube Short storyboard for a US audience on: '{topic}'. "
                    "Ensure 16 to 18 fast-paced scenes, each 6 to 8 words of narration (~120 words total so the audio stays under 55 seconds). "
                    f"CRITICAL: Set `prefer_media` to 'wikimedia' for almost all scenes (at most {max_ai} scenes may use 'ai_image'). "
                    "Every `wikimedia_query` and `wikipedia_article` must strictly include the historical subject/era so no modern celebrity or irrelevant image is ever matched. "
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
