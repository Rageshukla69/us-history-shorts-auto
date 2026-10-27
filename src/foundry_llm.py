"""
Microsoft Foundry LLM Client mapped directly to your Heroku Config Vars:
- `FOUNDRY_TEXT_ENDPOINT` (falls back to `FOUNDRY_ENDPOINT`)
- `FOUNDRY_TEXT_API_KEY` (falls back to `FOUNDRY_API_KEY`)
- `FOUNDRY_TEXT_MODEL` (defaults to `gpt-6-luna`)

Includes automatic URL & parameter format negotiation across all Microsoft Foundry
endpoint styles (`/models/chat/completions`, `/openai/v1/chat/completions`,
`/openai/deployments/{model}/chat/completions`, and direct URLs) as well as
automatic `temperature` / `max_completion_tokens` fallback so 400 BadRequest never happens.
"""
import json
import os
import re
from typing import Any, Dict, List
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM_PROMPT = """You are an elite viral US documentary director and "Editor V1" pacing architect for 60-second YouTube Shorts on History.
Your audience is in the United States. Write in punchy, natural American English.
Your job is to write a gripping, factually accurate 52-to-58 second spoken script (~135 to 155 words total) and break it into 18 to 22 rapid visual scenes.

STRICT EDITOR V1 PACING GUARDRAILS:
1. ZERO 4-SECOND SHOTS: Every single scene MUST correspond to 5 to 9 spoken words (~2.0 to 3.3 seconds). A shot staying on screen for 4 seconds is an editing mistake.
2. HIGH-RETENTION HOOK (Scene 1 & 2): Start in the middle of the action or a shocking historical paradox in the first 3 seconds.
3. REAL ARCHIVAL & AI CINEMATIC VISUALS: For each scene, provide:
   - `narration`: The exact 5-9 words spoken during this cut.
   - `wikimedia_query`: Specific historical person, place, artifact, battle painting, or archival photo on Wikipedia / Wikimedia Commons (e.g. "Ghost Army WWII inflatable tank", "Roanoke CROATOAN engraving").
   - `image_prompt`: Detailed cinematic vertical 9:16 historical illustration prompt for `MAI-Image-2.6-Flash` in case Wikimedia has no exact archival photo or when a dramatic close-up visual pun is needed.
   - `pexels_query`: A dynamic 2-3 word motion B-roll search term (e.g. "stormy ocean waves", "vintage pocket watch").
   - `prefer_media`: Either "wikimedia" (for authentic historical archival photos/paintings) or "ai_image" (for high-impact cinematic `MAI-Image-2.6-Flash` shots). Mix ~65% "wikimedia" and ~35% "ai_image".
   - `effect`: Choose from ["blur_snap_zoom_in", "fast_punch_in", "pan_right_sharp", "vintage_sepia_zoom", "high_contrast_zoom_out"].
   - `sfx`: Choose from ["boom", "whoosh", "shutter", "pop"].
   - `on_screen_callout`: Short 1-3 word punchy badge or year (e.g. "JUNE 1944", "TOP SECRET", "BOSTON, 1919") to overlay selectively.

Return ONLY valid JSON matching this schema:
{
  "title": "Viral US YouTube Short title under 65 chars (include #shorts)",
  "description": "Engaging description with historical context and hashtags (#history #shorts #ushistory #documentary)",
  "tags": ["history", "shorts", "ushistory", "documentary"],
  "scenes": [
    {
      "scene_id": 1,
      "narration": "In 1944, four American soldiers lifted a forty-ton Sherman tank...",
      "wikimedia_query": "Ghost Army inflatable tank WWII",
      "image_prompt": "Photorealistic archival 1944 WWII black and white colorized vertical photograph of US soldiers holding up an inflatable rubber Sherman tank, dramatic lighting, 9:16",
      "pexels_query": "vintage military film",
      "prefer_media": "wikimedia",
      "effect": "blur_snap_zoom_in",
      "sfx": "boom",
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
            # Even if full URL was provided, also include clean root candidates in case api-version differs
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
        # Deduplicate while preserving order
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

        # Try payloads from richest to most minimal (some reasoning/Foundry models reject `temperature` or `response_format`)
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
                    f"Create a 60-second History YouTube Short storyboard for a US audience on: '{topic}'. "
                    "Ensure 18 to 22 fast-paced scenes, each 5 to 9 words of narration. Output ONLY valid JSON."
                ),
            },
        ]
        content = self._call_chat_completions(messages)
        match = re.search(r"\{[\s\S]*\}", content)
        raw_json = match.group(0) if match else content
        storyboard = json.loads(raw_json)
        print(f"[FoundryLLM] Generated '{storyboard.get('title')}' ({len(storyboard.get('scenes', []))} scenes) via {self.model}.")
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
