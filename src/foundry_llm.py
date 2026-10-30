"""
Microsoft Foundry LLM Client (v7.0 — 100% Script-True Multi-Camera Cinema Architect):
- Mapped directly to your Heroku / GitHub Secrets (`FOUNDRY_TEXT_ENDPOINT`, `FOUNDRY_TEXT_API_KEY`, `gpt-6-luna`).
- Eliminates ALL random Wikimedia/Internet Archive video queries that previously pulled unrelated Zoom calls,
  parades, modern politicians, or unrelated paintings.
- Structures each 50-second Short into 11 to 12 ultra-vivid story beats (~10 words / ~4.2s per beat).
  Each story beat has a hyper-specific, period-accurate `image_prompt` that literally depicts the exact spoken words
  of that sentence, and `VideoComposer` v7.0 renders each beat as a 2-Camera Hollywood Sequence
  (Shot A: Wide Establishing Gimbal Shot -> Shot B: 1.32x Director's Axial Close-Up with live 35mm film grain & breathing light),
  giving you 22 to 24 fast-paced, 100% script-true cinema shots per Short!
"""
import json
import os
import re
from typing import Any, Dict, List
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM_PROMPT = """You are an elite viral US YouTube Shorts documentary director ("History Bypass / Fern / Johnny Harris / AI Historian" style).
Your goal is 100% SCRIPT-TO-VISUAL ACCURACY on every single frame and 85%+ "Stayed to Watch" retention.
Write a gripping, fast-paced 48-to-52 second spoken script (~118 to 128 words total) structured into **11 to 12 vivid cinematic story beats** (~10 to 11 words per beat).

STRICT 100% SCRIPT-TRUE VISUAL & STORYTELLING RULES:
1. 0.0s THUMB-STOPPING HOOK (Scene 1):
   - Scene 1 MUST open with a shocking, counter-intuitive hook containing a concrete year, number, or high-stakes twist (e.g., "In 1967, the CIA spent twenty million dollars turning a cat into a live wiretap.", "In 1932, the Australian military declared war on twenty thousand giant birds—and lost.").
2. 100% LITERAL SCRIPT-TO-VISUAL MATCHING (`image_prompt` on EVERY scene):
   - Every single scene's `image_prompt` MUST literally, vividly, and accurately depict the **EXACT action, subject, setting, and historical year spoken in that scene's `narration`**!
   - Include in every `image_prompt`:
     * The exact historical year/decade and setting (e.g., `"1967 Cold War Washington D.C."`, `"1944 World War II Normandy"`).
     * Clear foreground subject in the upper-center third (`y=25%–55%`) so our Dual-Camera Engine can cut from a Wide Establishing Shot (Shot A) to a 1.32x Tight Subject Close-Up (Shot B)!
     * Cinematic lighting style: `"authentic historical documentary photography, dramatic chiaroscuro cinema lighting, 35mm Kodak archival film grain, vertical 9:16 composition, no text, no words, no watermarks"`.
   - Set `"prefer_media": "ai_image"` on ALL scenes by default so every single visual is 100% true to the spoken words (no random internet clips!).
3. EXACT CANONICAL WIKIPEDIA ARTICLE (`canonical_wikipedia_article`):
   - Provide ONLY the **single exact English Wikipedia article title** for the specific historical event (e.g., `"Acoustic Kitty"`, `"Emu War"`, `"Operation Mincemeat"`). Do NOT list generic city or country articles like `"Washington, D.C."` or `"Cold War"`.
4. SMOOTH CINEMA CAMERA MOTION (`effect` per scene — ZERO SHAKE):
   - Choose from: `"smooth_push_in"`, `"smooth_pull_out"`, `"cinema_pan_right"`, `"cinema_pan_left"`, `"cinema_tilt_up"`.
5. PUNCHY CALLOUT BADGES & DEBATE CTA:
   - `on_screen_callout`: Short 1-3 word uppercase badge under 18 chars (e.g. `"1967: $20M SPY"`, `"SOVIET EMBASSY"`, `"THE TAXI TEST"`).
   - End the final scene's narration with a punchy question + `"Subscribe for daily history."`

Return ONLY valid JSON matching this schema:
{
  "title": "Viral US YouTube Short title under 62 chars (include #shorts)",
  "description": "Gripping 2-sentence hook + historical context + #shorts #history #ushistory #documentary",
  "tags": ["history", "shorts", "ushistory", "documentary", "viralhistory"],
  "is_space_or_science": false,
  "canonical_wikipedia_article": "Acoustic Kitty",
  "wikipedia_articles": ["Acoustic Kitty"],
  "required_visual_keywords": ["acoustic kitty", "1967"],
  "scenes": [
    {
      "scene_id": 1,
      "narration": "In 1967, the CIA spent twenty million dollars turning a live cat into a spy.",
      "image_prompt": "1967 Cold War CIA secret electronics laboratory in Langley Virginia, a sleek grey tabby cat sitting on a wooden workbench beside glowing vintage reel-to-reel tape recorders and a shadowy intelligence officer in a 1960s suit, dramatic noir lighting, 35mm archival film photography, vertical 9:16, no text",
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Create a 100% script-true, high-retention 50-second US History YouTube Short storyboard on: '{topic}'. "
                    "Structure the script into **11 to 12 vivid cinematic story beats** (~10 words per beat, ~120 words total). "
                    "CRITICAL: Every single scene MUST have a rich, hyper-specific, period-accurate `image_prompt` that literally "
                    "depicts the exact spoken sentence of that scene, with the main subject positioned clearly in the upper-center "
                    "so our Dual-Camera Engine can render both a Wide Establishing Shot and a Tight Close-Up Shot! "
                    "Set `prefer_media` to `'ai_image'` on every scene. Output ONLY valid JSON."
                ),
            },
        ]
        content = self._call_chat_completions(messages)
        match = re.search(r"\{[\s\S]*\}", content)
        raw_json = match.group(0) if match else content
        storyboard = json.loads(raw_json)
        storyboard["topic"] = topic
        print(
            f"[FoundryLLM] Generated v7.0 100% Script-True Storyboard '{storyboard.get('title')}' "
            f"({len(storyboard.get('scenes', []))} story beats -> {len(storyboard.get('scenes', []))*2} camera shots) via {self.model}."
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
                for c in candidates[:15]
            ]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an ultra-strict historical documentary fact-checker. "
                        "Return `{\"approved_ids\": [...]}` containing ONLY the IDs of photographs that literally and directly "
                        "depict the exact historical event named in `Historical Topic` (not a general city, not a painting of a founder, "
                        "not a different war, not a logo/seal, not a text page). If NONE of the candidates literally depict the exact "
                        "event, return `{\"approved_ids\": []}`."
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
            print(f"[FoundryLLM] Candidate verification notice: {exc}")
            return []

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
