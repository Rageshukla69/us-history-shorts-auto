"""
Microsoft Foundry LLM Client (v10.0 — HyperFrames Studio Pro + Up to 10 MAI-Image-2.6-Flash Visual Director):
- Mapped directly to your Heroku / GitHub Secrets (`FOUNDRY_TEXT_ENDPOINT`, `FOUNDRY_TEXT_API_KEY`, `gpt-6-luna`).
- Directs up to 10 bespoke 9:16 vertical `MAI-Image-2.6-Flash` cinema reconstructions per Short (`MAX_AI_IMAGES_PER_SHORT = 10` default),
  combined with verified archival motion footage and unique historical photographs.
- Authors rich per-scene HyperFrames motion metadata (`on_screen_callout`, `stat_value`, `stat_label`, `era_tag`, `scene_archetype`)
  so every scene drives custom HTML+CSS+SVG+GSAP choreography.
"""
import json
import os
import re
from typing import Any, Dict, List
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM_PROMPT = """You are the executive story editor and HyperFrames motion-graphics director for a flagship US/World history documentary Shorts channel.
Write a factual, high-retention 48-58 second script (110-135 spoken words) structured into 9-10 tight scenes.

EDITORIAL RULES:
- Open Scene 1 with an immediate, verifiable historical surprise or paradox.
- Distinguish established facts, disputed claims, and legends. Never invent quotations, dollar values, motives, or dialogue.
- Give each scene ONE clear story beat and ONE distinct, concrete visual subject (never repeat the same visual subject across two scenes).
- Provide 3-6 authoritative sources in `sources` (NARA, LOC, Smithsonian, British Museum, peer-reviewed studies, official archives).
- Keep narration punchy and natural when spoken aloud (10-15 words per scene).

HYPERFRAMES & MAI-IMAGE-2.6-FLASH VISUAL RULES:
- You have a budget of up to 10 `MAI-Image-2.6-Flash` bespoke 9:16 vertical cinema reconstructions (`prefer_media: "ai_image"`).
- Set `prefer_media: "ai_image"` on 7 to 10 scenes (especially Scene 1 Hook, dramatic human action beats, close-up mechanical/historical reconstructions, and the Climax) where a razor-sharp 9:16 vertical cinema shot looks far better than a low-res horizontal crop or schematic diagram.
- Use `prefer_media: "video_clip"` or `"archival"` ONLY when an iconic, high-resolution photograph or archival film clip of that exact distinct subject exists.
- Every `image_prompt` MUST describe a hyper-detailed, photorealistic vertical 9:16 cinema shot (specific lighting, lens, period wardrobe/hardware, foreground/background depth, zero text).
- Include HyperFrames HUD metadata on each scene:
  - `on_screen_callout`: 2-4 uppercase words summarizing the beat (e.g., `"FOUND AT 140 FEET"`, `"30+ BRONZE GEARS"`, `"1901 AEGEAN WRECK"`).
  - `stat_value`: Optional short number/year if relevant to the scene (e.g., `"1901"`, `"2,000+"`, `"30+"`, `"82"`, or `""` if no number).
  - `stat_label`: Optional 2-3 word unit/label for `stat_value` (e.g., `"DISCOVERY YEAR"`, `"YEARS OLD"`, `"BRONZE WHEELS"`, `"RECOVERED FRAGMENTS"`).
  - `era_tag`: Short era/location stamp (e.g., `"AEGEAN SEA // 1901"`, `"2ND CENTURY BCE"`, `"ATHENS MUSEUM"`).

Return ONLY valid JSON with this shape:
{
  "title": "Title under 62 characters including #Shorts",
  "description": "Two or three concise sentences with historical context and source note",
  "tags": ["history", "shorts", "ushistory", "documentary"],
  "sources": [{"name":"Source name","url":"https://..."}],
  "historical_era_keywords": ["1901", "Ancient Greece"],
  "wikipedia_articles": ["Relevant article title"],
  "video_pool_queries": ["specific physical action or object query"],
  "required_visual_keywords": ["distinctive subject terms"],
  "is_space_or_science": false,
  "scenes": [{
    "scene_id": 1,
    "narration": "One tight, punchy spoken beat.",
    "visual_subject": "Specific visible subject and action",
    "visual_keywords": ["subject", "object", "period"],
    "video_query": "Specific physical footage query",
    "pexels_query": "Specific visual query, not an abstract topic",
    "wikimedia_query": "Exact historical subject photograph",
    "wikipedia_article": "Relevant article title",
    "image_prompt": "Photorealistic 9:16 vertical historical cinema shot of ..., dramatic rim lighting, 35mm Kodak film grain, shallow depth of field, no text",
    "motion_prompt": "Slow cinematic dolly push-in, natural realistic physical motion of ..., subtle atmospheric smoke/light movement, 35mm documentary realism",
    "prefer_media": "ai_image",
    "effect": "smooth_push_in",
    "on_screen_callout": "SHORT FACT OR DATE",
    "stat_value": "2,000+",
    "stat_label": "YEARS OLD",
    "era_tag": "AEGEAN SEA // 1901"
  }]
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
        max_ai = max(0, int(os.getenv("MAX_AI_IMAGES_PER_SHORT", "10")))
        target_ai_min = max_ai
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Create a 48-58 second documentary Short about: {topic!r}. "
                "Use 9-10 scenes, 110-135 spoken words total. "
                f"Your `MAI-Image-2.6-Flash` budget is {max_ai} images per Short: set `prefer_media` to `\"ai_image\"` on "
                f"between {target_ai_min} and {max_ai} scenes (including Scene 1 Hook, dramatic reconstructions, and Climax) "
                "so the Short has breathtaking 9:16 vertical visuals and never repeats archival photos. "
                "Include `on_screen_callout`, `stat_value`, `stat_label`, and `era_tag` on each scene. Return JSON only."
            )},
        ]
        content = self._call_chat_completions(messages)
        match = re.search(r"\{[\s\S]*\}", content)
        storyboard = json.loads(match.group(0) if match else content)
        storyboard["topic"] = topic
        scenes = storyboard.get("scenes") or []
        if not isinstance(scenes, list) or not scenes:
            raise RuntimeError("Storyboard contained no scenes.")

        for idx, scene in enumerate(scenes, 1):
            scene["scene_id"] = int(scene.get("scene_id") or idx)
            scene["narration"] = str(scene.get("narration") or "").strip()
            scene["visual_subject"] = str(
                scene.get("visual_subject") or scene.get("wikimedia_query") or scene.get("video_query") or topic
            ).strip()
            scene["visual_keywords"] = scene.get("visual_keywords") or []
            scene["motion_prompt"] = str(
                scene.get("motion_prompt")
                or f"Cinematic 35mm documentary dolly movement, natural realistic motion of {scene['visual_subject']}, atmospheric lighting"
            ).strip()
            if scene.get("prefer_media") not in {"video_clip", "archival", "ai_image"}:
                scene["prefer_media"] = "ai_image" if max_ai >= 6 else "archival"

        # If max_ai >= 6 and the LLM under-allocated `ai_image`, promote key narrative beats up to target_ai_min
        current_ai = sum(1 for s in scenes if s.get("prefer_media") == "ai_image")
        if max_ai >= 1 and current_ai < min(max_ai, len(scenes)):
            priority_indices = [0, len(scenes) - 1, len(scenes) // 2, 1, len(scenes) - 2, 3, 5, 2, 4, 6]
            for p_idx in priority_indices:
                if current_ai >= min(max_ai, len(scenes)):
                    break
                if 0 <= p_idx < len(scenes) and scenes[p_idx].get("prefer_media") != "ai_image":
                    scenes[p_idx]["prefer_media"] = "ai_image"
                    current_ai += 1

        # Enforce hard cap independently of model compliance
        ai_left = max_ai
        for scene in scenes:
            if scene.get("prefer_media") == "ai_image" and ai_left > 0:
                ai_left -= 1
            elif scene.get("prefer_media") == "ai_image":
                scene["prefer_media"] = "archival"

        ai_allocated = sum(1 for s in scenes if s.get("prefer_media") == "ai_image")
        print(
            f"[FoundryLLM] Storyboard '{storyboard.get('title')}' "
            f"({len(scenes)} scenes; ai_image allocated={ai_allocated}/{max_ai})."
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
                        "You are an ultra-strict documentary video editor. Given a historical topic and a list of candidate video clips "
                        "(with id, title, desc), return `{\"approved_ids\": [...]}` containing ONLY the IDs of video clips that show:\n"
                        "1. Genuine 1900s-1980s archival film / newsreel footage relevant to the topic/era, OR\n"
                        "2. Clean, realistic B-roll of the exact physical objects, vintage hardware, vehicles, or outdoor animals/landscapes in the story.\n"
                        "STRICTLY REJECT (return `[]` if none qualify):\n"
                        "- ANY modern politician, president (Obama, Biden, Trump, Putin), senator, or government official speaking\n"
                        "- ANY Zoom call, panel discussion, interview, lecture, or webcam video\n"
                        "- ANY street parade, festival, protest, or rally\n"
                        "- ANY amateur indoor pet video on a bedsheet/couch\n"
                        "- ANY cartoon, animation, video game, or LEGO clip."
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
            print(f"[FoundryLLM] Video verification notice: {exc}")
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
                        "You are an ultra-strict archival documentary photo editor. Given a historical topic and candidate images "
                        "(with id, title, desc), return `{\"approved_ids\": [...]}` containing ONLY the IDs of genuine historical photographs "
                        "that directly depict the exact era, event, hardware, or location of `Historical Topic`.\n"
                        "STRICTLY REJECT (return `[]` if none qualify):\n"
                        "- Flat technical schematics, computer diagrams, CAD drawings, or charts\n"
                        "- Paintings of 18th/19th century presidents on 20th-century topics\n"
                        "- Unrelated wars or desert artillery cannons that do not belong to the exact event\n"
                        "- Flat vector logos, agency seals, coats of arms, or flags\n"
                        "- Typed text pages or modern radio telescopes."
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
            print(f"[FoundryLLM] Photo verification notice: {exc}")
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
