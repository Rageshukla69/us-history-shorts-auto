"""
Microsoft Foundry LLM Client (v7.5 — Hybrid Script-True Video Clip + Cost-Capped 4 AI Image Architect):
- Mapped directly to your Heroku / GitHub Secrets (`FOUNDRY_TEXT_ENDPOINT`, `FOUNDRY_TEXT_API_KEY`, `gpt-6-luna`).
- Balances:
  1. **Real Moving Video Clips (`4 to 6 clips per Short`)**: Sourced from the Prelinger Archives (1920s–1970s 16mm/35mm
     archival film), US National Archives, NASA Video Archive, and Wikimedia physical B-roll using physical/visual
     `video_pool_queries` (banning political/agency names that return C-SPAN speeches or Zoom interviews!).
  2. **Strict Cost-Capped AI Visuals (`MAX_AI_IMAGES_PER_SHORT = 4` Default)**: Uses `MAI-Image-2.6-Flash` on the 4
     most specific story beats (Scene 1 Hook, Scene 4, Scene 7 Twist, Scene 10 Climax) so ultra-specific script beats
     (like a veterinarian wiring a cat) are 100% literal while your AI image cost is cut by 67% (4 images instead of 12!).
  3. **Strict LLM Verification with ZERO `else:` Override**: `verify_video_candidates()` and `verify_archival_candidates()`
     explicitly reject any clip or photo that does not genuinely match the script's era and physical action.
"""
import json
import os
import re
from typing import Any, Dict, List
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM_PROMPT = """You are the story editor and visual director for a premium US-history documentary Shorts channel.
Write a factual, source-conscious 45-60 second script (105-135 spoken words) in 8-12 scenes.

EDITORIAL RULES:
- Hook with a verifiable surprise, not fabricated numbers or clickbait certainty.
- Distinguish established facts, disputed claims and legends. Never invent quotations, dollar values, motives or dialogue.
- Avoid repeating the same fact. Give each scene one clear story beat and one concrete visual subject.
- Provide 3-6 source URLs or named sources in `sources`; prefer primary/authoritative sources (NARA, LOC, Smithsonian, official archives, academic institutions). If a claim is uncertain, say so in narration or description.
- Keep narration natural when spoken aloud. Avoid long clauses and dense dates.

VISUAL RULES:
- Image generation is optional and disabled by default. Never force an AI image in scene 1.
- Prefer authentic archival footage for genuine motion, then historically relevant photographs, maps, artefacts and documents.
- `video_query` must describe what should actually be visible (object/action/place), not just an institution or abstract topic.
- `wikimedia_query` must describe the exact subject and period. `image_prompt` is a fallback only when paid generation is enabled.
- `visual_subject` should be a short, concrete description of the visible subject. `visual_keywords` should contain 3-8 distinctive nouns/entities.
- Do not imply that generic stock footage is archival evidence. Label reconstructions honestly.
- Choose `prefer_media` from `video_clip`, `archival`, `ai_image`; use `ai_image` only for scenes where no authentic source can reasonably exist.

Return ONLY valid JSON with this shape:
{
  "title": "Title under 62 characters including #Shorts",
  "description": "Two or three concise sentences with historical context and source note",
  "tags": ["history", "shorts", "ushistory", "documentary"],
  "sources": [{"name":"Source name","url":"https://..."}],
  "historical_era_keywords": ["1960s", "Cold War"],
  "wikipedia_articles": ["Relevant article title"],
  "video_pool_queries": ["specific physical action or object query"],
  "required_visual_keywords": ["distinctive subject terms"],
  "is_space_or_science": false,
  "scenes": [{
    "scene_id": 1,
    "narration": "One short spoken beat.",
    "visual_subject": "Specific visible subject and action",
    "visual_keywords": ["subject", "object", "period"],
    "video_query": "Specific physical footage query",
    "pexels_query": "Specific visual query, not an abstract topic",
    "wikimedia_query": "Exact historical subject photograph",
    "wikipedia_article": "Relevant article title",
    "image_prompt": "Historically accurate documentary reconstruction prompt; no text",
    "prefer_media": "archival",
    "effect": "smooth_push_in",
    "on_screen_callout": "SHORT FACT OR DATE"
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
        max_ai = max(0, int(os.getenv("MAX_AI_IMAGES_PER_SHORT", "0")))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Create a 45-60 second documentary Short about: {topic!r}. "
                "Use 8-12 scenes, 105-135 spoken words total. "
                f"Paid image-generation budget is {max_ai}; set at most {max_ai} scenes to ai_image, "
                "and do not force AI imagery for the opening. Make every scene's visual_subject concrete. "
                "Provide source URLs/names for key factual claims. Return JSON only."
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
            scene["visual_subject"] = str(scene.get("visual_subject") or scene.get("wikimedia_query") or scene.get("video_query") or topic).strip()
            scene["visual_keywords"] = scene.get("visual_keywords") or []
            if scene.get("prefer_media") not in {"video_clip", "archival", "ai_image"}:
                scene["prefer_media"] = "archival"
        # Enforce budget independently of model compliance.
        ai_left = max_ai
        for scene in scenes:
            if scene.get("prefer_media") == "ai_image" and ai_left > 0:
                ai_left -= 1
            elif scene.get("prefer_media") == "ai_image":
                scene["prefer_media"] = "archival"
        print(f"[FoundryLLM] Storyboard '{storyboard.get('title')}' ({len(scenes)} scenes; paid image cap={max_ai}).")
        return storyboard

    def verify_video_candidates(
        self, topic: str, candidates: List[Dict[str, str]]
    ) -> List[str]:
        """
        Strict LLM Video Clip Gate (WITH ZERO `else:` FALLBACK):
        Approves ONLY genuine vintage archival footage (Prelinger / NARA / NASA) or clean physical B-roll
        of the exact machines, vehicles, animals, or historical locations in the script.
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
                        "- Paintings of 18th/19th century presidents (like George Washington) on 20th-century topics\n"
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
