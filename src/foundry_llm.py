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

SYSTEM_PROMPT = """You are an elite viral US YouTube Shorts documentary director ("History Bypass / Fern / Johnny Harris" style).
Your goal is 100% SCRIPT-TO-VISUAL ACCURACY, real moving archival/B-roll video clips, and high viewer retention.
Write a gripping, fast-paced 48-to-52 second spoken script (~115 to 125 words total) structured into **10 to 12 vivid cinematic story beats** (~10 words per beat).

STRICT VISUAL ROUTING, REAL VIDEO CLIP SOURCING & COST CAP RULES:
1. 0.0s THUMB-STOPPING HOOK (Scene 1):
   - Scene 1 MUST open with a shocking, counter-intuitive hook containing a concrete year, number, or secret.
   - ALWAYS set `"prefer_media": "ai_image"` on **Scene 1** so Frame 1 (`00:00`) is a jaw-dropping vertical 9:16 cinema poster!
2. BALANCED HYBRID MEDIA MIX (4 AI Images + 4–5 Real Video Clips + 3–4 Archival Photos):
   - Set `"prefer_media": "ai_image"` on **4 scenes** (Scene 1 Hook + the 3 most specific literal scenes where a custom illustration is essential, e.g., a surgeon wiring a cat).
   - Set `"prefer_media": "video_clip"` on **5 scenes** where real physical motion footage or archival 16mm/35mm film exists (e.g., vintage tape recorder spinning, 1960s city street traffic, vintage automobile at night, cat walking outdoors, soldiers marching, artillery firing).
   - Set `"prefer_media": "archival"` on **2 to 3 scenes** naming real historical buildings, declassified programs, or historical figures documented on Wikipedia.
3. PHYSICAL & ARCHIVAL VIDEO QUERIES (`video_pool_queries` + per-scene `video_query`):
   - CRITICAL: NEVER put political agencies, presidents, or abstract nouns (like `"CIA"`, `"Central Intelligence Agency"`, `"Soviet"`, `"Government"`, `"Intelligence"`) in `video_pool_queries` or `video_query` — on public video archives, those words return C-SPAN speeches, Obama, and Zoom interviews!
   - Instead, EVERY entry in `video_pool_queries` (8 strings) and `video_query` MUST describe a **physical object, vintage machine, vehicle, animal action, or 1940s–1970s city/landscape** that appears in vintage Prelinger 16mm film or nature/object B-roll (e.g., for Acoustic Kitty: `["reel to reel tape recorder", "1960s city street traffic", "tabby cat walking", "vintage radio equipment", "1960s automobile night", "telephone switchboard 1960s", "Capitol dome Washington", "vintage microphone"]`).
4. LITERAL `image_prompt` ON EVERY SCENE:
   - Write a hyper-specific, period-accurate `image_prompt` for EVERY scene (even `video_clip` and `archival` scenes) so every scene has an exact visual blueprint.
5. SMOOTH CINEMA CAMERA MOTION (`effect` per scene — ZERO SHAKE):
   - Choose from: `"smooth_push_in"`, `"smooth_pull_out"`, `"cinema_pan_right"`, `"cinema_pan_left"`, `"cinema_tilt_up"`.

Return ONLY valid JSON matching this schema:
{
  "title": "Viral US YouTube Short title under 62 chars (include #shorts)",
  "description": "Gripping 2-sentence hook + historical context + #shorts #history #ushistory #documentary",
  "tags": ["history", "shorts", "ushistory", "documentary", "viralhistory"],
  "is_space_or_science": false,
  "wikipedia_articles": ["Acoustic Kitty", "Central Intelligence Agency", "Cold War", "Signals intelligence"],
  "video_pool_queries": [
    "reel to reel tape recorder",
    "1960s city street traffic",
    "tabby cat walking",
    "vintage radio transmitter",
    "1960s automobile night",
    "vintage laboratory electronics",
    "Washington Monument",
    "vintage microphone"
  ],
  "required_visual_keywords": ["1960s", "1967", "tape", "radio", "cat", "street", "traffic", "automobile", "washington", "cia", "cold war", "microphone"],
  "scenes": [
    {
      "scene_id": 1,
      "narration": "In 1967, the CIA spent twenty million dollars turning a live cat into a spy.",
      "video_query": "reel to reel tape recorder",
      "pexels_query": "vintage reel to reel tape recorder spinning",
      "wikimedia_query": "1960s reel to reel tape recorder historical",
      "wikipedia_article": "Acoustic Kitty",
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
        max_ai = int(os.getenv("MAX_AI_IMAGES_PER_SHORT", "4"))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Create a high-retention 50-second US History YouTube Short storyboard on: '{topic}'. "
                    "Structure the script into **11 to 12 vivid cinematic story beats** (~10 words per beat, ~120 words total). "
                    f"CRITICAL HYBRID MIX: Set `prefer_media` to `'ai_image'` on {max_ai} key scenes (ALWAYS including Scene 1 Hook + the 3 most literal action scenes), "
                    "set `'video_clip'` on 5 scenes (using physical/object/archival `video_query` phrases like vintage machines, vehicles, city streets, nature, or archival hardware — NEVER political speeches or agency names!), "
                    "and set `'archival'` on 2 to 3 scenes. Output ONLY valid JSON."
                ),
            },
        ]
        content = self._call_chat_completions(messages)
        match = re.search(r"\{[\s\S]*\}", content)
        raw_json = match.group(0) if match else content
        storyboard = json.loads(raw_json)
        storyboard["topic"] = topic
        print(
            f"[FoundryLLM] Generated v7.5 Hybrid Storyboard '{storyboard.get('title')}' "
            f"({len(storyboard.get('scenes', []))} beats, max_ai_cap={max_ai}) via {self.model}."
        )
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
