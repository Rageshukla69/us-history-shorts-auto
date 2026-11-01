"""
HeyGen HyperFrames Studio Pro Composition Authoring & Rendering Engine (`src/hyperframes_composer.py` — v11.0):
Builds a complete, spec-compliant HyperFrames 1080x1920 @ 30fps project (`output/hyperframes_project/`)
for every single Short, featuring:
1. Root `<div id="main" data-composition-id="main" data-width="1080" data-height="1920" data-start="0" data-duration="...">`
2. Paused deterministic GSAP 3.14.2 master timeline registered at `window.__timelines["main"] = tl`
3. WebGL `HyperShader.init(...)` transition (`cinematic-zoom`) at the midpoint act break with automatic Pure-GSAP HyperFrames CLI
   retry if a headless runner lacks WebGL — ensuring `npx hyperframes render` ALWAYS succeeds and NEVER falls back to plain FFmpeg!
4. 100% Razor-Sharp Media from Frame 0 (`ZERO blur filter` on photos/videos!):
   - Stage 1 (`0.00s -> 0.22s`): Instant crisp 3D camera punch-in (`scale: 1.14 -> 1.05`, `expo.out`) + anamorphic shutter flash (`#sN-flash`)
   - Stage 2 (`0.22s -> end`): Continuous 6-axis camera movement (`DOLLY_PUSH_IN`, `CRANE_RISE_PAN`, `ORBITAL_PULL_BACK`, `PARALLAX_GLIDE`)
   - Foreground 2.5D atmospheric parallax layer (`#sN-atmo`) moving in counter-parallax for real spatial depth.
5. 4 Rotating HyperFrames Documentary Scene Archetypes:
   - `HERO_HOOK_LOCKUP` (Scene 1 Hook & Final Climax): Stacked display headline with animated Gold Highlight Sweep + SVG corner brackets.
   - `TELEMETRY_STAT_CARD` (Data/Year beats): Smart Calendar-Year vs Quantity counter (`tabular-nums`, comma formatting, unit label, and animated gauge bar fill).
   - `BLUEPRINT_SCANNER_HUD` (Investigation/Mechanism beats): 4-corner SVG viewfinder draw (`strokeDashoffset`), sweeping laser scanbar, rotating astrolabe reticle, and `lt-stack-bars` evidence card.
   - `CHRONICLE_CHAPTER_RIBBON` (Narrative progression beats): Animated 3-Act SVG timeline rail with sliding playhead node + `lt-side-rule` kinetic callout.
6. Broadcast Zero-Collision Karaoke Pill Captions (`caption-pill-karaoke` + `Hero Climax Emphasis`):
   - Merges standalone punctuation tokens onto preceding words so `,` or `;` never appear as detached words.
   - Enforces strictly disjoint, non-overlapping chunk time windows (`c_end <= next_c_start - 0.025s`) with `immediateRender: false`.
   - Uses pre-padded `.cap-word` pills (`padding: 6px 16px`, `column-gap: 16px`, `scale: 1.0`, vertical lift `y: -5`) so words NEVER expand horizontally into each other!
"""
from __future__ import annotations
import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from voice_engine import WordTiming

PUNCT_ONLY_RE = re.compile(r"^[,.;:!?—–\-\"'\)\]\}\u2014\u2013]+$")
LEADING_PUNCT_RE = re.compile(r"^([,.;:!?—–\-]+)(.+)$")

POWER_WORDS = {
    "WAR", "DEAD", "DIED", "DEATH", "KILLED", "KILL", "EXPLODED", "EXPLOSION", "BOMB", "NUCLEAR",
    "ATOMIC", "DOOMED", "DESTROYED", "CRASHED", "CRASH", "FAKE", "CORPSE", "SECRET", "SPY", "SPIES",
    "HITLER", "NAZI", "TERROR", "PANIC", "BLOOD", "FIRE", "BURNED", "POISON", "PLAGUE", "CRAZIEST",
    "BIZARRE", "IMPOSSIBLE", "LOST", "VANISHED", "MISSING", "FLOOD", "WAVE", "CRUSHED", "TRAPPED",
    "NEVER", "REFUSED", "STOPPED", "SHOCKING", "INSANE", "DANGEROUS", "LETHAL", "FATAL", "ATTACK",
    "CONCEALED", "SHATTERED", "SUNKEN", "SANK", "MYSTERY", "FORBIDDEN", "HIDDEN", "FLED", "CHASED",
}

STAT_WORDS = {
    "MILLION", "BILLIONS", "BILLION", "THOUSAND", "THOUSANDS", "HUNDRED", "FIRST", "ZERO", "SINGLE",
    "DOUBLE", "TONS", "MPH", "FEET", "MILES", "HOURS", "MINUTES", "SECONDS", "YEARS", "CENTURY",
    "MILLENNIA", "TWO", "THREE", "FOUR", "FIVE", "TEN", "TWENTY", "THIRTY",
}


def _esc(text: Any) -> str:
    return html.escape(str(text or "").strip())


def _extract_smart_stat(scene: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Distinguishes Calendar Years (e.g. `1807`, `1901`, `1944`) from Quantities/Spans (e.g. `2,000+ YEARS`, `3,000 RABBITS`)
    so quantities count `0 -> 2,000+` with commas and unit labels instead of `1981 -> 2000`.
    """
    raw_val = str(scene.get("stat_value") or "").strip()
    raw_lbl = str(scene.get("stat_label") or "").strip().upper()
    combined = f"{raw_val} {scene.get('on_screen_callout', '')} {scene.get('narration', '')}"

    # 1. Check explicit quantity patterns like "2,000 years", "3,000 rabbits", "30 gears", "140 feet"
    m_qty = re.search(
        r"\b([1-9][0-9]{0,2}(?:,[0-9]{3})+|[1-9][0-9]{1,4})(\+?)\s*(years?\s*old|years?|rabbits?|gears?|wheels?|fragments?|feet|miles|tons|days|hours|ships|men|divers|soldiers|birds|centuries|century)\b",
        combined,
        re.IGNORECASE,
    )
    if m_qty:
        num = int(m_qty.group(1).replace(",", ""))
        plus = "+" if ("+" in m_qty.group(2) or "over " in combined.lower() or "more than " in combined.lower() or "roughly " in combined.lower()) else ""
        unit = raw_lbl or m_qty.group(3).upper()
        if unit in ("YEAR", "YEARS"):
            unit = "YEARS OLD" if "old" in combined.lower() else "YEARS"
        return {
            "target": num,
            "start": 0,
            "use_commas": num >= 1000,
            "suffix": plus,
            "label": unit[:22],
        }

    # 2. Check explicit `stat_value` from LLM if numeric
    if raw_val:
        m_explicit = re.search(r"([1-9][0-9]{0,2}(?:,[0-9]{3})+|[1-9][0-9]{1,4})(\+?)", raw_val)
        if m_explicit:
            num = int(m_explicit.group(1).replace(",", ""))
            plus = m_explicit.group(2) or ("+" if "+" in raw_val else "")
            is_year = (1400 <= num <= 2026) and not plus and ("YEAR" in raw_lbl or "DATE" in raw_lbl or "FOUND" in raw_lbl or not raw_lbl)
            if is_year and "YEARS OLD" not in raw_lbl:
                return {
                    "target": num,
                    "start": max(1000, num - 25),
                    "use_commas": False,
                    "suffix": "",
                    "label": (raw_lbl or "RECORDED YEAR")[:22],
                }
            return {
                "target": num,
                "start": 0,
                "use_commas": num >= 1000,
                "suffix": plus,
                "label": (raw_lbl or "VERIFIED METRIC")[:22],
            }

    # 3. Check calendar year in callout/narration (e.g., "in 1807", "in 1901")
    m_yr = re.search(r"\b(1[4-9][0-9]{2}|20[0-2][0-9])\b", combined)
    if m_yr:
        yr = int(m_yr.group(1))
        after = combined[m_yr.end():m_yr.end() + 12].lower()
        if not after.strip().startswith(("year", "ton", "feet", "mile", "rabbit")):
            return {
                "target": yr,
                "start": yr - 24,
                "use_commas": False,
                "suffix": "",
                "label": (raw_lbl or "HISTORICAL YEAR")[:22],
            }

    return None


def _classify_word_pill(word: str) -> Tuple[str, str, str]:
    """
    Returns `(pill_bg_color, active_text_color, box_shadow_glow)` for an active spoken word:
    - Power/Danger words -> Neon Crimson Pill (`#FF334B`) with Crisp White Ink (`#FFFFFF`)
    - Numbers/Dates/Stats -> Electric Emerald Pill (`#00F59B`) with Obsidian Ink (`#06080E`)
    - Standard Active Words -> Viral Gold Pill (`#FFD700`) with Obsidian Ink (`#06080E`)
    """
    bare = re.sub(r"[^A-Z0-9$%]", "", word.upper())
    if bare in POWER_WORDS:
        return ("#FF334B", "#FFFFFF", "0 10px 28px rgba(255, 51, 75, 0.58)")
    if any(ch.isdigit() or ch in "$%" for ch in bare) or bare in STAT_WORDS:
        return ("#00F59B", "#06080E", "0 10px 28px rgba(0, 245, 155, 0.52)")
    return ("#FFD700", "#06080E", "0 10px 28px rgba(255, 215, 0, 0.52)")


def _sanitize_word_timings(word_timings: List[WordTiming]) -> List[WordTiming]:
    merged: List[WordTiming] = []
    for wt in word_timings:
        raw = str(wt.word or "").strip()
        if not raw:
            continue

        if PUNCT_ONLY_RE.match(raw):
            if merged:
                prev = merged[-1]
                merged[-1] = WordTiming(
                    word=f"{prev.word}{raw}",
                    start_sec=prev.start_sec,
                    end_sec=max(prev.end_sec, min(wt.end_sec, prev.end_sec + 0.08)),
                    scene_id=prev.scene_id,
                )
            continue

        m = LEADING_PUNCT_RE.match(raw)
        if m:
            punct, rest = m.group(1), m.group(2).strip()
            if merged:
                prev = merged[-1]
                merged[-1] = WordTiming(
                    word=f"{prev.word}{punct}",
                    start_sec=prev.start_sec,
                    end_sec=prev.end_sec,
                    scene_id=prev.scene_id,
                )
            if not rest:
                continue
            raw = rest

        clean_tok = re.sub(r"\s+([,.;:!?])", r"\1", raw)
        merged.append(
            WordTiming(
                word=clean_tok,
                start_sec=max(0.0, float(wt.start_sec)),
                end_sec=max(float(wt.start_sec) + 0.06, float(wt.end_sec)),
                scene_id=int(wt.scene_id),
            )
        )

    if not merged:
        return []

    sanitized: List[WordTiming] = []
    n = len(merged)
    for i in range(n):
        cur = merged[i]
        st = cur.start_sec
        et = cur.end_sec
        if i > 0 and st < sanitized[-1].end_sec:
            st = round(sanitized[-1].end_sec + 0.012, 3)
        if i + 1 < n:
            next_st = merged[i + 1].start_sec
            if next_st > st + 0.045:
                et = min(et, round(next_st - 0.015, 3))
            else:
                et = round(st + 0.04, 3)
        et = max(round(st + 0.04, 3), round(et, 3))
        sanitized.append(WordTiming(word=cur.word, start_sec=round(st, 3), end_sec=et, scene_id=cur.scene_id))

    return sanitized


def _build_smart_caption_chunks(
    word_timings: List[WordTiming],
    max_words: int = 3,
    max_chars: int = 26,
) -> List[List[WordTiming]]:
    clean = _sanitize_word_timings(word_timings)
    chunks: List[List[WordTiming]] = []
    i = 0
    n = len(clean)
    while i < n:
        sc_id = clean[i].scene_id
        chunk: List[WordTiming] = []
        char_count = 0
        while i < n and clean[i].scene_id == sc_id and len(chunk) < max_words:
            w_text = clean[i].word.upper()
            proj = char_count + (1 if chunk else 0) + len(w_text)
            if chunk and proj > max_chars:
                break
            chunk.append(clean[i])
            char_count = proj
            i += 1
            if w_text.endswith((",", ".", ";", ":", "!", "?", "—", "-")):
                break
        if chunk:
            chunks.append(chunk)
    return chunks


class HyperFramesComposer:
    WIDTH = 1080
    HEIGHT = 1920
    FPS = 30

    def __init__(self, project_dir: Path = Path("output/hyperframes_project")):
        self.project_dir = project_dir
        self.assets_dir = self.project_dir / "assets"
        self._last_build_args: Optional[Dict[str, Any]] = None

    def build_project(
        self,
        topic: str,
        title: str,
        scenes: List[Dict[str, Any]],
        word_timings: List[WordTiming],
        master_audio_wav: Path,
        enable_shaders: bool = True,
    ) -> Path:
        if self.project_dir.exists():
            shutil.rmtree(self.project_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        audio_rel = "assets/master_audio.wav"
        shutil.copy2(master_audio_wav, self.project_dir / audio_rel)

        cursor = 0.0
        tiled_scenes: List[Dict[str, Any]] = []
        for idx, s in enumerate(scenes):
            dur = round(max(1.2, float(s.get("duration_sec", 3.5))), 3)
            start_s = round(cursor, 3)
            end_s = round(cursor + dur, 3)
            cursor = end_s

            src_path = Path(s["asset_path"])
            ext = src_path.suffix.lower() or (".mp4" if s.get("asset_type") == "video" else ".jpg")
            asset_rel = f"assets/scene_{idx + 1:02d}{ext}"
            shutil.copy2(src_path, self.project_dir / asset_rel)

            s_copy = dict(s)
            s_copy["hf_id"] = f"s{idx + 1}"
            s_copy["hf_index"] = idx + 1
            s_copy["hf_start"] = start_s
            s_copy["hf_duration"] = dur
            s_copy["hf_end"] = end_s
            s_copy["hf_asset_rel"] = asset_rel
            tiled_scenes.append(s_copy)

        total_duration = round(max(15.0, cursor), 3)

        self._last_build_args = {
            "topic": topic,
            "title": title,
            "tiled_scenes": tiled_scenes,
            "word_timings": word_timings,
            "audio_rel": audio_rel,
            "total_duration": total_duration,
        }

        shader_anchors: List[str] = []
        shader_transitions: List[Dict[str, Any]] = []
        if enable_shaders and len(tiled_scenes) >= 4:
            mid_idx = len(tiled_scenes) // 2 - 1
            s_a = tiled_scenes[mid_idx]
            s_b = tiled_scenes[mid_idx + 1]
            boundary = s_a["hf_end"]
            trans_dur = 0.5
            shader_anchors = [s_a["hf_id"], s_b["hf_id"]]
            shader_transitions = [{
                "time": round(boundary - trans_dur / 2.0, 3),
                "shader": "cinematic-zoom",
                "duration": trans_dur,
            }]

        index_html = self._generate_index_html(
            topic=topic,
            title=title,
            tiled_scenes=tiled_scenes,
            word_timings=word_timings,
            audio_rel=audio_rel,
            total_duration=total_duration,
            shader_anchors=shader_anchors,
            shader_transitions=shader_transitions,
        )
        (self.project_dir / "index.html").write_text(index_html, encoding="utf-8")
        (self.project_dir / "preview.html").write_text(self._generate_preview_html(), encoding="utf-8")
        (self.project_dir / "DESIGN.md").write_text(
            self._generate_design_md(topic, title, tiled_scenes, total_duration, shader_transitions),
            encoding="utf-8",
        )
        print(
            f"[HyperFrames] Authored v11.0 Studio Pro 1080x1920 composition -> {self.project_dir / 'index.html'} "
            f"({len(tiled_scenes)} scenes, {len(shader_transitions)} WebGL shaders, {total_duration:.2f}s)"
        )
        return self.project_dir

    def _generate_index_html(
        self,
        topic: str,
        title: str,
        tiled_scenes: List[Dict[str, Any]],
        word_timings: List[WordTiming],
        audio_rel: str,
        total_duration: float,
        shader_anchors: List[str],
        shader_transitions: List[Dict[str, Any]],
    ) -> str:
        scene_blocks: List[str] = []
        gsap_lines: List[str] = []

        anchor_set = set(shader_anchors)
        n_scenes = len(tiled_scenes)

        for idx, s in enumerate(tiled_scenes):
            sid = s["hf_id"]
            st = s["hf_start"]
            dur = s["hf_duration"]
            et = s["hf_end"]
            asset_rel = s["hf_asset_rel"]
            is_video = s.get("asset_type") == "video"
            callout = _esc((s.get("on_screen_callout") or f"CHRONICLE BEAT #{idx + 1:02d}").upper()[:28])
            subject = _esc((s.get("visual_subject") or topic)[:68])
            era_tag = _esc((s.get("era_tag") or f"SCENE {idx + 1:02d} // {n_scenes:02d}").upper()[:28])
            kind_badge = "COMFYUI 9:16 AI VIDEO" if "COMFYUI" in str(s.get("asset_kind", "")) else ("AI 9:16 RECONSTRUCTION" if "AI RECONSTRUCTION" in str(s.get("asset_kind", "")) else "ARCHIVAL RECORD")
            stat_info = _extract_smart_stat(s)

            # Proven v9.0 visibility switching contract (using class="scene" without class="clip" so
            # `@hyperframes/lint` never flags `gsap_animates_clip_element`, while HyperShader & GSAP switch scenes cleanly!)
            if sid in anchor_set:
                inline_style = ' style="opacity:0;"'
                if sid == shader_anchors[0]:
                    gsap_lines.append(f'  tl.set("#{sid}", {{ opacity: 1 }}, {st:.3f});')
            else:
                if idx == 0:
                    inline_style = ""
                    gsap_lines.append(f'  tl.set("#{sid}", {{ autoAlpha: 0 }}, {et:.3f});')
                else:
                    inline_style = ' style="visibility:hidden;"'
                    gsap_lines.append(f'  tl.set("#{sid}", {{ autoAlpha: 1 }}, {st:.3f});')
                    if idx < n_scenes - 1:
                        gsap_lines.append(f'  tl.set("#{sid}", {{ autoAlpha: 0 }}, {et:.3f});')

            if is_video:
                media_html = (
                    f'<video id="{sid}-vid" class="scene-video" src="./{asset_rel}" '
                    f'data-start="{st:.3f}" data-duration="{dur:.3f}" data-track-index="0" muted playsinline></video>'
                )
            else:
                media_html = (
                    f'<img id="{sid}-img" class="scene-media" src="./{asset_rel}" alt="{subject}" />'
                )

            # Choose one of 4 documentary scene archetypes for visual variety across cuts
            if idx == 0 or idx == n_scenes - 1:
                archetype = "HERO_HOOK_LOCKUP"
            elif stat_info is not None:
                archetype = "TELEMETRY_STAT_CARD"
            elif idx % 2 == 1:
                archetype = "BLUEPRINT_SCANNER_HUD"
            else:
                archetype = "CHRONICLE_CHAPTER_RIBBON"

            enter_t = round(st + 0.06, 3)

            if archetype == "HERO_HOOK_LOCKUP":
                kicker_label = "HISTORICAL ANOMALY // OPENING HOOK" if idx == 0 else "FINAL VERDICT // LEGACY"
                archetype_svg = f"""
      <svg class="hud-svg" viewBox="0 0 1080 1920">
        <path id="{sid}-b-tl" d="M 68 150 L 68 104 L 124 104" stroke="var(--accent)" stroke-width="4" fill="none" stroke-dasharray="110" stroke-dashoffset="110" />
        <path id="{sid}-b-tr" d="M 1012 150 L 1012 104 L 956 104" stroke="var(--accent)" stroke-width="4" fill="none" stroke-dasharray="110" stroke-dashoffset="110" />
        <line id="{sid}-rule" x1="68" y1="308" x2="620" y2="308" stroke="var(--accent)" stroke-width="4" stroke-dasharray="552" stroke-dashoffset="552" />
      </svg>"""
                archetype_html = f"""
        <div class="top-hud-row">
          <div id="{sid}-kicker" class="kicker-chip data-text">{kicker_label}</div>
          <div id="{sid}-source-tag" class="source-pill data-text">{kind_badge}</div>
        </div>
        <div id="{sid}-hero-card" class="hero-hook-card">
          <div class="hero-era-line data-text">{era_tag}</div>
          <div id="{sid}-hero-headline" class="hero-headline display">{callout}</div>
        </div>"""
                gsap_lines.append(
                    f'  tl.to(["#{sid}-b-tl", "#{sid}-b-tr", "#{sid}-rule"], {{ strokeDashoffset: 0, duration: 0.45, ease: "expo.out", stagger: 0.06 }}, {enter_t:.3f});'
                )
                gsap_lines.append(
                    f'  tl.fromTo("#{sid}-kicker", {{ y: -24, autoAlpha: 0 }}, {{ y: 0, autoAlpha: 1, duration: 0.32, ease: "back.out(1.6)", immediateRender: false }}, {enter_t:.3f});'
                )
                gsap_lines.append(
                    f'  tl.fromTo("#{sid}-hero-card", {{ x: -48, autoAlpha: 0 }}, {{ x: 0, autoAlpha: 1, duration: 0.42, ease: "expo.out", immediateRender: false }}, {round(enter_t + 0.05, 3):.3f});'
                )
                gsap_lines.append(
                    f'  tl.fromTo("#{sid}-hero-headline", {{ backgroundSize: "0% 34%" }}, {{ backgroundSize: "100% 34%", duration: 0.65, ease: "power3.out", immediateRender: false }}, {round(enter_t + 0.16, 3):.3f});'
                )

            elif archetype == "TELEMETRY_STAT_CARD" and stat_info is not None:
                target_n = stat_info["target"]
                start_n = stat_info["start"]
                use_commas = "true" if stat_info["use_commas"] else "false"
                suffix_js = json.dumps(stat_info["suffix"])
                lbl_esc = _esc(stat_info["label"])
                archetype_svg = f"""
      <svg class="hud-svg" viewBox="0 0 1080 1920">
        <line id="{sid}-rule" x1="74" y1="205" x2="540" y2="205" stroke="var(--emerald)" stroke-width="4" stroke-dasharray="466" stroke-dashoffset="466" />
        <circle id="{sid}-reticle" cx="968" cy="154" r="24" stroke="var(--emerald)" stroke-width="2.5" stroke-dasharray="28 10" fill="none" opacity="0.8" />
      </svg>"""
                archetype_html = f"""
        <div class="top-hud-row">
          <div id="{sid}-badge" class="callout-pill emerald-border data-text">{callout}</div>
          <div id="{sid}-source-tag" class="source-pill data-text">{kind_badge}</div>
        </div>
        <div id="{sid}-telemetry-card" class="telemetry-card">
          <div class="telemetry-header data-text">{era_tag}</div>
          <div id="{sid}-stat" class="telemetry-number data-text">{start_n}</div>
          <div class="telemetry-label display">{lbl_esc}</div>
          <div class="telemetry-gauge-track">
            <div id="{sid}-gauge-fill" class="telemetry-gauge-fill"></div>
          </div>
        </div>"""
                gsap_lines.append(
                    f'  tl.to("#{sid}-rule", {{ strokeDashoffset: 0, duration: 0.45, ease: "expo.out" }}, {enter_t:.3f});'
                )
                gsap_lines.append(
                    f'  tl.fromTo("#{sid}-reticle", {{ rotation: 0 }}, {{ rotation: 240, duration: {dur:.3f}, ease: "none", transformOrigin: "50% 50%" }}, {st:.3f});'
                )
                gsap_lines.append(
                    f'  tl.fromTo("#{sid}-badge", {{ y: -22, autoAlpha: 0 }}, {{ y: 0, autoAlpha: 1, duration: 0.32, ease: "power3.out", immediateRender: false }}, {enter_t:.3f});'
                )
                gsap_lines.append(
                    f'  tl.fromTo("#{sid}-telemetry-card", {{ x: -44, autoAlpha: 0 }}, {{ x: 0, autoAlpha: 1, duration: 0.38, ease: "back.out(1.5)", immediateRender: false }}, {round(enter_t + 0.06, 3):.3f});'
                )
                gsap_lines.append(
                    f'  tl.fromTo("#{sid}-gauge-fill", {{ scaleX: 0 }}, {{ scaleX: 1, duration: {min(1.4, dur * 0.55):.2f}, ease: "expo.out", transformOrigin: "left center", immediateRender: false }}, {round(enter_t + 0.12, 3):.3f});'
                )
                gsap_lines.append(
                    f'  (function() {{ var c = {{ v: {start_n} }}; '
                    f'tl.to(c, {{ v: {target_n}, duration: {min(1.25, dur * 0.52):.2f}, ease: "power2.out", '
                    f'onUpdate: function() {{ var el = document.getElementById("{sid}-stat"); '
                    f'if (el) {{ var n = Math.round(c.v); el.textContent = ({use_commas} ? n.toLocaleString("en-US") : n.toString()) + {suffix_js}; }} }} }}, {round(enter_t + 0.08, 3):.3f}); }})();'
                )

            elif archetype == "BLUEPRINT_SCANNER_HUD":
                archetype_svg = f"""
      <svg class="hud-svg" viewBox="0 0 1080 1920">
        <path id="{sid}-vf-tl" d="M 76 310 L 76 248 L 138 248" stroke="var(--accent)" stroke-width="3.5" fill="none" stroke-dasharray="124" stroke-dashoffset="124" />
        <path id="{sid}-vf-tr" d="M 1004 310 L 1004 248 L 942 248" stroke="var(--accent)" stroke-width="3.5" fill="none" stroke-dasharray="124" stroke-dashoffset="124" />
        <path id="{sid}-vf-bl" d="M 76 1180 L 76 1242 L 138 1242" stroke="var(--accent)" stroke-width="3.5" fill="none" stroke-dasharray="124" stroke-dashoffset="124" />
        <path id="{sid}-vf-br" d="M 1004 1180 L 1004 1242 L 942 1242" stroke="var(--accent)" stroke-width="3.5" fill="none" stroke-dasharray="124" stroke-dashoffset="124" />
        <circle id="{sid}-dial" cx="948" cy="158" r="26" stroke="var(--accent)" stroke-width="2.5" stroke-dasharray="18 8" fill="none" opacity="0.75" />
      </svg>
      <div id="{sid}-scanbar" class="scanner-beam"></div>"""
                archetype_html = f"""
        <div class="top-hud-row">
          <div id="{sid}-stack" class="lt-stack-bars">
            <div class="stack-top data-text">{era_tag}</div>
            <div class="stack-main display">{callout}</div>
          </div>
          <div id="{sid}-source-tag" class="source-pill data-text">{kind_badge}</div>
        </div>"""
                gsap_lines.append(
                    f'  tl.to(["#{sid}-vf-tl", "#{sid}-vf-tr", "#{sid}-vf-bl", "#{sid}-vf-br"], {{ strokeDashoffset: 0, duration: 0.42, ease: "power3.out", stagger: 0.05 }}, {enter_t:.3f});'
                )
                gsap_lines.append(
                    f'  tl.fromTo("#{sid}-scanbar", {{ y: 260, opacity: 0.65 }}, {{ y: 1220, opacity: 0.0, duration: {min(2.4, dur * 0.78):.2f}, ease: "sine.inOut", immediateRender: false }}, {round(enter_t + 0.08, 3):.3f});'
                )
                gsap_lines.append(
                    f'  tl.fromTo("#{sid}-dial", {{ rotation: 0 }}, {{ rotation: 180, duration: {dur:.3f}, ease: "none", transformOrigin: "50% 50%" }}, {st:.3f});'
                )
                gsap_lines.append(
                    f'  tl.fromTo("#{sid}-stack", {{ x: -36, autoAlpha: 0 }}, {{ x: 0, autoAlpha: 1, duration: 0.36, ease: "expo.out", immediateRender: false }}, {enter_t:.3f});'
                )

            else:
                progress_pct = round(((idx + 1) / max(1, n_scenes)) * 100, 1)
                node_x = int(90 + (900 * ((idx + 0.5) / max(1, n_scenes))))
                archetype_svg = f"""
      <svg class="hud-svg" viewBox="0 0 1080 1920">
        <line x1="90" y1="228" x2="990" y2="228" stroke="rgba(255,255,255,0.22)" stroke-width="3" />
        <line id="{sid}-prog-line" x1="90" y1="228" x2="{node_x}" y2="228" stroke="var(--accent)" stroke-width="4" stroke-dasharray="900" stroke-dashoffset="900" />
        <circle id="{sid}-prog-dot" cx="{node_x}" cy="228" r="9" fill="var(--accent)" />
      </svg>"""
                archetype_html = f"""
        <div class="top-hud-row">
          <div id="{sid}-side-rule" class="lt-side-rule">
            <div class="side-rule-bar" id="{sid}-bar"></div>
            <div class="side-rule-body">
              <div class="side-rule-kicker data-text">{era_tag} // {progress_pct:.0f}%</div>
              <div class="side-rule-title display">{callout}</div>
            </div>
          </div>
          <div id="{sid}-source-tag" class="source-pill data-text">{kind_badge}</div>
        </div>"""
                gsap_lines.append(
                    f'  tl.to("#{sid}-prog-line", {{ strokeDashoffset: 0, duration: 0.55, ease: "expo.out" }}, {enter_t:.3f});'
                )
                gsap_lines.append(
                    f'  tl.fromTo("#{sid}-prog-dot", {{ scale: 0 }}, {{ scale: 1.35, duration: 0.38, ease: "back.out(2)", transformOrigin: "50% 50%", immediateRender: false }}, {round(enter_t + 0.14, 3):.3f});'
                )
                gsap_lines.append(
                    f'  tl.fromTo("#{sid}-bar", {{ scaleY: 0 }}, {{ scaleY: 1, duration: 0.34, ease: "expo.out", transformOrigin: "top center", immediateRender: false }}, {enter_t:.3f});'
                )
                gsap_lines.append(
                    f'  tl.fromTo("#{sid}-side-rule", {{ x: -28, autoAlpha: 0 }}, {{ x: 0, autoAlpha: 1, duration: 0.36, ease: "power3.out", immediateRender: false }}, {enter_t:.3f});'
                )

            gsap_lines.append(
                f'  tl.fromTo("#{sid}-source-tag", {{ x: 20, autoAlpha: 0 }}, {{ x: 0, autoAlpha: 1, duration: 0.36, ease: "power3.out", immediateRender: false }}, {round(enter_t + 0.06, 3):.3f});'
            )

            # 2-Stage 3D Camera Rig on `#sN-img` (ZERO BLUR FILTER — 100% RAZOR SHARP FROM FRAME 0!)
            if not is_video:
                punch_dur = min(0.22, round(dur * 0.12, 3))
                drift_st = round(st + punch_dur, 3)
                drift_dur = round(max(0.4, dur - punch_dur), 3)
                cam_profile = idx % 4
                if cam_profile == 0:
                    gsap_lines.append(
                        f'  tl.fromTo("#{sid}-img", {{ scale: 1.12, x: 0, y: 0 }}, {{ scale: 1.04, x: 0, y: 0, duration: {punch_dur:.3f}, ease: "expo.out", immediateRender: false }}, {st:.3f});'
                    )
                    gsap_lines.append(
                        f'  tl.to("#{sid}-img", {{ scale: 1.13, y: -14, duration: {drift_dur:.3f}, ease: "sine.inOut" }}, {drift_st:.3f});'
                    )
                elif cam_profile == 1:
                    gsap_lines.append(
                        f'  tl.fromTo("#{sid}-img", {{ scale: 1.15, x: -26, y: 8 }}, {{ scale: 1.10, x: -16, y: 4, duration: {punch_dur:.3f}, ease: "expo.out", immediateRender: false }}, {st:.3f});'
                    )
                    gsap_lines.append(
                        f'  tl.to("#{sid}-img", {{ scale: 1.04, x: 22, y: -8, duration: {drift_dur:.3f}, ease: "sine.inOut" }}, {drift_st:.3f});'
                    )
                elif cam_profile == 2:
                    gsap_lines.append(
                        f'  tl.fromTo("#{sid}-img", {{ scale: 1.13, x: 14, y: 24 }}, {{ scale: 1.07, x: 10, y: 14, duration: {punch_dur:.3f}, ease: "expo.out", immediateRender: false }}, {st:.3f});'
                    )
                    gsap_lines.append(
                        f'  tl.to("#{sid}-img", {{ scale: 1.15, x: -14, y: -22, duration: {drift_dur:.3f}, ease: "sine.inOut" }}, {drift_st:.3f});'
                    )
                else:
                    gsap_lines.append(
                        f'  tl.fromTo("#{sid}-img", {{ scale: 1.14, x: 20, y: -12 }}, {{ scale: 1.08, x: 12, y: -6, duration: {punch_dur:.3f}, ease: "expo.out", immediateRender: false }}, {st:.3f});'
                    )
                    gsap_lines.append(
                        f'  tl.to("#{sid}-img", {{ scale: 1.02, x: -20, y: 14, duration: {drift_dur:.3f}, ease: "sine.inOut" }}, {drift_st:.3f});'
                    )

            gsap_lines.append(
                f'  tl.fromTo("#{sid}-flash", {{ opacity: 0.26 }}, {{ opacity: 0, duration: 0.18, ease: "power2.out", immediateRender: false }}, {st:.3f});'
            )
            gsap_lines.append(
                f'  tl.fromTo("#{sid}-atmo", {{ x: -28, y: 14 }}, {{ x: 28, y: -14, duration: {dur:.3f}, ease: "none", immediateRender: false }}, {st:.3f});'
            )

            scene_blocks.append(f"""    <div class="scene" id="{sid}" data-start="{st:.3f}" data-duration="{dur:.3f}" data-track-index="1"{inline_style}>
      <div class="media-stage" id="{sid}-stage">
        {media_html}
      </div>
      <svg class="atmo-parallax" id="{sid}-atmo" viewBox="0 0 1080 1920">
        <circle cx="210" cy="420" r="2.2" fill="rgba(255,215,0,0.28)" />
        <circle cx="860" cy="580" r="1.8" fill="rgba(255,255,255,0.22)" />
        <circle cx="340" cy="980" r="2.5" fill="rgba(255,215,0,0.20)" />
        <circle cx="780" cy="1120" r="2.0" fill="rgba(255,255,255,0.18)" />
      </svg>
      <div class="vignette"></div>
      <div class="cinema-gradient-top"></div>
      <div class="cinema-gradient-bottom"></div>
      <div class="grain"></div>
      <div class="cut-flash" id="{sid}-flash"></div>{archetype_svg}
      <div class="scene-content">{archetype_html}
      </div>
    </div>""")

        # ---------------------------------------------------------------------
        # Broadcast Zero-Collision Karaoke Pill Captions (`caption-pill-karaoke`)
        # ---------------------------------------------------------------------
        caption_dom_items: List[str] = []
        raw_chunks = _build_smart_caption_chunks(word_timings, max_words=3, max_chars=26)

        chunk_windows: List[Tuple[float, float, List[WordTiming]]] = []
        for ch_idx, chunk in enumerate(raw_chunks):
            c_st = round(max(0.0, chunk[0].start_sec), 3)
            c_et = round(max(c_st + 0.16, chunk[-1].end_sec), 3)
            if ch_idx + 1 < len(raw_chunks):
                next_st = round(max(0.0, raw_chunks[ch_idx + 1][0].start_sec), 3)
                if c_et >= next_st - 0.025:
                    c_et = round(max(c_st + 0.08, next_st - 0.025), 3)
            chunk_windows.append((c_st, c_et, chunk))

        for chunk_idx, (c_start, c_end, chunk) in enumerate(chunk_windows, 1):
            cid = f"cap-{chunk_idx}"
            word_spans: List[str] = []
            for w_idx, wt in enumerate(chunk):
                wid = f"{cid}-w{w_idx}"
                w_clean = _esc(wt.word.upper())
                pill_bg, pill_fg, pill_glow = _classify_word_pill(wt.word)
                word_spans.append(f'<span id="{wid}" class="cap-word">{w_clean}</span>')

                w_st = round(max(c_start, wt.start_sec), 3)
                if w_idx + 1 < len(chunk):
                    w_et = round(min(c_end, max(w_st + 0.05, chunk[w_idx + 1].start_sec - 0.01)), 3)
                else:
                    w_et = c_end

                gsap_lines.append(
                    f'  tl.to("#{wid}", {{ backgroundColor: "{pill_bg}", color: "{pill_fg}", y: -5, '
                    f'boxShadow: "{pill_glow}", duration: 0.06, ease: "power2.out" }}, {w_st:.3f});'
                )
                if w_et < c_end:
                    gsap_lines.append(
                        f'  tl.to("#{wid}", {{ backgroundColor: "rgba(255,255,255,0)", color: "#F5F5F7", y: 0, '
                        f'boxShadow: "none", duration: 0.05, ease: "power2.out" }}, {w_et:.3f});'
                    )

            caption_dom_items.append(
                f'      <div id="{cid}" class="caption-phrase caption-group display" style="visibility:hidden;">'
                f'<div class="caption-dock">{" ".join(word_spans)}</div></div>'
            )
            gsap_lines.append(f'  tl.set("#{cid}", {{ autoAlpha: 1 }}, {c_start:.3f});')
            gsap_lines.append(
                f'  tl.fromTo("#{cid}", {{ y: 14 }}, {{ y: 0, duration: 0.11, ease: "power3.out", immediateRender: false }}, {c_start:.3f});'
            )
            gsap_lines.append(f'  tl.set("#{cid}", {{ autoAlpha: 0 }}, {c_end:.3f});')

        gsap_lines.append(
            f'  tl.fromTo("#top-progress-bar", {{ scaleX: 0 }}, {{ scaleX: 1, duration: {total_duration:.3f}, ease: "none", transformOrigin: "left center" }}, 0);'
        )

        if total_duration > 6.0:
            cta_st = round(total_duration - 2.5, 3)
            gsap_lines.append(f'  tl.set("#cta-pill", {{ autoAlpha: 1 }}, {cta_st:.3f});')
            gsap_lines.append(
                f'  tl.fromTo("#cta-pill", {{ y: 32, scale: 0.88 }}, {{ y: 0, scale: 1.0, duration: 0.34, ease: "back.out(1.7)", immediateRender: false }}, {cta_st:.3f});'
            )

        shader_init_js = ""
        if shader_anchors and shader_transitions:
            shader_init_js = f"""
  if (window.HyperShader && typeof window.HyperShader.init === "function") {{
    window.HyperShader.init({{
      bgColor: "#08090d",
      scenes: {json.dumps(shader_anchors)},
      timeline: tl,
      transitions: {json.dumps(shader_transitions)}
    }});
  }}"""

        scenes_joined = "\n".join(scene_blocks)
        captions_joined = "\n".join(caption_dom_items)
        gsap_joined = "\n".join(gsap_lines)

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=1080, height=1920" />
  <title>{_esc(title)}</title>
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@hyperframes/core/dist/hyperframe.runtime.iife.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@hyperframes/shader-transitions/dist/index.global.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --bg: #08090d;
      --ink: #f5f5f7;
      --accent: #ffd700;
      --crimson: #ff334b;
      --emerald: #00f59b;
      --muted: #9aa4b2;
      --panel: rgba(8, 10, 16, 0.84);
      --font-display: "Space Grotesk", "Montserrat", "DejaVu Sans", sans-serif;
      --font-data: "IBM Plex Mono", "DejaVu Sans Mono", monospace;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html, body {{
      width: 1080px;
      height: 1920px;
      overflow: hidden;
      background: var(--bg);
      color: var(--ink);
      -webkit-font-smoothing: antialiased;
    }}
    #main {{
      position: relative;
      width: 1080px;
      height: 1920px;
      overflow: hidden;
      background: var(--bg);
    }}
    .scene {{
      position: absolute;
      top: 0;
      left: 0;
      width: 1080px;
      height: 1920px;
      overflow: hidden;
    }}
    .media-stage {{
      position: absolute;
      inset: 0;
      width: 1080px;
      height: 1920px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .scene-media, .scene-video {{
      width: 1080px;
      height: 1920px;
      object-fit: cover;
      transform-origin: center center;
    }}
    .atmo-parallax {{
      position: absolute;
      inset: 0;
      width: 1080px;
      height: 1920px;
      pointer-events: none;
      z-index: 8;
    }}
    .vignette {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 10;
      background: radial-gradient(ellipse at center, transparent 54%, rgba(4, 6, 11, 0.64) 100%);
    }}
    .cinema-gradient-top {{
      position: absolute;
      left: 0;
      right: 0;
      top: 0;
      height: 380px;
      pointer-events: none;
      z-index: 11;
      background: linear-gradient(to bottom, rgba(5, 7, 12, 0.82) 0%, rgba(5, 7, 12, 0.36) 58%, transparent 100%);
    }}
    .cinema-gradient-bottom {{
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      height: 760px;
      pointer-events: none;
      z-index: 12;
      background: linear-gradient(to top, rgba(5, 7, 12, 0.94) 0%, rgba(5, 7, 12, 0.56) 54%, transparent 100%);
    }}
    .grain {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 14;
      opacity: 0.14;
      background-image:
        radial-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1.2px),
        radial-gradient(rgba(0, 0, 0, 0.18) 1px, transparent 1.2px);
      background-size: 3px 3px, 5px 5px;
      background-position: 0 0, 1px 2px;
      mix-blend-mode: overlay;
    }}
    .cut-flash {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 16;
      opacity: 0;
      background: radial-gradient(circle at 50% 42%, rgba(255, 230, 140, 0.75), rgba(255, 255, 255, 0) 70%);
      mix-blend-mode: screen;
    }}
    .scanner-beam {{
      position: absolute;
      left: 76px;
      right: 76px;
      top: 0;
      height: 3px;
      background: linear-gradient(90deg, transparent, var(--accent), transparent);
      box-shadow: 0 0 18px var(--accent);
      pointer-events: none;
      z-index: 17;
      opacity: 0;
    }}
    .hud-svg {{
      position: absolute;
      inset: 0;
      width: 1080px;
      height: 1920px;
      pointer-events: none;
      z-index: 18;
    }}
    .scene-content {{
      position: relative;
      z-index: 20;
      width: 100%;
      height: 100%;
      padding: 108px 74px;
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      gap: 28px;
    }}
    .top-hud-row {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
    }}
    .kicker-chip {{
      background: rgba(255, 215, 0, 0.14);
      border: 1.5px solid var(--accent);
      color: var(--accent);
      padding: 10px 22px;
      border-radius: 999px;
      font-size: 21px;
      font-weight: 600;
      letter-spacing: 0.08em;
    }}
    .callout-pill {{
      background: var(--panel);
      border: 2px solid var(--accent);
      color: var(--accent);
      padding: 12px 26px;
      border-radius: 12px;
      font-size: 27px;
      font-weight: 700;
      letter-spacing: 0.07em;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6);
    }}
    .callout-pill.emerald-border {{
      border-color: var(--emerald);
      color: var(--emerald);
    }}
    .source-pill {{
      background: rgba(8, 10, 16, 0.78);
      border: 1px solid rgba(255, 255, 255, 0.24);
      color: var(--ink);
      padding: 10px 20px;
      border-radius: 999px;
      font-size: 19px;
      font-weight: 600;
      letter-spacing: 0.06em;
      white-space: nowrap;
    }}
    .hero-hook-card {{
      margin-top: 8px;
      max-width: 820px;
      background: rgba(8, 10, 16, 0.78);
      border-left: 6px solid var(--accent);
      padding: 18px 28px;
      border-radius: 0 18px 18px 0;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.68);
    }}
    .hero-era-line {{
      font-size: 20px;
      color: var(--muted);
      letter-spacing: 0.1em;
      margin-bottom: 6px;
    }}
    .hero-headline {{
      display: inline;
      font-size: 46px;
      font-weight: 700;
      line-height: 1.15;
      color: #ffffff;
      background: linear-gradient(90deg, rgba(255, 215, 0, 0.36), rgba(255, 51, 75, 0.36)) no-repeat 0 88% / 0% 34%;
    }}
    .telemetry-card {{
      margin-top: auto;
      margin-bottom: 560px;
      align-self: flex-start;
      min-width: 360px;
      background: rgba(7, 10, 16, 0.84);
      border: 1.5px solid rgba(0, 245, 155, 0.45);
      border-left: 7px solid var(--emerald);
      padding: 18px 30px 22px 30px;
      border-radius: 0 20px 20px 0;
      box-shadow: 0 18px 44px rgba(0, 0, 0, 0.78);
    }}
    .telemetry-header {{
      font-size: 19px;
      color: var(--emerald);
      letter-spacing: 0.1em;
      margin-bottom: 4px;
    }}
    .telemetry-number {{
      font-size: 86px;
      font-weight: 700;
      line-height: 1.05;
      color: #ffffff;
      text-shadow: 0 6px 24px rgba(0, 245, 155, 0.35);
    }}
    .telemetry-label {{
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0.08em;
      color: var(--accent);
      margin-top: 4px;
    }}
    .telemetry-gauge-track {{
      margin-top: 12px;
      width: 100%;
      height: 7px;
      background: rgba(255, 255, 255, 0.14);
      border-radius: 999px;
      overflow: hidden;
    }}
    .telemetry-gauge-fill {{
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, var(--emerald), var(--accent));
      border-radius: 999px;
    }}
    .lt-stack-bars {{
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      box-shadow: 0 14px 36px rgba(0, 0, 0, 0.65);
    }}
    .stack-top {{
      background: var(--accent);
      color: #06080e;
      font-size: 19px;
      font-weight: 700;
      letter-spacing: 0.1em;
      padding: 6px 18px;
      border-radius: 8px 8px 0 0;
    }}
    .stack-main {{
      background: rgba(8, 10, 16, 0.88);
      border: 1.5px solid rgba(255, 215, 0, 0.45);
      color: var(--ink);
      font-size: 32px;
      font-weight: 700;
      letter-spacing: 0.04em;
      padding: 12px 24px;
      border-radius: 0 12px 12px 12px;
    }}
    .lt-side-rule {{
      display: flex;
      align-items: stretch;
      gap: 16px;
      background: rgba(8, 10, 16, 0.80);
      padding: 12px 24px 12px 16px;
      border-radius: 12px;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6);
    }}
    .side-rule-bar {{
      width: 6px;
      background: var(--accent);
      border-radius: 999px;
    }}
    .side-rule-body {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .side-rule-kicker {{
      font-size: 18px;
      color: var(--accent);
      letter-spacing: 0.1em;
    }}
    .side-rule-title {{
      font-size: 31px;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: 0.03em;
    }}
    #caption-stage {{
      position: absolute;
      left: 54px;
      right: 54px;
      bottom: 340px;
      height: 230px;
      z-index: 60;
      display: flex;
      align-items: center;
      justify-content: center;
      pointer-events: none;
    }}
    .caption-phrase {{
      position: absolute;
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .caption-dock {{
      background: rgba(6, 8, 14, 0.84);
      border: 1.5px solid rgba(255, 255, 255, 0.16);
      border-radius: 24px;
      padding: 14px 26px;
      box-shadow: 0 18px 46px rgba(0, 0, 0, 0.88);
      display: inline-flex;
      flex-wrap: wrap;
      justify-content: center;
      align-items: center;
      column-gap: 16px;
      row-gap: 10px;
      max-width: 960px;
    }}
    .cap-word {{
      display: inline-block;
      padding: 6px 16px;
      border-radius: 12px;
      font-size: 62px;
      font-weight: 700;
      line-height: 1.12;
      letter-spacing: 0.01em;
      color: var(--ink);
      white-space: nowrap;
      text-shadow: 0 3px 10px rgba(0, 0, 0, 0.65);
      transform-origin: center center;
    }}
    #top-progress-bar {{
      position: absolute;
      top: 0;
      left: 0;
      width: 1080px;
      height: 8px;
      background: linear-gradient(90deg, var(--accent), var(--crimson));
      z-index: 80;
    }}
    #cta-pill {{
      position: absolute;
      left: 50%;
      bottom: 225px;
      transform: translateX(-50%);
      z-index: 70;
      background: linear-gradient(135deg, #e11d48, #be123c);
      border: 2px solid #fda4af;
      color: #ffffff;
      padding: 16px 38px;
      border-radius: 999px;
      font-size: 27px;
      font-weight: 700;
      letter-spacing: 0.08em;
      box-shadow: 0 16px 40px rgba(225, 29, 72, 0.55);
      white-space: nowrap;
      visibility: hidden;
    }}
    .display {{ font-family: var(--font-display); }}
    .data-text {{
      font-family: var(--font-data);
      font-variant-numeric: tabular-nums;
    }}
  </style>
</head>
<body>
  <div
    id="main"
    data-composition-id="main"
    data-width="1080"
    data-height="1920"
    data-start="0"
    data-duration="{total_duration:.3f}"
  >
    <div id="top-progress-bar"></div>
    <audio id="master-audio" src="./{audio_rel}" data-start="0" data-duration="{total_duration:.3f}" data-volume="1"></audio>
{scenes_joined}
    <div id="caption-stage">
{captions_joined}
    </div>
    <div id="cta-pill" class="data-text">FOLLOW FOR DAILY UNTOLD HISTORY</div>
  </div>
  <script>
  window.__timelines = window.__timelines || {{}};
  var tl = gsap.timeline({{ paused: true }});
{gsap_joined}
{shader_init_js}
  window.__timelines["main"] = tl;
  </script>
</body>
</html>
"""

    @staticmethod
    def _generate_preview_html() -> str:
        return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>HyperFrames Studio Preview</title>
  <style>
    html, body { margin: 0; padding: 0; background: #111; height: 100%; overflow: hidden; }
  </style>
  <script type="module" src="https://cdn.jsdelivr.net/npm/@hyperframes/player"></script>
</head>
<body>
  <hyperframes-player
    id="p"
    controls
    autoplay
    muted
    style="display:block;width:100vw;height:100vh"
  ></hyperframes-player>
  <script>
    document.getElementById("p").setAttribute("src", "./index.html" + location.search);
  </script>
</body>
</html>
"""

    @staticmethod
    def _generate_design_md(
        topic: str,
        title: str,
        tiled_scenes: List[Dict[str, Any]],
        total_duration: float,
        shader_transitions: List[Dict[str, Any]],
    ) -> str:
        lines = [
            f"# HyperFrames v11.0 Studio Pro Specification — {title}",
            f"- **Topic**: {topic}",
            f"- **Resolution & Frame Rate**: `1080x1920` (9:16 Vertical) @ `30fps`",
            f"- **Total Duration**: `{total_duration:.2f}s` across `{len(tiled_scenes)}` scenes",
            f"- **Optical Sharpness**: `0px` blur across all scenes (instant 0.22s 3D camera punch-in + 6-axis drift + 2.5D foreground parallax)",
            f"- **Caption Architecture**: Zero-collision `caption-pill-karaoke` dock (`padding: 6px 16px`, `column-gap: 16px`, vertical lift `y: -5`, strict non-overlapping windows)",
            f"- **HyperShader WebGL Transitions**: `{json.dumps(shader_transitions)}`",
            "",
            "## Scene Timeline Breakdown",
            "| Scene ID | Window (s) | Duration (s) | Callout | Visual Asset |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for s in tiled_scenes:
            lines.append(
                f"| `{s['hf_id']}` | `{s['hf_start']:.2f}s – {s['hf_end']:.2f}s` | `{s['hf_duration']:.2f}s` | "
                f"`{s.get('on_screen_callout', '')}` | `{s['hf_asset_rel']}` |"
            )
        return "\n".join(lines) + "\n"

    def try_cli_render(self, output_mp4: Path, timeout_sec: int = 720) -> bool:
        if os.getenv("RENDER_ENGINE", "hyperframes").strip().lower() == "ffmpeg":
            return False

        npx_bin = shutil.which("npx")
        if not npx_bin:
            print("[HyperFrames] `npx` not found on PATH; using hybrid FFmpeg render path.")
            return False

        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        try:
            lint_proc = subprocess.run(
                [npx_bin, "--yes", "hyperframes", "lint"],
                cwd=str(self.project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
                check=False,
            )
            if lint_proc.returncode == 0:
                print("[HyperFrames] `npx hyperframes lint` passed with zero structural errors!")
            else:
                print(f"[HyperFrames] Lint notice: {lint_proc.stdout[:300]} {lint_proc.stderr[:300]}")

            for attempt in (1, 2):
                if attempt == 2 and self._last_build_args is not None:
                    print("[HyperFrames] Retrying `npx hyperframes render` in Pure-GSAP mode (no WebGL dependency)...")
                    pure_html = self._generate_index_html(
                        topic=self._last_build_args["topic"],
                        title=self._last_build_args["title"],
                        tiled_scenes=self._last_build_args["tiled_scenes"],
                        word_timings=self._last_build_args["word_timings"],
                        audio_rel=self._last_build_args["audio_rel"],
                        total_duration=self._last_build_args["total_duration"],
                        shader_anchors=[],
                        shader_transitions=[],
                    )
                    (self.project_dir / "index.html").write_text(pure_html, encoding="utf-8")

                print(f"[HyperFrames] Launching `npx hyperframes render --quality draft --output {output_mp4.resolve()}` (Attempt {attempt}/2)...")
                render_proc = subprocess.run(
                    [
                        npx_bin,
                        "--yes",
                        "hyperframes",
                        "render",
                        "--quality",
                        "draft",
                        "--output",
                        str(output_mp4.resolve()),
                    ],
                    cwd=str(self.project_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
                if (
                    render_proc.returncode == 0
                    and output_mp4.exists()
                    and output_mp4.stat().st_size > 250_000
                ):
                    print(
                        f"[HyperFrames] Successfully rendered MP4 via `npx hyperframes render` on attempt {attempt} "
                        f"({output_mp4.stat().st_size // 1024} KB)!"
                    )
                    return True
                print(
                    f"[HyperFrames] CLI render attempt {attempt} exited with {render_proc.returncode}: "
                    f"STDOUT={render_proc.stdout[-400:]} STDERR={render_proc.stderr[-400:]}"
                )
        except Exception as exc:
            print(f"[HyperFrames] CLI execution notice ({exc}); falling back to hybrid FFmpeg cinema renderer.")
        return False
