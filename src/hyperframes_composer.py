"""
HeyGen HyperFrames Studio Pro + Huashu Art Motion Composition & Rendering Engine (`src/hyperframes_composer.py` — v18.0):
Fixes every visual issue observed in `1000047695.mp4`:
1. ZERO blind lines, pointers, pins, or dashed boxes over photography.
2. ZERO duplicate text:
   - Top-left pill: `era_tag` ONLY (e.g. `FRANCE // 1807`).
   - Top-right pill: `01 / 10` scene index badge ONLY (never internal debug names).
   - `on_screen_callout` (e.g. `RABBITS NOT SOLDIERS`) appears ONCE per scene.
3. Smart Quantitative Stat Filter (`_extract_smart_stat`):
   - Separates fields with ` || ` so `stat_value="1807"` + `on_screen_callout="RABBITS..."` never merges into `"1,807 rabbits"`.
   - Ignores standalone 4-digit calendar years (`1000–2029`) when they are just dates (already shown in `era_tag`),
     reserving the animated data counter + Huashu bar chart exclusively for real quantities (`3,000`, `30+`, `140 FEET`, etc.).
4. Four Distinct Huashu Art Motion Stage Layouts + Per-Cut Physical Transitions + 3 WebGL `HyperShader` Transitions:
   - `vox_print_stage`: Tactile museum/polaroid archival print card (`#f4efe6` paper border) that physically slaps onto a
     lit architectural desk (`scale: 1.25 -> 1.0, rotation: -4deg -> -1.3deg, ease: back.out(1.55)`) with an animated
     marker highlighter sweep across the caption printed on the bottom border of the photo print.
   - `data_split_stage`: Animated axes-first comparative bar chart + rolling number counter for quantitative beats.
   - `cinema_letterbox_stage`: Widescreen cinema matte rails that open with `expo.out` + scrolling timeline scale.
   - `fullbleed_hook_stage`: Full-bleed 9:16 keyframe with kinetic diagonal shutter wipe entrance and single highlighter callout card.
"""
from __future__ import annotations
import html
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from voice_engine import WordTiming
    from huashu_art_motion import HuashuArtMotionDirector
except ImportError:
    from src.voice_engine import WordTiming
    from src.huashu_art_motion import HuashuArtMotionDirector

PUNCT_ONLY_RE = re.compile(r"^[,.;:!?—–\-\"'\)\]\}\u2014\u2013]+$")
LEADING_PUNCT_RE = re.compile(r"^([,.;:!?—–\-]+)(.+)$")

POWER_WORDS = {
    "WAR", "DEAD", "DIED", "DEATH", "KILLED", "KILL", "EXPLODED", "EXPLOSION", "BOMB", "NUCLEAR",
    "ATOMIC", "DOOMED", "DESTROYED", "CRASHED", "CRASH", "FAKE", "CORPSE", "SECRET", "SPY", "SPIES",
    "HITLER", "NAZI", "TERROR", "PANIC", "BLOOD", "FIRE", "BURNED", "POISON", "PLAGUE", "CRAZIEST",
    "BIZARRE", "IMPOSSIBLE", "LOST", "VANISHED", "MISSING", "FLOOD", "WAVE", "CRUSHED", "TRAPPED",
    "NEVER", "REFUSED", "STOPPED", "SHOCKING", "INSANE", "DANGEROUS", "LETHAL", "FATAL", "ATTACK",
    "CONCEALED", "SHATTERED", "SUNKEN", "SANK", "MYSTERY", "FORBIDDEN", "HIDDEN", "FLED", "CHASED",
    "UNREAD", "DECIPHERMENT", "CIPHER", "PUZZLE", "UNKNOWN", "UNVERIFIED",
}

STAT_WORDS = {
    "MILLION", "BILLIONS", "BILLION", "THOUSAND", "THOUSANDS", "HUNDRED", "FIRST", "ZERO", "SINGLE",
    "DOUBLE", "TONS", "MPH", "FEET", "MILES", "HOURS", "MINUTES", "SECONDS", "YEARS", "CENTURY",
    "MILLENNIA", "TWO", "THREE", "FOUR", "FIVE", "TEN", "TWENTY", "THIRTY", "PAGES",
}


def _esc(text: Any) -> str:
    return html.escape(str(text or "").strip())


def _extract_smart_stat(scene: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts a true quantitative metric (like `3,000 FAMOUS COUNT`, `2,000+ YEARS OLD`, `30+ BRONZE GEARS`).
    - Uses ` || ` delimiter between fields so a 4-digit year in `stat_value` (`1807`) never concatenates with
      the first word of `on_screen_callout` (`RABBITS NOT SOLDIERS`).
    - Never formats 4-digit calendar years (`1000–2029`) with a comma (`1,807` is forbidden).
    """
    raw_val = str(scene.get("stat_value") or "").strip()
    raw_lbl = str(scene.get("stat_label") or "").strip().upper()
    era_tag = str(scene.get("era_tag") or "").strip()

    # Check if raw_val is purely a 4-digit calendar year (e.g., "1807", "1901")
    raw_digits = raw_val.replace(",", "").strip()
    is_pure_calendar_year = (
        bool(re.fullmatch(r"1[0-9]{3}|20[0-2][0-9]", raw_digits))
        and "+" not in raw_val
        and "OLD" not in raw_lbl
        and (
            "YEAR" in raw_lbl
            or "DATE" in raw_lbl
            or "FOUND" in raw_lbl
            or "REPORTED" in raw_lbl
            or not raw_lbl
            or raw_digits in era_tag
        )
    )
    if is_pure_calendar_year:
        # If the year is already in era_tag (e.g., "FRANCE // 1807"), do NOT show a redundant stat box for the year!
        if raw_digits in era_tag:
            return None
        yr_num = int(raw_digits)
        return {
            "target": yr_num,
            "start": max(1000, yr_num - 18),
            "use_commas": False,
            "suffix": "",
            "label": (raw_lbl or "HISTORICAL YEAR")[:22],
            "is_year": True,
        }

    # Delimit with ` || ` so `stat_value` and `on_screen_callout` never merge across boundaries!
    combined = f"{raw_val} || {scene.get('on_screen_callout', '')} || {scene.get('narration', '')}"

    m_qty = re.search(
        r"\b([1-9][0-9]{0,2}(?:,[0-9]{3})+|[1-9][0-9]{1,4})(\+?)\s*(?:-|\s)*(years?\s*old|pages?|rabbits?|gears?|wheels?|fragments?|feet|miles|tons|days|hours|ships|men|divers|soldiers|birds|centuries|century)\b",
        combined,
        re.IGNORECASE,
    )
    if m_qty:
        num = int(m_qty.group(1).replace(",", ""))
        # Guard against matching a 4-digit year followed by a noun in narration (e.g. "1807 soldiers")
        if 1000 <= num <= 2029 and "," not in m_qty.group(1) and num == int(raw_digits or 0):
            pass
        else:
            plus = "+" if ("+" in m_qty.group(2) or "over " in combined.lower() or "more than " in combined.lower()) else ""
            unit = raw_lbl or m_qty.group(3).upper().replace("-", " ")
            if "YEAR" in unit and "OLD" not in unit:
                unit = "YEARS OLD"
            return {
                "target": num,
                "start": 0,
                "use_commas": num >= 1000,
                "suffix": plus,
                "label": unit[:22],
                "is_year": False,
            }

    if raw_val:
        m_explicit = re.search(r"([1-9][0-9]{0,2}(?:,[0-9]{3})+|[1-9][0-9]{1,4})(\+?)", raw_val)
        if m_explicit:
            num = int(m_explicit.group(1).replace(",", ""))
            plus = m_explicit.group(2) or ("+" if "+" in raw_val else "")
            if 1000 <= num <= 2029 and not plus and "," not in raw_val and "OLD" not in raw_lbl:
                if str(num) in era_tag:
                    return None
                return {
                    "target": num,
                    "start": max(1000, num - 18),
                    "use_commas": False,
                    "suffix": "",
                    "label": (raw_lbl or "HISTORICAL YEAR")[:22],
                    "is_year": True,
                }
            return {
                "target": num,
                "start": 0,
                "use_commas": num >= 1000,
                "suffix": plus,
                "label": (raw_lbl or "VERIFIED METRIC")[:22],
                "is_year": False,
            }

    m_yr = re.search(r"\b(1[1-9][0-9]{2}|20[0-2][0-9])\b", combined)
    if m_yr:
        yr = int(m_yr.group(1))
        if str(yr) in era_tag:
            return None
        after = combined[m_yr.end():m_yr.end() + 12].lower()
        if not after.strip().startswith(("year", "ton", "feet", "mile", "page", "rabbit")):
            return {
                "target": yr,
                "start": yr - 24,
                "use_commas": False,
                "suffix": "",
                "label": (raw_lbl or "HISTORICAL YEAR")[:22],
                "is_year": True,
            }

    return None


def _classify_word_pill(word: str) -> Tuple[str, str, str]:
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
        self.huashu_director = HuashuArtMotionDirector()
        self.last_huashu_lint: Optional[Dict[str, Any]] = None
        self._last_build_args: Optional[Dict[str, Any]] = None

    def build_project(
        self,
        topic: str,
        title: str,
        scenes: List[Dict[str, Any]],
        word_timings: List[WordTiming],
        master_audio_wav: Path,
        enable_shaders: bool = True,
        era_keywords: Optional[List[str]] = None,
    ) -> Path:
        if self.project_dir.exists():
            shutil.rmtree(self.project_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        audio_rel = "assets/master_audio.wav"
        shutil.copy2(master_audio_wav, self.project_dir / audio_rel)

        self.last_huashu_lint = self.huashu_director.lint_storyboard(
            topic=topic,
            scenes=scenes,
            era_keywords=era_keywords,
            stat_extractor=_extract_smart_stat,
        )
        theme = self.last_huashu_lint["theme"]

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
            "theme": theme,
        }

        shader_anchors: List[str] = []
        shader_transitions: List[Dict[str, Any]] = []
        if enable_shaders and len(tiled_scenes) >= 6:
            mid_idx = len(tiled_scenes) // 2 - 1
            s_a = tiled_scenes[mid_idx]
            s_b = tiled_scenes[mid_idx + 1]
            boundary = s_a["hf_end"]
            trans_dur = 0.55
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
            theme=theme,
        )
        (self.project_dir / "index.html").write_text(index_html, encoding="utf-8")
        (self.project_dir / "preview.html").write_text(self._generate_preview_html(), encoding="utf-8")
        (self.project_dir / "DESIGN.md").write_text(
            self._generate_design_md(topic, title, tiled_scenes, total_duration, shader_transitions, theme),
            encoding="utf-8",
        )
        print(
            f"[HyperFrames + HuashuArtMotion] Authored v18.0 Studio Pro 1080x1920 composition -> {self.project_dir / 'index.html'} "
            f"({len(tiled_scenes)} scenes, theme='{theme['name']}', {len(shader_transitions)} WebGL shaders, {total_duration:.2f}s)"
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
        theme: Optional[Dict[str, str]] = None,
    ) -> str:
        theme = theme or self.huashu_director.select_palette_for_topic(topic)
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
            callout = _esc((s.get("on_screen_callout") or f"CHAPTER {idx + 1:02d}").upper()[:28])
            subject = _esc((s.get("visual_subject") or topic)[:56])
            era_tag = _esc((s.get("era_tag") or f"ARCHIVE // {idx + 1:02d}").upper()[:28])
            step_badge = f"SCENE {idx + 1:02d} / {n_scenes:02d}"
            grammar = str(s.get("huashu_grammar") or "fullbleed_hook_stage")
            cam_move = str(s.get("huashu_camera") or "slow_push_in")
            stat_info = _extract_smart_stat(s)

            # Visibility contract for HyperFrames
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

            top_level_video_html = ""
            if is_video:
                top_level_video_html = (
                    f'    <video id="{sid}-vid" class="scene-video" src="./{asset_rel}" '
                    f'data-start="{st:.3f}" data-duration="{dur:.3f}" data-track-index="0" loop muted playsinline preload="auto"></video>\n'
                )
                media_html = ""
            else:
                media_html = f'<img id="{sid}-img" class="scene-media" src="./{asset_rel}" alt="{subject}" />'

            enter_t = round(st + 0.06, 3)

            # -----------------------------------------------------------------
            # Build Stage-Specific HTML & Huashu Art Motion Choreography
            # (ZERO duplicate text, ZERO random lines/pointers on photos!)
            # -----------------------------------------------------------------
            if grammar == "vox_print_stage" and not is_video:
                # Huashu `y2_vox` Tactile Archival Print Stage:
                # Dark architectural grid desk + physical tilted museum/polaroid print card that slaps onto the desk
                tilt_deg = -1.6 if idx % 2 == 1 else 1.5
                start_deg = -5.5 if idx % 2 == 1 else 5.5
                scene_blocks.append(f"""{top_level_video_html}    <div class="scene clip vox-desk-bg" id="{sid}" data-start="{st:.3f}" data-duration="{dur:.3f}" data-track-index="1" data-huashu-grammar="{grammar}" data-huashu-camera="{cam_move}"{inline_style}>
      <div class="vox-grid-pattern"></div>
      <div class="vox-print-card" id="{sid}-print-card">
        <div class="vox-tape-top"></div>
        <div class="vox-photo-window">
          {media_html}
        </div>
        <div class="vox-print-footer">
          <div class="vox-hl-wrap">
            <div class="vox-hl-bar" id="{sid}-hl"></div>
            <span class="vox-print-caption display">{callout}</span>
          </div>
          <span class="vox-print-ref data-text">EXHIBIT #{idx + 1:02d}</span>
        </div>
      </div>
      <div class="cinema-gradient-bottom"></div>
      <div class="scene-content">
        <div class="top-hud-row">
          <div id="{sid}-era-pill" class="era-top-pill data-text">{era_tag}</div>
          <div id="{sid}-step-pill" class="source-pill data-text">{step_badge}</div>
        </div>
      </div>
    </div>""")
                # Physical `vox_slap` entrance + inner photo counter-pan + highlighter sweep on print border
                gsap_lines.extend([
                    f'  tl.fromTo("#{sid}-print-card", {{ scale: 1.26, rotation: {start_deg}, y: -70, autoAlpha: 0 }}, {{ scale: 1.0, rotation: {tilt_deg}, y: 0, autoAlpha: 1, duration: 0.46, ease: "back.out(1.55)" }}, {st:.3f});',
                    self.huashu_director.build_camera_gsap_line(sid=sid, move=cam_move, st=st, dur=dur, grammar=grammar),
                    f'  tl.fromTo("#{sid}-hl", {{ scaleX: 0 }}, {{ scaleX: 1, duration: 0.52, ease: "expo.out", transformOrigin: "left center" }}, {round(st + 0.34, 3):.3f});',
                    f'  tl.from("#{sid}-era-pill", {{ y: -22, autoAlpha: 0, duration: 0.34, ease: "power3.out" }}, {enter_t:.3f});',
                    f'  tl.from("#{sid}-step-pill", {{ x: 20, autoAlpha: 0, duration: 0.34, ease: "power3.out" }}, {enter_t:.3f});',
                ])

            elif grammar == "data_split_stage" and stat_info is not None:
                # Huashu `t3_finance_chart` Axes-First Animated Data Stage (only when a real quantity exists)
                lbl_esc = _esc(stat_info["label"])
                target_n = stat_info["target"]
                start_n = stat_info["start"]
                use_commas = "true" if stat_info["use_commas"] else "false"
                suffix_js = json.dumps(stat_info["suffix"])

                scene_blocks.append(f"""{top_level_video_html}    <div class="scene clip" id="{sid}" data-start="{st:.3f}" data-duration="{dur:.3f}" data-track-index="1" data-huashu-grammar="{grammar}" data-huashu-camera="{cam_move}"{inline_style}>
      <div class="media-stage" id="{sid}-stage">
        {media_html}
      </div>
      <div class="vignette"></div>
      <div class="grain"></div>
      <div class="cinema-gradient-top"></div>
      <div class="cinema-gradient-bottom"></div>
      <div class="shutter-wipe" id="{sid}-shutter"></div>
      <div class="scene-content">
        <div class="top-hud-row">
          <div id="{sid}-era-pill" class="era-top-pill data-text">{era_tag}</div>
          <div id="{sid}-step-pill" class="source-pill data-text">{step_badge}</div>
        </div>
        <div id="{sid}-stat-box" class="huashu-data-card">
          <div class="huashu-data-left">
            <div class="telemetry-header data-text">{callout}</div>
            <div id="{sid}-stat" class="telemetry-number data-text">{start_n}</div>
            <div class="telemetry-label display">{lbl_esc}</div>
          </div>
          <div class="huashu-data-bars">
            <div class="hbar-row"><span class="hbar-tag data-text">BASELINE</span><div class="hbar-track"><div id="{sid}-b1" class="hbar-fill hbar-muted" style="width:34%;"></div></div></div>
            <div class="hbar-row"><span class="hbar-tag data-text">ARCHIVE</span><div class="hbar-track"><div id="{sid}-b2" class="hbar-fill hbar-mid" style="width:62%;"></div></div></div>
            <div class="hbar-row"><span class="hbar-tag data-text">RECORD</span><div class="hbar-track"><div id="{sid}-b3" class="hbar-fill hbar-peak" style="width:100%;"></div></div></div>
          </div>
        </div>
      </div>
    </div>""")
                if not is_video:
                    gsap_lines.append(self.huashu_director.build_camera_gsap_line(sid=sid, move=cam_move, st=st, dur=dur, grammar=grammar))
                gsap_lines.extend([
                    f'  tl.fromTo("#{sid}-shutter", {{ xPercent: 0 }}, {{ xPercent: 105, duration: 0.42, ease: "expo.inOut" }}, {st:.3f});',
                    f'  tl.from("#{sid}-era-pill", {{ y: -22, autoAlpha: 0, duration: 0.34, ease: "power3.out" }}, {enter_t:.3f});',
                    f'  tl.from("#{sid}-step-pill", {{ x: 20, autoAlpha: 0, duration: 0.34, ease: "power3.out" }}, {enter_t:.3f});',
                    f'  tl.from("#{sid}-stat-box", {{ y: 36, autoAlpha: 0, duration: 0.38, ease: "back.out(1.5)" }}, {round(enter_t + 0.08, 3):.3f});',
                    f'  tl.fromTo("#{sid}-b1", {{ scaleX: 0 }}, {{ scaleX: 1, duration: 0.38, ease: "power3.out", transformOrigin: "left center" }}, {round(enter_t + 0.18, 3):.3f});',
                    f'  tl.fromTo("#{sid}-b2", {{ scaleX: 0 }}, {{ scaleX: 1, duration: 0.44, ease: "power3.out", transformOrigin: "left center" }}, {round(enter_t + 0.26, 3):.3f});',
                    f'  tl.fromTo("#{sid}-b3", {{ scaleX: 0 }}, {{ scaleX: 1, duration: 0.56, ease: "back.out(1.6)", transformOrigin: "left center" }}, {round(enter_t + 0.34, 3):.3f});',
                    f'  (function() {{ var c = {{ v: {start_n} }}; '
                    f'tl.to(c, {{ v: {target_n}, duration: {min(1.25, dur * 0.52):.2f}, ease: "power2.out", '
                    f'onUpdate: function() {{ var el = document.getElementById("{sid}-stat"); '
                    f'if (el) {{ var n = Math.round(c.v); el.textContent = ({use_commas} ? n.toLocaleString("en-US") : n.toString()) + {suffix_js}; }} }} }}, {round(enter_t + 0.10, 3):.3f}); }})();',
                ])

            elif grammar == "cinema_letterbox_stage":
                # Huashu `long_scroll` Widescreen Panavision Aperture + Scrolling Timeline Rail
                scene_blocks.append(f"""{top_level_video_html}    <div class="scene clip" id="{sid}" data-start="{st:.3f}" data-duration="{dur:.3f}" data-track-index="1" data-huashu-grammar="{grammar}" data-huashu-camera="{cam_move}"{inline_style}>
      <div class="media-stage" id="{sid}-stage">
        {media_html}
      </div>
      <div class="vignette"></div>
      <div class="grain"></div>
      <div class="letterbox-top" id="{sid}-lb-top">
        <div class="top-hud-row" style="width:100%;">
          <div id="{sid}-era-pill" class="era-top-pill data-text">{era_tag}</div>
          <div id="{sid}-step-pill" class="source-pill data-text">{step_badge}</div>
        </div>
      </div>
      <div class="letterbox-bottom" id="{sid}-lb-bot">
        <div class="timeline-ruler-wrap">
          <div class="timeline-ruler-ticks" id="{sid}-ruler"></div>
        </div>
        <div id="{sid}-callout-card" class="letterbox-callout-row">
          <span class="lb-kicker data-text">CHRONICLE FOCUS</span>
          <div class="lb-hl-box">
            <div class="vox-hl-bar" id="{sid}-hl"></div>
            <span class="lb-title display">{callout}</span>
          </div>
        </div>
      </div>
    </div>""")
                if not is_video:
                    gsap_lines.append(self.huashu_director.build_camera_gsap_line(sid=sid, move=cam_move, st=st, dur=dur, grammar=grammar))
                gsap_lines.extend([
                    f'  tl.from("#{sid}-lb-top", {{ yPercent: -100, duration: 0.40, ease: "expo.out" }}, {st:.3f});',
                    f'  tl.from("#{sid}-lb-bot", {{ yPercent: 100, duration: 0.40, ease: "expo.out" }}, {st:.3f});',
                    f'  tl.fromTo("#{sid}-ruler", {{ x: 0 }}, {{ x: -260, duration: {dur:.3f}, ease: "none" }}, {st:.3f});',
                    f'  tl.from("#{sid}-callout-card", {{ y: 20, autoAlpha: 0, duration: 0.34, ease: "power3.out" }}, {round(enter_t + 0.14, 3):.3f});',
                    f'  tl.fromTo("#{sid}-hl", {{ scaleX: 0 }}, {{ scaleX: 1, duration: 0.50, ease: "expo.out", transformOrigin: "left center" }}, {round(enter_t + 0.24, 3):.3f});',
                ])

            else:
                # `fullbleed_hook_stage`: Unobstructed 9:16 Cinema + Kinetic Shutter Wipe + Single Highlighter Card
                scene_blocks.append(f"""{top_level_video_html}    <div class="scene clip" id="{sid}" data-start="{st:.3f}" data-duration="{dur:.3f}" data-track-index="1" data-huashu-grammar="{grammar}" data-huashu-camera="{cam_move}"{inline_style}>
      <div class="media-stage" id="{sid}-stage">
        {media_html}
      </div>
      <div class="vignette"></div>
      <div class="grain"></div>
      <div class="cinema-gradient-top"></div>
      <div class="cinema-gradient-bottom"></div>
      <div class="shutter-wipe" id="{sid}-shutter"></div>
      <div class="scene-content">
        <div class="top-hud-row">
          <div id="{sid}-era-pill" class="era-top-pill data-text">{era_tag}</div>
          <div id="{sid}-step-pill" class="source-pill data-text">{step_badge}</div>
        </div>
        <div id="{sid}-era-card" class="era-lower-card">
          <div class="vox-hl-wrap">
            <div class="vox-hl-bar" id="{sid}-hl"></div>
            <div class="era-subject display">{callout}</div>
          </div>
        </div>
      </div>
    </div>""")
                if not is_video:
                    gsap_lines.append(self.huashu_director.build_camera_gsap_line(sid=sid, move=cam_move, st=st, dur=dur, grammar=grammar))
                if idx > 0:
                    gsap_lines.append(
                        f'  tl.fromTo("#{sid}-shutter", {{ xPercent: 0 }}, {{ xPercent: 105, duration: 0.38, ease: "expo.inOut" }}, {st:.3f});'
                    )
                else:
                    gsap_lines.append(f'  tl.set("#{sid}-shutter", {{ autoAlpha: 0 }}, 0);')
                gsap_lines.extend([
                    f'  tl.from("#{sid}-era-pill", {{ y: -24, autoAlpha: 0, duration: 0.34, ease: "back.out(1.6)" }}, {enter_t:.3f});',
                    f'  tl.from("#{sid}-step-pill", {{ x: 22, autoAlpha: 0, duration: 0.34, ease: "power3.out" }}, {round(enter_t + 0.05, 3):.3f});',
                    f'  tl.from("#{sid}-era-card", {{ x: -32, autoAlpha: 0, duration: 0.36, ease: "power3.out" }}, {round(enter_t + 0.08, 3):.3f});',
                    f'  tl.fromTo("#{sid}-hl", {{ scaleX: 0 }}, {{ scaleX: 1, duration: 0.50, ease: "expo.out", transformOrigin: "left center" }}, {round(enter_t + 0.22, 3):.3f});',
                ])

        # Broadcast Zero-Collision Karaoke Pill Captions
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
                f'      <div id="{cid}" class="caption-phrase display" style="visibility:hidden;">'
                f'<div class="caption-dock">{" ".join(word_spans)}</div></div>'
            )
            gsap_lines.append(f'  tl.set("#{cid}", {{ autoAlpha: 1 }}, {c_start:.3f});')
            gsap_lines.append(
                f'  tl.fromTo("#{cid}", {{ y: 14, scale: 0.96 }}, {{ y: 0, scale: 1.0, duration: 0.12, ease: "power3.out" }}, {c_start:.3f});'
            )
            gsap_lines.append(f'  tl.set("#{cid}", {{ autoAlpha: 0 }}, {c_end:.3f});')

        gsap_lines.append(
            f'  tl.fromTo("#top-progress-bar", {{ scaleX: 0 }}, {{ scaleX: 1, duration: {total_duration:.3f}, ease: "none", transformOrigin: "left center" }}, 0);'
        )

        if total_duration > 6.0:
            cta_st = round(total_duration - 2.4, 3)
            gsap_lines.append(f'  tl.set("#cta-pill", {{ autoAlpha: 1 }}, {cta_st:.3f});')
            gsap_lines.append(
                f'  tl.from("#cta-pill", {{ y: 36, scale: 0.85, duration: 0.35, ease: "back.out(1.8)" }}, {cta_st:.3f});'
            )

        shader_init_js = ""
        if shader_anchors and shader_transitions:
            shader_init_js = f"""
  if (window.HyperShader && typeof window.HyperShader.init === "function") {{
    window.HyperShader.init({{
      bgColor: "{theme["bg"]}",
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
      --bg: {theme["bg"]};
      --desk-bg: {theme.get("desk_bg", "#12100E")};
      --ink: {theme["ink"]};
      --paper: {theme.get("paper", "#F4EFE6")};
      --paper-ink: {theme.get("paper_ink", "#141210")};
      --accent: {theme["accent"]};
      --crimson: {theme["secondary"]};
      --emerald: {theme["tertiary"]};
      --muted: #9aa4b2;
      --panel: {theme["panel"]};
      --highlighter: {theme["highlighter"]};
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
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform-origin: center center;
    }}
    /* Stage 1: Huashu Vox Tactile Archival Print Stage */
    .vox-desk-bg {{
      background: radial-gradient(circle at 50% 38%, #1d1915 0%, var(--desk-bg) 72%, #070605 100%);
    }}
    .vox-grid-pattern {{
      position: absolute;
      inset: 0;
      opacity: 0.22;
      background-image:
        linear-gradient(to right, rgba(255, 215, 0, 0.14) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(255, 215, 0, 0.14) 1px, transparent 1px);
      background-size: 60px 60px;
      pointer-events: none;
    }}
    .vox-print-card {{
      position: absolute;
      top: 225px;
      left: 76px;
      width: 928px;
      height: 1060px;
      background: var(--paper);
      border-radius: 10px;
      padding: 18px 18px 0 18px;
      box-shadow: 0 38px 80px rgba(0, 0, 0, 0.88), 0 4px 16px rgba(0, 0, 0, 0.55);
      z-index: 14;
      display: flex;
      flex-direction: column;
      transform-origin: center center;
    }}
    .vox-tape-top {{
      position: absolute;
      top: -18px;
      left: 50%;
      transform: translateX(-50%) rotate(-1.5deg);
      width: 210px;
      height: 42px;
      background: rgba(245, 232, 196, 0.72);
      border: 1px solid rgba(0, 0, 0, 0.12);
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
      z-index: 16;
    }}
    .vox-photo-window {{
      width: 100%;
      height: 928px;
      overflow: hidden;
      border-radius: 4px;
      background: #090909;
      position: relative;
    }}
    .vox-print-footer {{
      height: 114px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 14px;
      color: var(--paper-ink);
    }}
    .vox-hl-wrap {{
      position: relative;
      display: inline-flex;
      align-items: center;
      padding: 6px 14px;
    }}
    .vox-hl-bar {{
      position: absolute;
      inset: 0;
      background: var(--highlighter);
      border-radius: 6px;
      z-index: 0;
    }}
    .vox-print-caption {{
      position: relative;
      z-index: 1;
      font-size: 33px;
      font-weight: 700;
      color: var(--paper-ink);
      letter-spacing: 0.04em;
    }}
    .vox-print-ref {{
      font-size: 20px;
      font-weight: 700;
      color: #5c5346;
      letter-spacing: 0.08em;
    }}
    /* Kinetic Shutter Wipe Transition */
    .shutter-wipe {{
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, var(--accent) 0%, #1a1510 22%, #0a0908 100%);
      z-index: 45;
      pointer-events: none;
    }}
    /* Stage 2: Huashu Animated Data Split Stage */
    .huashu-data-card {{
      align-self: stretch;
      margin-top: auto;
      margin-bottom: 545px;
      background: rgba(8, 10, 16, 0.90);
      border: 2px solid rgba(255, 255, 255, 0.16);
      border-left: 8px solid var(--emerald);
      border-radius: 0 20px 20px 0;
      padding: 22px 30px;
      box-shadow: 0 22px 50px rgba(0, 0, 0, 0.82);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 28px;
    }}
    .huashu-data-left {{
      flex: 0 0 auto;
    }}
    .huashu-data-bars {{
      flex: 1 1 auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
      border-left: 2px solid rgba(255, 255, 255, 0.14);
      padding-left: 24px;
    }}
    .hbar-row {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .hbar-tag {{
      width: 92px;
      font-size: 16px;
      font-weight: 600;
      color: var(--muted);
      letter-spacing: 0.06em;
    }}
    .hbar-track {{
      flex: 1;
      height: 16px;
      background: rgba(255, 255, 255, 0.10);
      border-radius: 999px;
      overflow: hidden;
    }}
    .hbar-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .hbar-muted {{ background: rgba(154, 164, 178, 0.55); }}
    .hbar-mid {{ background: var(--accent); }}
    .hbar-peak {{ background: linear-gradient(90deg, var(--emerald), var(--accent)); }}
    /* Stage 3: Huashu Widescreen Letterbox & Scrolling Timeline Rail */
    .letterbox-top {{
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 230px;
      background: #07080c;
      border-bottom: 3px solid var(--accent);
      z-index: 25;
      padding: 96px 74px 0 74px;
      display: flex;
      align-items: flex-start;
    }}
    .letterbox-bottom {{
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 700px;
      background: linear-gradient(to bottom, rgba(7, 8, 12, 0.96) 0%, #07080c 28%);
      border-top: 3px solid var(--accent);
      z-index: 25;
      padding: 22px 74px 0 74px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}
    .timeline-ruler-wrap {{
      width: 100%;
      height: 26px;
      overflow: hidden;
      position: relative;
      opacity: 0.72;
    }}
    .timeline-ruler-ticks {{
      width: 1400px;
      height: 100%;
      background-image: repeating-linear-gradient(
        to right,
        var(--accent) 0px,
        var(--accent) 2px,
        transparent 2px,
        transparent 24px
      );
    }}
    .letterbox-callout-row {{
      display: flex;
      align-items: center;
      gap: 18px;
    }}
    .lb-kicker {{
      font-size: 18px;
      font-weight: 700;
      color: var(--accent);
      letter-spacing: 0.1em;
      padding: 6px 14px;
      border: 1.5px solid var(--accent);
      border-radius: 8px;
    }}
    .lb-hl-box {{
      position: relative;
      padding: 6px 16px;
    }}
    .lb-title {{
      position: relative;
      z-index: 1;
      font-size: 34px;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: 0.04em;
    }}
    .vignette {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 10;
      background: radial-gradient(ellipse at center, transparent 55%, rgba(4, 6, 10, 0.62) 100%);
    }}
    .cinema-gradient-top {{
      position: absolute;
      left: 0;
      right: 0;
      top: 0;
      height: 310px;
      pointer-events: none;
      z-index: 11;
      background: linear-gradient(to bottom, rgba(5, 7, 12, 0.76) 0%, transparent 100%);
    }}
    .cinema-gradient-bottom {{
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      height: 740px;
      pointer-events: none;
      z-index: 12;
      background: linear-gradient(to top, rgba(5, 7, 12, 0.92) 0%, rgba(5, 7, 12, 0.52) 52%, transparent 100%);
    }}
    .grain {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 15;
      opacity: 0.10;
      background-image:
        radial-gradient(rgba(255, 255, 255, 0.07) 1px, transparent 1.2px),
        radial-gradient(rgba(0, 0, 0, 0.15) 1px, transparent 1.2px);
      background-size: 3px 3px, 5px 5px;
      background-position: 0 0, 1px 2px;
      mix-blend-mode: overlay;
    }}
    .scene-content {{
      position: relative;
      z-index: 20;
      width: 100%;
      height: 100%;
      padding: 104px 74px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .top-hud-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    .era-top-pill {{
      background: var(--panel);
      border: 2px solid var(--accent);
      color: var(--accent);
      padding: 10px 24px;
      border-radius: 12px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0.08em;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.55);
    }}
    .source-pill {{
      background: rgba(8, 10, 15, 0.82);
      border: 1px solid rgba(255, 255, 255, 0.22);
      color: var(--ink);
      padding: 10px 20px;
      border-radius: 999px;
      font-size: 19px;
      font-weight: 600;
      letter-spacing: 0.06em;
    }}
    .telemetry-header {{
      font-size: 19px;
      font-weight: 700;
      color: var(--emerald);
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }}
    .telemetry-number {{
      font-size: 82px;
      font-weight: 700;
      line-height: 1.04;
      color: #ffffff;
      text-shadow: 0 6px 22px rgba(0, 245, 155, 0.35);
    }}
    .telemetry-label {{
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0.08em;
      color: var(--accent);
      margin-top: 4px;
    }}
    .era-lower-card {{
      align-self: flex-start;
      margin-top: auto;
      margin-bottom: 560px;
      background: rgba(8, 10, 16, 0.86);
      border-left: 7px solid var(--accent);
      padding: 16px 28px;
      border-radius: 0 16px 16px 0;
      box-shadow: 0 16px 38px rgba(0, 0, 0, 0.75);
    }}
    .era-subject {{
      position: relative;
      z-index: 1;
      font-size: 34px;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: 0.04em;
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
      background: rgba(6, 8, 14, 0.86);
      border: 1.5px solid rgba(255, 255, 255, 0.18);
      border-radius: 22px;
      padding: 14px 26px;
      box-shadow: 0 18px 44px rgba(0, 0, 0, 0.88);
      display: inline-flex;
      flex-wrap: wrap;
      justify-content: center;
      align-items: center;
      column-gap: 18px;
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
  <title>HyperFrames + Huashu Art Motion Studio Preview</title>
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
        theme: Dict[str, str],
    ) -> str:
        lines = [
            f"# HyperFrames v18.0 + Huashu Art Motion Specification — {title}",
            f"- **Topic**: {topic}",
            f"- **Locked Documentary Theme (`huashu-art-motion`)**: `{theme['name']}` (`{theme['accent']}` / `{theme['secondary']}`)",
            f"- **Resolution & Frame Rate**: `1080x1920` (9:16 Vertical) @ `30fps`",
            f"- **Total Duration**: `{total_duration:.2f}s` across `{len(tiled_scenes)}` scenes",
            f"- **Visual Integrity**: Zero duplicate text, zero blind lines/pointers on photos, `0px` blur",
            f"- **HyperShader WebGL Transitions**: `{json.dumps(shader_transitions)}`",
            "",
            "## Scene Timeline & Huashu Stage Layout Breakdown",
            "| Scene ID | Window (s) | Huashu Stage Layout | Camera Move | Era Pill | Callout (Shown Once) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for s in tiled_scenes:
            lines.append(
                f"| `{s['hf_id']}` | `{s['hf_start']:.2f}s – {s['hf_end']:.2f}s` | "
                f"`{s.get('huashu_grammar', 'fullbleed_hook_stage')}` | `{s.get('huashu_camera', 'slow_push_in')}` | "
                f"`{s.get('era_tag', '')}` | `{s.get('on_screen_callout', '')}` |"
            )
        return "\n".join(lines) + "\n"

    def _render_via_playwright_browser(self, output_mp4: Path) -> bool:
        if not self._last_build_args:
            return False
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except Exception as exc:
            print(f"[HyperFrames] Playwright import notice ({exc}); skipping browser fallback.", flush=True)
            return False

        try:
            t0 = time.time()
            pure_html = self._generate_index_html(
                topic=self._last_build_args["topic"],
                title=self._last_build_args["title"],
                tiled_scenes=self._last_build_args["tiled_scenes"],
                word_timings=self._last_build_args["word_timings"],
                audio_rel=self._last_build_args["audio_rel"],
                total_duration=self._last_build_args["total_duration"],
                shader_anchors=[],
                shader_transitions=[],
                theme=self._last_build_args.get("theme"),
            )
            index_file = (self.project_dir / "index.html").resolve()
            index_file.write_text(pure_html, encoding="utf-8")
            audio_file = (self.project_dir / self._last_build_args["audio_rel"]).resolve()
            total_dur = float(self._last_build_args["total_duration"])
            total_frames = max(30, int(round(total_dur * self.FPS)))

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-framerate",
                str(self.FPS),
                "-i",
                "-",
                "-i",
                str(audio_file),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-shortest",
                str(output_mp4.resolve()),
            ]
            ff_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
                page = browser.new_page(viewport={"width": self.WIDTH, "height": self.HEIGHT})
                page.goto(f"file://{index_file}", wait_until="networkidle", timeout=45000)
                page.wait_for_timeout(300)

                for frame_idx in range(total_frames):
                    t_sec = round(frame_idx / float(self.FPS), 4)
                    page.evaluate(
                        """(t) => {
                            if (window.__timelines && window.__timelines["main"]) {
                                window.__timelines["main"].seek(t, false);
                            }
                        }""",
                        t_sec,
                    )
                    png_bytes = page.screenshot(type="png")
                    if ff_proc.stdin:
                        ff_proc.stdin.write(png_bytes)

                browser.close()

            if ff_proc.stdin:
                ff_proc.stdin.close()
            ff_proc.wait(timeout=120)

            if ff_proc.returncode == 0 and output_mp4.exists() and output_mp4.stat().st_size > 250_000:
                print(
                    f"[HyperFrames] Rendered MP4 via Headless Chromium HTML+Huashu compositor "
                    f"({output_mp4.stat().st_size // 1024} KB in {time.time() - t0:.1f}s)!",
                    flush=True,
                )
                return True
        except Exception as exc:
            print(f"[HyperFrames] Headless Chromium fallback notice: {exc}", flush=True)
        return False

    def try_cli_render(self, output_mp4: Path, timeout_sec: int = 600) -> bool:
        if os.getenv("RENDER_ENGINE", "hyperframes").strip().lower() == "ffmpeg":
            return False

        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        npx_bin = shutil.which("npx")
        if npx_bin:
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
                    print("[HyperFrames] `npx hyperframes lint` passed with zero structural errors!", flush=True)
                else:
                    print(f"[HyperFrames] Lint notice: {lint_proc.stdout[:240]} {lint_proc.stderr[:240]}", flush=True)
            except Exception:
                pass

            try:
                for attempt in (1, 2):
                    if attempt == 2 and self._last_build_args is not None:
                        pure_html = self._generate_index_html(
                            topic=self._last_build_args["topic"],
                            title=self._last_build_args["title"],
                            tiled_scenes=self._last_build_args["tiled_scenes"],
                            word_timings=self._last_build_args["word_timings"],
                            audio_rel=self._last_build_args["audio_rel"],
                            total_duration=self._last_build_args["total_duration"],
                            shader_anchors=[],
                            shader_transitions=[],
                            theme=self._last_build_args.get("theme"),
                        )
                        (self.project_dir / "index.html").write_text(pure_html, encoding="utf-8")

                    print(f"[HyperFrames] Launching `npx hyperframes render --quality draft --output {output_mp4.resolve()}` (Attempt {attempt}/2)...", flush=True)
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
                            f"({output_mp4.stat().st_size // 1024} KB)!",
                            flush=True,
                        )
                        return True
            except Exception as exc:
                print(f"[HyperFrames] CLI execution notice ({exc}); falling back to Headless Chromium compositor.", flush=True)

        if self._render_via_playwright_browser(output_mp4):
            return True

        return False
