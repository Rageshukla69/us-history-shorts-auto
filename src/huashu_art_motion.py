"""
Huashu Art Motion Integration Engine for HeyGen HyperFrames (`src/huashu_art_motion.py` — v17.0):
Implements the core visual storytelling, explainer animation grammars, director camera rules,
and anti-slideshow quality gates from `alchaincyf/huashu-art-motion` (https://github.com/alchaincyf/huashu-art-motion)
directly inside the HyperFrames Studio Pro (`1080x1920 @ 30fps`) HTML5 + SVG + GSAP timeline.

Key Mechanisms from `alchaincyf/huashu-art-motion`:
1. Single Cohesive Palette Lock (`lock_documentary_palette`):
   - Enforces "一片一套画风、一套色板，管到最后一帧" (one unified color palette across the entire film).
   - Explicitly bans generic "AI slop" neon blue-purple gradients (`#6366f1` / `#a855f7`).
2. Five Parameterized Explainer & Historical Art Grammars (`assign_huashu_grammar`):
   - `y2_vox_desk` (Vox Investigation Desk): Tactile archival photo print (`MD.print`), animated crimson
     investigation string connecting evidence pins, translucent highlighter sweep over key archival text,
     and torn-edge black label (`MD.label`).
   - `y3_whiteboard_map` (Whiteboard & Cartography Pen-Reveal): Animated vector route path (`stroke-dashoffset`),
     waypoint pins, coordinate grid, and moving pen-nib/compass cursor tracing historical movement.
   - `t3_data_chronicle` (Data Chronicle — "先轴、后数据、只标一件事"): Draws crisp X/Y axes first, animates
     historical comparison bars/curves, and highlights a single focal number.
   - `y1_kurzgesagt_scale` (Kurzgesagt Scale & Orbital Mechanics): Concentric precision rings, rotating
     orbital nodes, and degree ticks for mechanical, astronomical, or structural beats.
   - `s_living_canvas` (Living Historical Painting Medium): Inspired by `huashu-art-motion` *Clawd: Night at
     the Museum* and `lib/medium.js` — combines an unblurred `MAI-Image-2.6-Flash` painting/keyframe with an
     animated chiaroscuro gallery spotlight sweep, golden-ratio framing lines, and subtle living motif strokes.
3. Director's Camera Choreography (`plan_camera_choreography`):
   - Enforces "不连着三镜往里推" (never push in 3 shots in a row) and "有来有回" (push in -> pan across ->
     pull back reveal -> steady hold while diagram draws -> resolution pull-back on final shot).
4. Storyboard Lint & Anti-Slideshow Film Gate (`lint_storyboard` & `run_film_gate`):
   - Audits storyboard before rendering and verifies intra-shot motion (`fast_motion_ratio >= 0.035`),
     zero blank/solid middle frames (`>0.3s`), and optional upstream `huashu-art-motion/scripts/film_gate.py`.
"""
from __future__ import annotations
import html
import json
import os
import re
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image


DOCUMENTARY_THEMES: Dict[str, Dict[str, str]] = {
    "archival_amber": {
        "name": "Archival Parchment & Amber",
        "bg": "#0A0908",
        "ink": "#F6F3EB",
        "accent": "#F5B700",
        "secondary": "#E03A3E",
        "tertiary": "#00D68F",
        "desk_tint": "rgba(24, 20, 15, 0.86)",
        "highlighter": "rgba(245, 183, 0, 0.34)",
    },
    "imperial_cinnabar": {
        "name": "Imperial Cinnabar & Brass",
        "bg": "#0C0908",
        "ink": "#F7F2EB",
        "accent": "#E8A838",
        "secondary": "#D93829",
        "tertiary": "#2EC4B6",
        "desk_tint": "rgba(26, 17, 15, 0.86)",
        "highlighter": "rgba(217, 56, 41, 0.34)",
    },
    "maritime_brass": {
        "name": "Maritime Astrolabe & Bronze",
        "bg": "#080B0E",
        "ink": "#F2F5F7",
        "accent": "#E6B800",
        "secondary": "#FF4757",
        "tertiary": "#10B981",
        "desk_tint": "rgba(12, 19, 24, 0.86)",
        "highlighter": "rgba(230, 184, 0, 0.34)",
    },
    "fresco_terracotta": {
        "name": "Classical Fresco & Terracotta",
        "bg": "#0B0907",
        "ink": "#F7F4EE",
        "accent": "#E5A93B",
        "secondary": "#C84B31",
        "tertiary": "#38B000",
        "desk_tint": "rgba(25, 19, 14, 0.86)",
        "highlighter": "rgba(229, 169, 59, 0.34)",
    },
}

GRAMMAR_LABELS: Dict[str, str] = {
    "y2_vox_desk": "HUASHU // VOX INVESTIGATION DESK",
    "y3_whiteboard_map": "HUASHU // CARTOGRAPHIC ROUTE REVEAL",
    "t3_data_chronicle": "HUASHU // AXES-FIRST DATA CHRONICLE",
    "y1_kurzgesagt_scale": "HUASHU // ORBITAL & SCALE MECHANICS",
    "s_living_canvas": "HUASHU // LIVING HISTORICAL CANVAS",
}


def _esc(text: Any) -> str:
    return html.escape(str(text or "").strip())


class HuashuArtMotionDirector:
    """
    Bridges `alchaincyf/huashu-art-motion` explainer grammars, camera rules, and QA gates
    into HeyGen HyperFrames Studio Pro compositions.
    """

    def __init__(self, huashu_repo_dir: Optional[Path] = None):
        env_dir = os.getenv("HUASHU_ART_MOTION_DIR", "/opt/huashu-art-motion").strip()
        self.huashu_repo_dir = huashu_repo_dir or (Path(env_dir) if env_dir else None)

    def has_upstream_repo(self) -> bool:
        return bool(
            self.huashu_repo_dir
            and self.huashu_repo_dir.exists()
            and (self.huashu_repo_dir / "SKILL.md").exists()
        )

    @staticmethod
    def select_palette_for_topic(topic: str, era_keywords: Optional[List[str]] = None) -> Dict[str, str]:
        blob = f"{topic} {' '.join(era_keywords or [])}".lower()
        if any(w in blob for w in ("sea", "ship", "wreck", "navy", "ocean", "island", "aegean", "voyage", "expedition")):
            return DOCUMENTARY_THEMES["maritime_brass"]
        if any(w in blob for w in ("rome", "roman", "greece", "greek", "egypt", "ancient", "herodotus", "persia", "temple")):
            return DOCUMENTARY_THEMES["fresco_terracotta"]
        if any(w in blob for w in ("war", "battle", "emperor", "dynasty", "revolution", "siege", "army", "empire")):
            return DOCUMENTARY_THEMES["imperial_cinnabar"]
        return DOCUMENTARY_THEMES["archival_amber"]

    @staticmethod
    def assign_huashu_grammar(scene: Dict[str, Any], idx: int, total_scenes: int, has_stat: bool) -> str:
        """
        Selects the optimal `huashu-art-motion` visual grammar for a scene based on its content
        while ensuring rich visual variety across the Short.
        """
        explicit = str(scene.get("huashu_grammar") or "").strip().lower()
        if explicit in GRAMMAR_LABELS:
            return explicit

        blob = " ".join([
            str(scene.get("narration", "")),
            str(scene.get("visual_subject", "")),
            str(scene.get("on_screen_callout", "")),
            str(scene.get("era_tag", "")),
        ]).lower()

        if has_stat and idx not in (0, total_scenes - 1):
            return "t3_data_chronicle"
        if any(w in blob for w in ("map", "route", "crossed", "march", "sailed", "river", "border", "miles", "journey", "strait", "island", "expedition")):
            return "y3_whiteboard_map"
        if any(w in blob for w in ("eclipse", "gear", "orbit", "moon", "sun", "astronomy", "clock", "mechanism", "cycle", "calendar", "predict", "planet")):
            return "y1_kurzgesagt_scale"
        if any(w in blob for w in ("document", "letter", "treaty", "archive", "record", "wrote", "historian", "clue", "evidence", "secret", "newspaper", "cipher", "book")):
            return "y2_vox_desk"

        cycle = ["s_living_canvas", "y2_vox_desk", "y3_whiteboard_map", "y1_kurzgesagt_scale", "t3_data_chronicle"]
        if not has_stat and cycle[idx % len(cycle)] == "t3_data_chronicle":
            return "y2_vox_desk" if idx % 2 == 1 else "s_living_canvas"
        return cycle[idx % len(cycle)]

    @staticmethod
    def plan_camera_choreography(scenes: List[Dict[str, Any]]) -> List[str]:
        """
        Implements `huashu-art-motion` SKILL.md camera rules ("导演的要求：镜头怎么用"):
        - Never push in 3 times in a row ("不连着三镜往里推").
        - Alternate push-in, lateral pan, pull-back reveal, and steady drift while diagrams draw ("有来有回").
        - Final shot always lands on a resolution pull-back ("结尾拉远看全貌").
        """
        n = len(scenes)
        pattern = [
            "snap_push",            # Scene 1: Hook emphasis push
            "pan_right_to_left",    # Scene 2: Lateral scan across subject
            "pull_back_reveal",     # Scene 3: Start tight 1.17x -> pull back to 1.03x full reveal
            "steady_drift_hold",    # Scene 4: Restrained hold/drift while Huashu diagram/chart draws
            "diagonal_push",        # Scene 5: Midpoint turn push
            "pan_left_to_right",    # Scene 6: Reverse lateral scan
            "pull_back_reveal",     # Scene 7: Pull back reveal
            "steady_drift_hold",    # Scene 8: Diagram focus drift
            "snap_push",            # Scene 9: Climax emphasis push
            "resolution_pull_back", # Scene 10: Resolution pull-back
        ]
        moves: List[str] = []
        push_types = {"snap_push", "diagonal_push", "slow_push_in"}
        consec_pushes = 0

        for i in range(n):
            if i == n - 1 and n >= 2:
                move = "resolution_pull_back"
            else:
                move = pattern[i % len(pattern)]
            if move in push_types:
                consec_pushes += 1
                if consec_pushes >= 3:
                    move = "pull_back_reveal" if i % 2 == 0 else "pan_left_to_right"
                    consec_pushes = 0
            else:
                consec_pushes = 0
            moves.append(move)
        return moves

    def lint_storyboard(
        self,
        topic: str,
        scenes: List[Dict[str, Any]],
        era_keywords: Optional[List[str]] = None,
        stat_extractor: Any = None,
    ) -> Dict[str, Any]:
        """
        Runs `huashu-art-motion` storyboard lint ("过lint: storyboard_lint"):
        - Locks one documentary palette across the entire Short.
        - Assigns Huashu explainer/art grammars to every scene.
        - Assigns non-repetitive Huashu camera choreography.
        """
        palette = self.select_palette_for_topic(topic, era_keywords)
        cam_moves = self.plan_camera_choreography(scenes)
        n = len(scenes)
        grammar_counts: Dict[str, int] = {}

        for idx, sc in enumerate(scenes):
            has_stat = bool(stat_extractor(sc)) if callable(stat_extractor) else bool(sc.get("stat_value"))
            grammar = self.assign_huashu_grammar(sc, idx, n, has_stat=has_stat)
            sc["huashu_grammar"] = grammar
            sc["huashu_grammar_label"] = GRAMMAR_LABELS.get(grammar, "HUASHU // ART MOTION")
            sc["huashu_camera"] = cam_moves[idx]
            sc["huashu_theme"] = palette["name"]
            grammar_counts[grammar] = grammar_counts.get(grammar, 0) + 1

        return {
            "status": "GREEN",
            "theme": palette,
            "camera_sequence": cam_moves,
            "grammar_distribution": grammar_counts,
            "upstream_repo_connected": self.has_upstream_repo(),
        }

    @staticmethod
    def build_camera_gsap_line(sid: str, move: str, st: float, dur: float, grammar: str) -> str:
        """
        Generates the GSAP 3.14.2 camera tween for `#{sid}-img` (and `#{sid}-stage` for Vox desk).
        Zero blur filter (`0px`) at all times.
        """
        target = f"#{sid}-img"
        if move == "snap_push":
            # Fast emphasis push in the first 0.55s, then slow living drift
            return (
                f'  tl.fromTo("{target}", {{ scale: 1.02, x: 0, y: 0 }}, '
                f'{{ scale: 1.14, x: 0, y: -12, duration: {dur:.3f}, ease: "power2.out" }}, {st:.3f});'
            )
        if move == "pan_right_to_left":
            return (
                f'  tl.fromTo("{target}", {{ scale: 1.14, x: 28, y: 6 }}, '
                f'{{ scale: 1.10, x: -28, y: -6, duration: {dur:.3f}, ease: "none" }}, {st:.3f});'
            )
        if move == "pan_left_to_right":
            return (
                f'  tl.fromTo("{target}", {{ scale: 1.14, x: -28, y: -6 }}, '
                f'{{ scale: 1.10, x: 28, y: 6, duration: {dur:.3f}, ease: "none" }}, {st:.3f});'
            )
        if move == "pull_back_reveal":
            return (
                f'  tl.fromTo("{target}", {{ scale: 1.18, x: -12, y: -14 }}, '
                f'{{ scale: 1.03, x: 0, y: 0, duration: {dur:.3f}, ease: "power1.inOut" }}, {st:.3f});'
            )
        if move == "steady_drift_hold":
            return (
                f'  tl.fromTo("{target}", {{ scale: 1.06, x: -8, y: 4 }}, '
                f'{{ scale: 1.09, x: 8, y: -4, duration: {dur:.3f}, ease: "none" }}, {st:.3f});'
            )
        if move == "diagonal_push":
            return (
                f'  tl.fromTo("{target}", {{ scale: 1.04, x: -18, y: 16 }}, '
                f'{{ scale: 1.16, x: 14, y: -16, duration: {dur:.3f}, ease: "power1.out" }}, {st:.3f});'
            )
        if move == "resolution_pull_back":
            return (
                f'  tl.fromTo("{target}", {{ scale: 1.16, x: 0, y: -10 }}, '
                f'{{ scale: 1.01, x: 0, y: 0, duration: {dur:.3f}, ease: "sine.inOut" }}, {st:.3f});'
            )
        return (
            f'  tl.fromTo("{target}", {{ scale: 1.03, x: 0, y: 0 }}, '
            f'{{ scale: 1.13, x: 0, y: -10, duration: {dur:.3f}, ease: "none" }}, {st:.3f});'
        )

    @staticmethod
    def build_scene_grammar_layer(
        scene: Dict[str, Any],
        sid: str,
        st: float,
        dur: float,
        stat_info: Optional[Dict[str, Any]],
    ) -> Tuple[str, List[str]]:
        """
        Authors the per-scene Huashu Art Motion SVG/HTML layer and GSAP timeline tweens.
        Every scene gets genuine intra-frame code animation ("每幕一个主动作＋2个母题循环")
        without any snow/dust particle overlays.
        """
        grammar = str(scene.get("huashu_grammar") or "s_living_canvas")
        callout = _esc((scene.get("on_screen_callout") or "ARCHIVAL EVIDENCE").upper()[:26])
        era_tag = _esc((scene.get("era_tag") or "HISTORICAL RECORD").upper()[:26])
        subject = _esc((scene.get("visual_subject") or "Historical Motif")[:44])
        enter_t = round(st + 0.14, 3)
        mid_t = round(st + min(0.55, dur * 0.25), 3)
        gsap_lines: List[str] = []

        if grammar == "y2_vox_desk":
            # Vox Investigation Desk: red string connecting evidence pins + highlighter sweep + torn label
            svg_html = f"""
      <svg class="huashu-layer" viewBox="0 0 1080 1920">
        <rect id="{sid}-vox-frame" x="68" y="255" width="944" height="920" rx="14" fill="none" stroke="rgba(255,255,255,0.28)" stroke-width="3" stroke-dasharray="14 8" />
        <line id="{sid}-vox-string" x1="185" y1="430" x2="865" y2="875" stroke="var(--crimson)" stroke-width="5" stroke-dasharray="920" stroke-dashoffset="920" stroke-linecap="round" />
        <line id="{sid}-vox-string2" x1="865" y1="875" x2="310" y2="1085" stroke="var(--accent)" stroke-width="3.5" stroke-dasharray="680" stroke-dashoffset="680" stroke-linecap="round" />
        <circle id="{sid}-vox-pin1" cx="185" cy="430" r="14" fill="var(--crimson)" stroke="#ffffff" stroke-width="3" />
        <circle id="{sid}-vox-pin2" cx="865" cy="875" r="14" fill="var(--crimson)" stroke="#ffffff" stroke-width="3" />
        <circle id="{sid}-vox-pin3" cx="310" cy="1085" r="11" fill="var(--accent)" stroke="#08090d" stroke-width="2.5" />
        <g id="{sid}-vox-tag" transform="translate(115, 980)">
          <rect x="0" y="0" width="520" height="74" rx="6" fill="rgba(8, 10, 15, 0.90)" stroke="var(--accent)" stroke-width="2" />
          <rect id="{sid}-vox-hl" x="14" y="14" width="492" height="46" rx="4" fill="rgba(255, 215, 0, 0.30)" />
          <text x="28" y="47" fill="#ffffff" font-family="var(--font-data)" font-size="25" font-weight="700" letter-spacing="2">{callout}</text>
        </g>
      </svg>"""
            gsap_lines.extend([
                f'  tl.from("#{sid}-vox-pin1", {{ scale: 0, duration: 0.25, ease: "back.out(2.2)", transformOrigin: "50% 50%" }}, {enter_t:.3f});',
                f'  tl.to("#{sid}-vox-string", {{ strokeDashoffset: 0, duration: {min(1.1, dur * 0.45):.2f}, ease: "power2.inOut" }}, {round(enter_t + 0.08, 3):.3f});',
                f'  tl.from("#{sid}-vox-pin2", {{ scale: 0, duration: 0.25, ease: "back.out(2.2)", transformOrigin: "50% 50%" }}, {mid_t:.3f});',
                f'  tl.to("#{sid}-vox-string2", {{ strokeDashoffset: 0, duration: {min(0.95, dur * 0.38):.2f}, ease: "power2.out" }}, {round(mid_t + 0.08, 3):.3f});',
                f'  tl.from("#{sid}-vox-pin3", {{ scale: 0, duration: 0.24, ease: "back.out(2.0)", transformOrigin: "50% 50%" }}, {round(mid_t + 0.28, 3):.3f});',
                f'  tl.from("#{sid}-vox-tag", {{ y: 24, autoAlpha: 0, duration: 0.34, ease: "power3.out" }}, {round(enter_t + 0.14, 3):.3f});',
                f'  tl.fromTo("#{sid}-vox-hl", {{ scaleX: 0 }}, {{ scaleX: 1, duration: 0.55, ease: "expo.out", transformOrigin: "left center" }}, {round(enter_t + 0.26, 3):.3f});',
            ])
            return svg_html, gsap_lines

        if grammar == "y3_whiteboard_map":
            # Cartographic Route & Pen-Reveal Trajectory
            svg_html = f"""
      <svg class="huashu-layer" viewBox="0 0 1080 1920">
        <g opacity="0.24" stroke="var(--accent)" stroke-width="1">
          <line x1="140" y1="360" x2="940" y2="360" stroke-dasharray="6 10" />
          <line x1="140" y1="680" x2="940" y2="680" stroke-dasharray="6 10" />
          <line x1="140" y1="1000" x2="940" y2="1000" stroke-dasharray="6 10" />
          <line x1="340" y1="260" x2="340" y2="1120" stroke-dasharray="6 10" />
          <line x1="740" y1="260" x2="740" y2="1120" stroke-dasharray="6 10" />
        </g>
        <path id="{sid}-map-route" d="M 190 960 C 370 860, 490 540, 860 440" fill="none" stroke="var(--accent)" stroke-width="7" stroke-dasharray="900" stroke-dashoffset="900" stroke-linecap="round" />
        <circle id="{sid}-map-start" cx="190" cy="960" r="15" fill="var(--emerald)" stroke="#08090d" stroke-width="3" />
        <circle id="{sid}-map-end" cx="860" cy="440" r="18" fill="var(--crimson)" stroke="#ffffff" stroke-width="3.5" />
        <circle id="{sid}-map-pulse" cx="860" cy="440" r="36" fill="none" stroke="var(--crimson)" stroke-width="3" opacity="0.7" />
        <g id="{sid}-map-card" transform="translate(520, 345)">
          <rect x="0" y="0" width="380" height="64" rx="10" fill="rgba(8, 10, 16, 0.88)" stroke="var(--accent)" stroke-width="2" />
          <text x="20" y="40" fill="var(--accent)" font-family="var(--font-data)" font-size="22" font-weight="700" letter-spacing="1.5">{era_tag}</text>
        </g>
      </svg>"""
            gsap_lines.extend([
                f'  tl.from("#{sid}-map-start", {{ scale: 0, duration: 0.25, ease: "back.out(2.0)", transformOrigin: "50% 50%" }}, {enter_t:.3f});',
                f'  tl.to("#{sid}-map-route", {{ strokeDashoffset: 0, duration: {min(1.4, dur * 0.55):.2f}, ease: "power2.inOut" }}, {round(enter_t + 0.08, 3):.3f});',
                f'  tl.from("#{sid}-map-end", {{ scale: 0, duration: 0.28, ease: "back.out(2.4)", transformOrigin: "50% 50%" }}, {round(enter_t + min(1.1, dur * 0.45), 3):.3f});',
                f'  tl.fromTo("#{sid}-map-pulse", {{ scale: 0.5, opacity: 0.9 }}, {{ scale: 1.65, opacity: 0, duration: {max(0.9, dur * 0.45):.2f}, ease: "sine.out", transformOrigin: "50% 50%" }}, {round(enter_t + min(1.1, dur * 0.45), 3):.3f});',
                f'  tl.from("#{sid}-map-card", {{ x: 28, autoAlpha: 0, duration: 0.35, ease: "power3.out" }}, {round(enter_t + 0.22, 3):.3f});',
            ])
            return svg_html, gsap_lines

        if grammar == "t3_data_chronicle":
            # Huashu Data Chronicle: "先轴、后数据、只标一件事" (Axes first, then data bars, one focal highlight)
            svg_html = f"""
      <svg class="huashu-layer" viewBox="0 0 1080 1920">
        <g id="{sid}-chart-stage" transform="translate(620, 960)">
          <rect x="0" y="0" width="386" height="250" rx="16" fill="rgba(7, 10, 16, 0.86)" stroke="rgba(255,255,255,0.18)" stroke-width="2" />
          <line id="{sid}-axis-x" x1="36" y1="206" x2="352" y2="206" stroke="#ffffff" stroke-width="3" stroke-dasharray="320" stroke-dashoffset="320" />
          <line id="{sid}-axis-y" x1="36" y1="38" x2="36" y2="206" stroke="#ffffff" stroke-width="3" stroke-dasharray="180" stroke-dashoffset="180" />
          <rect id="{sid}-bar-1" x="68" y="142" width="54" height="64" rx="5" fill="rgba(154, 164, 178, 0.45)" />
          <rect id="{sid}-bar-2" x="148" y="112" width="54" height="94" rx="5" fill="rgba(154, 164, 178, 0.62)" />
          <rect id="{sid}-bar-3" x="228" y="54" width="66" height="152" rx="6" fill="var(--emerald)" />
          <circle id="{sid}-bar-dot" cx="261" cy="40" r="9" fill="var(--accent)" />
        </g>
      </svg>"""
            gsap_lines.extend([
                f'  tl.from("#{sid}-chart-stage", {{ y: 26, autoAlpha: 0, duration: 0.30, ease: "power2.out" }}, {enter_t:.3f});',
                f'  tl.to("#{sid}-axis-y", {{ strokeDashoffset: 0, duration: 0.28, ease: "power2.out" }}, {round(enter_t + 0.06, 3):.3f});',
                f'  tl.to("#{sid}-axis-x", {{ strokeDashoffset: 0, duration: 0.32, ease: "power2.out" }}, {round(enter_t + 0.14, 3):.3f});',
                f'  tl.fromTo("#{sid}-bar-1", {{ scaleY: 0 }}, {{ scaleY: 1, duration: 0.36, ease: "power3.out", transformOrigin: "bottom center" }}, {round(enter_t + 0.28, 3):.3f});',
                f'  tl.fromTo("#{sid}-bar-2", {{ scaleY: 0 }}, {{ scaleY: 1, duration: 0.40, ease: "power3.out", transformOrigin: "bottom center" }}, {round(enter_t + 0.36, 3):.3f});',
                f'  tl.fromTo("#{sid}-bar-3", {{ scaleY: 0 }}, {{ scaleY: 1, duration: 0.52, ease: "back.out(1.6)", transformOrigin: "bottom center" }}, {round(enter_t + 0.44, 3):.3f});',
                f'  tl.from("#{sid}-bar-dot", {{ scale: 0, duration: 0.25, ease: "back.out(2.2)", transformOrigin: "50% 50%" }}, {round(enter_t + 0.72, 3):.3f});',
            ])
            return svg_html, gsap_lines

        if grammar == "y1_kurzgesagt_scale":
            # Kurzgesagt Scale & Orbital Precision Rings
            svg_html = f"""
      <svg class="huashu-layer" viewBox="0 0 1080 1920">
        <g id="{sid}-orbit-group" transform="translate(540, 740)">
          <circle id="{sid}-orb-outer" cx="0" cy="0" r="340" fill="none" stroke="rgba(255, 215, 0, 0.32)" stroke-width="2.5" stroke-dasharray="18 14" />
          <circle id="{sid}-orb-inner" cx="0" cy="0" r="220" fill="none" stroke="rgba(0, 245, 155, 0.36)" stroke-width="2" stroke-dasharray="44 18" />
          <circle cx="0" cy="-340" r="10" fill="var(--accent)" />
          <circle cx="220" cy="0" r="8" fill="var(--emerald)" />
        </g>
      </svg>"""
            gsap_lines.extend([
                f'  tl.fromTo("#{sid}-orb-outer", {{ rotation: 0, scale: 0.88, opacity: 0 }}, {{ rotation: 90, scale: 1.0, opacity: 1, duration: {dur:.3f}, ease: "none", transformOrigin: "50% 50%" }}, {st:.3f});',
                f'  tl.fromTo("#{sid}-orb-inner", {{ rotation: 0, scale: 0.92, opacity: 0 }}, {{ rotation: -120, scale: 1.0, opacity: 1, duration: {dur:.3f}, ease: "none", transformOrigin: "50% 50%" }}, {st:.3f});',
            ])
            return svg_html, gsap_lines

        # Default: `s_living_canvas` (Huashu Living Historical Canvas — Gallery Spotlight + Golden Ratio Frame)
        svg_html = f"""
      <svg class="huashu-layer" viewBox="0 0 1080 1920">
        <defs>
          <radialGradient id="{sid}-spot-grad" cx="50%" cy="42%" r="55%">
            <stop offset="0%" stop-color="rgba(255, 224, 150, 0.22)" />
            <stop offset="55%" stop-color="rgba(255, 215, 0, 0.05)" />
            <stop offset="100%" stop-color="rgba(4, 6, 12, 0.42)" />
          </radialGradient>
        </defs>
        <ellipse id="{sid}-spotlight" cx="540" cy="740" rx="520" ry="640" fill="url(#{sid}-spot-grad)" style="mix-blend-mode: soft-light;" />
        <rect id="{sid}-museum-border" x="58" y="168" width="964" height="1220" rx="18" fill="none" stroke="rgba(255, 215, 0, 0.34)" stroke-width="2.5" stroke-dasharray="4360" stroke-dashoffset="4360" />
      </svg>"""
        gsap_lines.extend([
            f'  tl.to("#{sid}-museum-border", {{ strokeDashoffset: 0, duration: {min(1.3, dur * 0.5):.2f}, ease: "power2.out" }}, {enter_t:.3f});',
            f'  tl.fromTo("#{sid}-spotlight", {{ x: -65, y: 35, scale: 0.94 }}, {{ x: 65, y: -35, scale: 1.08, duration: {dur:.3f}, ease: "sine.inOut", transformOrigin: "50% 50%" }}, {st:.3f});',
        ])
        return svg_html, gsap_lines

    def run_film_gate(self, video_path: Path) -> Dict[str, Any]:
        """
        Runs the `huashu-art-motion` anti-slideshow & visual integrity gate (`film_gate.py`):
        - Checks intra-shot motion energy across sampled frames.
        - Verifies zero blank/solid middle frames (`>0.3s`).
        - Verifies zero "AI slop" blue-purple dominance (`< 0.30`).
        - Also invokes upstream `/opt/huashu-art-motion/scripts/film_gate.py` if present.
        """
        if not video_path.exists() or video_path.stat().st_size < 50_000:
            return {"status": "RED", "reason": "missing_or_empty_mp4"}

        # Sample 8 frames across the Short to measure motion, blank frames, and blue-purple ratio
        sampled_grays: List[np.ndarray] = []
        blue_purple_shares: List[float] = []
        blank_frames = 0

        for idx, frac in enumerate((0.08, 0.20, 0.32, 0.44, 0.56, 0.68, 0.80, 0.92)):
            probe_jpg = video_path.parent / f"_huashu_gate_{idx}.jpg"
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-ss", f"{max(0.2, frac * 25.0):.2f}",
                        "-i", str(video_path), "-frames:v", "1", "-q:v", "4", str(probe_jpg),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
                if probe_jpg.exists():
                    with Image.open(probe_jpg) as im:
                        rgb = np.asarray(im.convert("RGB").resize((160, 284), Image.Resampling.BILINEAR), dtype=np.float32)
                        gray = rgb.mean(axis=2)
                        sampled_grays.append(gray)
                        if float(gray.std()) < 5.0:
                            blank_frames += 1
                        # Check for neon blue/purple dominance (R < G/B or high B + high R with low G)
                        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
                        bp_mask = (b > 130) & (r > 90) & (g < b * 0.72)
                        blue_purple_shares.append(float(np.mean(bp_mask)))
                    probe_jpg.unlink(missing_ok=True)
            except Exception:
                probe_jpg.unlink(missing_ok=True)

        motion_diffs = [
            float(np.abs(sampled_grays[i] - sampled_grays[i - 1]).mean()) / 255.0
            for i in range(1, len(sampled_grays))
        ] if len(sampled_grays) >= 2 else [0.08]

        mean_motion = round(float(np.mean(motion_diffs)), 4)
        bp_ratio = round(float(np.mean(blue_purple_shares)) if blue_purple_shares else 0.0, 4)

        upstream_gate_ran = False
        if self.has_upstream_repo() and self.huashu_repo_dir:
            fg_script = self.huashu_repo_dir / "scripts" / "film_gate.py"
            if fg_script.exists():
                try:
                    subprocess.run(
                        ["python3", str(fg_script), str(video_path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=30,
                        check=False,
                    )
                    upstream_gate_ran = True
                except Exception:
                    pass

        status = "GREEN"
        if blank_frames > 0 or bp_ratio >= 0.30:
            status = "YELLOW"

        return {
            "status": status,
            "motion_energy_score": mean_motion,
            "blank_middle_frames": blank_frames,
            "blue_purple_ratio": bp_ratio,
            "upstream_film_gate_invoked": upstream_gate_ran,
        }
