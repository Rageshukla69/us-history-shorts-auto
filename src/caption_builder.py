"""
Builds dynamic word-synced ASS subtitles in the viral "Editor V1" style:
- Groups words into tight 3-word phrases.
- Active spoken word pops in vibrant gold (`#FFD700`) at 112% scale while surrounding words stay crisp white.
"""
from pathlib import Path
from typing import List
from voice_engine import WordTiming


def _format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centis = int(round((secs - int(secs)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hours}:{minutes:02d}:{int(secs):02d}.{centis:02d}"


def _clean_word(word: str) -> str:
    return word.replace("{", "").replace("}", "").replace("\\", "").upper()


def generate_synced_ass_captions(
    word_timings: List[WordTiming],
    scenes: List[dict],
    output_ass: Path,
) -> Path:
    output_ass.parent.mkdir(parents=True, exist_ok=True)

    header = """[Script Info]
Title: US History Short Word-Synced Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: MainCaption,Montserrat,82,&H00FFFFFF,&H0000D7FF,&H00000000,&H96000000,-1,0,0,0,100,100,2,0,1,6,4,2,70,70,410,1
Style: TopBadge,Montserrat,46,&H0000D7FF,&H00FFFFFF,&H00000000,&HB4000000,-1,0,0,0,100,100,4,0,3,14,0,8,60,60,175,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    CHUNK_SIZE = 3
    i = 0
    n = len(word_timings)
    while i < n:
        scene_id = word_timings[i].scene_id
        chunk: List[WordTiming] = []
        while i < n and len(chunk) < CHUNK_SIZE and word_timings[i].scene_id == scene_id:
            chunk.append(word_timings[i])
            i += 1

        for active_idx, active_wt in enumerate(chunk):
            ev_start = active_wt.start_sec
            if active_idx + 1 < len(chunk):
                ev_end = chunk[active_idx + 1].start_sec
            else:
                ev_end = active_wt.end_sec

            if ev_end <= ev_start:
                ev_end = ev_start + 0.12

            formatted_words = []
            for j, wt in enumerate(chunk):
                w_str = _clean_word(wt.word)
                if j == active_idx:
                    formatted_words.append(
                        f"{{\\c&H00D7FF&\\fscx112\\fscy112}}{w_str}{{\\c&HFFFFFF&\\fscx100\\fscy100}}"
                    )
                else:
                    formatted_words.append(w_str)

            line_text = " ".join(formatted_words)
            events.append(
                f"Dialogue: 2,{_format_ass_time(ev_start)},{_format_ass_time(ev_end)},MainCaption,,0,0,0,,{line_text}"
            )

    for scene in scenes:
        badge = (scene.get("on_screen_callout") or "").strip().upper()
        if not badge:
            continue
        s_start = float(scene.get("start_sec", 0.0)) + 0.08
        s_end = min(float(scene.get("end_sec", s_start + 1.5)), s_start + 2.2)
        if s_end > s_start:
            clean_badge = _clean_word(badge)
            events.append(
                f"Dialogue: 1,{_format_ass_time(s_start)},{_format_ass_time(s_end)},TopBadge,,0,0,0,,{{\\fad(120,180)}}  {clean_badge}  "
            )

    output_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"[CaptionBuilder] Generated {len(events)} word-synced subtitle events -> {output_ass}")
    return output_ass
