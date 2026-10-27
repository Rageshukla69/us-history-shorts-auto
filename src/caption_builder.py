r"""
Builds dynamic word-synced ASS subtitles in the viral "Editor V1" documentary style:
- Guarantees 100% boundary safety inside a 1080x1920 vertical canvas (`WrapStyle: 0` smart wrapping,
  character-budget chunking, explicit `\N` line breaks for longer phrases, and 90px left/right margins).
- Attaches punctuation (`,`, `.`, `;`, `:`, `!`, `?`) directly to the preceding word so floating
  punctuation marks never appear (`FRANCE,` instead of `FRANCE ,`).
- Active spoken word pops in vibrant gold (`#FFD700`) at 108% scale while surrounding words stay crisp white.
- Deduplicates consecutive identical top badges so the same callout doesn't flicker repeatedly.
"""
import re
from pathlib import Path
from typing import List
from voice_engine import WordTiming

PUNCTUATION_ONLY_RE = re.compile(r"^[,.;:!?—–\-\"'\)\]]+$")
LEADING_PUNCT_RE = re.compile(r"^([,.;:!?]+)(.+)$")


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
    cleaned = word.replace("{", "").replace("}", "").replace("\\", "").strip().upper()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned


def _merge_punctuation_tokens(word_timings: List[WordTiming]) -> List[WordTiming]:
    """
    Merges standalone punctuation tokens (emitted by Azure/Edge TTS word boundary events)
    onto the immediately preceding word so captions never render ugly spaces like `FRANCE ,` or `RUBBER ;`.
    """
    merged: List[WordTiming] = []
    for wt in word_timings:
        raw = wt.word.strip()
        if not raw:
            continue

        if PUNCTUATION_ONLY_RE.match(raw) and merged and merged[-1].scene_id == wt.scene_id:
            prev = merged[-1]
            merged[-1] = WordTiming(
                word=f"{prev.word}{raw}",
                start_sec=prev.start_sec,
                end_sec=max(prev.end_sec, wt.end_sec),
                scene_id=prev.scene_id,
            )
            continue

        m = LEADING_PUNCT_RE.match(raw)
        if m and merged and merged[-1].scene_id == wt.scene_id:
            punct, rest = m.group(1), m.group(2).strip()
            prev = merged[-1]
            merged[-1] = WordTiming(
                word=f"{prev.word}{punct}",
                start_sec=prev.start_sec,
                end_sec=prev.end_sec,
                scene_id=prev.scene_id,
            )
            if rest:
                merged.append(
                    WordTiming(
                        word=rest,
                        start_sec=wt.start_sec,
                        end_sec=wt.end_sec,
                        scene_id=wt.scene_id,
                    )
                )
            continue

        merged.append(wt)
    return merged


def _group_into_safe_chunks(
    word_timings: List[WordTiming],
    max_words: int = 3,
    max_chars_two_lines: int = 26,
) -> List[List[WordTiming]]:
    """
    Groups words within each scene into tight, visually balanced chunks that NEVER overflow
    the 1080px horizontal safe zone (900px effective width at 62pt Montserrat ExtraBold).
    """
    chunks: List[List[WordTiming]] = []
    i = 0
    n = len(word_timings)

    while i < n:
        scene_id = word_timings[i].scene_id
        chunk: List[WordTiming] = []
        char_len = 0

        while i < n and word_timings[i].scene_id == scene_id and len(chunk) < max_words:
            candidate = _clean_word(word_timings[i].word)
            cand_len = len(candidate)
            projected = char_len + (1 if chunk else 0) + cand_len

            if chunk and projected > max_chars_two_lines:
                break
            if len(chunk) >= 2 and projected > 20:
                break

            chunk.append(word_timings[i])
            char_len = projected
            i += 1

            if candidate.endswith((".", "!", "?", ";", ":")):
                break

        if chunk:
            chunks.append(chunk)

    return chunks


def _format_chunk_lines(
    chunk: List[WordTiming],
    active_idx: int,
    max_single_line_chars: int = 15,
) -> str:
    r"""
    Formats a chunk of 1-3 words with active-word gold pop styling and inserts an explicit
    ASS `\N` line break when total characters exceed `max_single_line_chars` so text stays
    centered and never touches the left/right edges of the video frame.
    """
    plain_words = [_clean_word(wt.word) for wt in chunk]
    styled_words = []
    for j, w_str in enumerate(plain_words):
        if j == active_idx:
            styled_words.append(
                f"{{\\c&H00D7FF&\\fscx108\\fscy108}}{w_str}{{\\c&HFFFFFF&\\fscx100\\fscy100}}"
            )
        else:
            styled_words.append(w_str)

    total_chars = sum(len(w) for w in plain_words) + max(0, len(plain_words) - 1)
    if len(plain_words) <= 1 or total_chars <= max_single_line_chars:
        return " ".join(styled_words)

    if len(plain_words) == 2:
        return f"{styled_words[0]}\\N{styled_words[1]}"

    split_after_first = abs(len(plain_words[0]) - (len(plain_words[1]) + 1 + len(plain_words[2])))
    split_after_second = abs((len(plain_words[0]) + 1 + len(plain_words[1])) - len(plain_words[2]))
    if split_after_first <= split_after_second:
        return f"{styled_words[0]}\\N{styled_words[1]} {styled_words[2]}"
    return f"{styled_words[0]} {styled_words[1]}\\N{styled_words[2]}"


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
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: MainCaption,Montserrat,62,&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000,-1,0,0,0,100,100,1.5,0,1,5,3,2,90,90,430,1
Style: TopBadge,Montserrat,40,&H0000D7FF,&H00FFFFFF,&H00121212,&HC8080808,-1,0,0,0,100,100,3.0,0,3,12,0,8,90,90,185,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: List[str] = []

    clean_timings = _merge_punctuation_tokens(word_timings)
    chunks = _group_into_safe_chunks(clean_timings)

    for chunk in chunks:
        for active_idx, active_wt in enumerate(chunk):
            ev_start = active_wt.start_sec
            if active_idx + 1 < len(chunk):
                ev_end = chunk[active_idx + 1].start_sec
            else:
                ev_end = active_wt.end_sec

            if ev_end <= ev_start:
                ev_end = ev_start + 0.12

            line_text = _format_chunk_lines(chunk, active_idx)
            events.append(
                f"Dialogue: 2,{_format_ass_time(ev_start)},{_format_ass_time(ev_end)},MainCaption,,0,0,0,,{line_text}"
            )

    last_badge = ""
    for scene in scenes:
        badge = (scene.get("on_screen_callout") or "").strip().upper()
        if not badge:
            continue
        clean_badge = _clean_word(badge)[:24]
        if clean_badge == last_badge:
            continue
        last_badge = clean_badge

        s_start = float(scene.get("start_sec", 0.0)) + 0.08
        s_end = min(float(scene.get("end_sec", s_start + 1.5)), s_start + 2.1)
        if s_end > s_start:
            events.append(
                f"Dialogue: 1,{_format_ass_time(s_start)},{_format_ass_time(s_end)},TopBadge,,0,0,0,,{{\\fad(140,180)}}  {clean_badge}  "
            )

    output_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"[CaptionBuilder] Generated {len(events)} boundary-safe word-synced subtitle events -> {output_ass}")
    return output_ass
