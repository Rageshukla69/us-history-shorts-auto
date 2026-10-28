r"""
Builds dynamic word-synced ASS subtitles in the viral "Pro Creator V2" documentary style:
- High-impact 90pt Bold Montserrat typography (with automatic step-down to 74pt for extra-long 14+ char lines)
  positioned in the lower-middle "Thumb-Stop Zone" (`MarginV: 530` on `1080x1920`) well above YouTube Shorts UI overlays.
- 8px deep-black contour outline + 5px drop shadow + `\blur0.7` edge anti-aliasing so captions pop with razor-sharp
  contrast over any bright or dark background without needing an intrusive box.
- Smooth 75ms elastic active-word pop animation (`\fscx100\fscy100\t(0,75,\fscx114\fscy114)`) in Warm Gold (`#FFD700`)
  while surrounding words stay crisp pure white (`#FFFFFF`) at 100% scale.
- Subtle 45ms phrase entrance fade (`\fad(45,0)`) on every new 2-to-3 word chunk.
- Attaches punctuation (`,`, `.`, `;`, `:`, `!`, `?`) directly to the preceding word and truncates top badges at whole-word boundaries.
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


def _truncate_badge_words(text: str, max_chars: int = 22, max_words: int = 4) -> str:
    """
    Truncates top callout badge text cleanly on whole-word boundaries so words are never cut in half.
    """
    words = [w for w in text.split() if w]
    selected: List[str] = []
    total = 0
    for w in words[:max_words]:
        proj = total + (1 if selected else 0) + len(w)
        if selected and proj > max_chars:
            break
        selected.append(w)
        total = proj
    return " ".join(selected) if selected else (words[0][:max_chars] if words else "")


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
    max_chars_two_lines: int = 22,
) -> List[List[WordTiming]]:
    """
    Groups words within each scene into punchy 2-to-3 word chunks that NEVER overflow
    the 1080px horizontal safe zone (950px effective width at 90pt Montserrat Bold).
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
            if len(chunk) >= 2 and projected > 17:
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
    max_single_line_chars: int = 12,
) -> str:
    r"""
    Formats a chunk of 1-3 words with a smooth 75ms elastic active-word gold pop animation
    (`\c&H00D7FF&\fscx100\fscy100\t(0,75,\fscx114\fscy114)`) and inserts an explicit ASS `\N`
    line break when total characters exceed `max_single_line_chars` so text stays centered
    inside the 1080x1920 vertical frame.
    """
    plain_words = [_clean_word(wt.word) for wt in chunk]
    total_chars = sum(len(w) for w in plain_words) + max(0, len(plain_words) - 1)

    if len(plain_words) <= 1 or total_chars <= max_single_line_chars:
        line_groups = [plain_words]
    elif len(plain_words) == 2:
        line_groups = [[plain_words[0]], [plain_words[1]]]
    else:
        split_after_first = abs(len(plain_words[0]) - (len(plain_words[1]) + 1 + len(plain_words[2])))
        split_after_second = abs((len(plain_words[0]) + 1 + len(plain_words[1])) - len(plain_words[2]))
        if split_after_first <= split_after_second:
            line_groups = [[plain_words[0]], [plain_words[1], plain_words[2]]]
        else:
            line_groups = [[plain_words[0], plain_words[1]], [plain_words[2]]]

    max_line_chars = max(len(" ".join(lg)) for lg in line_groups)
    # Automatic dynamic font-size step-down if a line has a very long English word (>13 chars)
    base_fs_tag = "\\fs74" if max_line_chars > 13 else "\\fs90"

    styled_words = []
    for j, w_str in enumerate(plain_words):
        if j == active_idx:
            styled_words.append(
                f"{{\\c&H00D7FF&\\fscx100\\fscy100\\t(0,75,\\fscx114\\fscy114)}}{w_str}{{\\c&HFFFFFF&\\fscx100\\fscy100}}"
            )
        else:
            styled_words.append(w_str)

    entry_prefix = f"{{\\fad(45,0)\\blur0.7{base_fs_tag}}}" if active_idx == 0 else f"{{\\blur0.7{base_fs_tag}}}"

    if len(plain_words) <= 1 or total_chars <= max_single_line_chars:
        return entry_prefix + " ".join(styled_words)

    if len(plain_words) == 2:
        return entry_prefix + f"{styled_words[0]}\\N{styled_words[1]}"

    split_after_first = abs(len(plain_words[0]) - (len(plain_words[1]) + 1 + len(plain_words[2])))
    split_after_second = abs((len(plain_words[0]) + 1 + len(plain_words[1])) - len(plain_words[2]))
    if split_after_first <= split_after_second:
        return entry_prefix + f"{styled_words[0]}\\N{styled_words[1]} {styled_words[2]}"
    return entry_prefix + f"{styled_words[0]} {styled_words[1]}\\N{styled_words[2]}"


def generate_synced_ass_captions(
    word_timings: List[WordTiming],
    scenes: List[dict],
    output_ass: Path,
) -> Path:
    output_ass.parent.mkdir(parents=True, exist_ok=True)

    header = """[Script Info]
Title: US History Short Word-Synced Captions (Pro Creator V2)
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: MainCaption,Montserrat,90,&H00FFFFFF,&H0000D7FF,&H00000000,&H96000000,-1,0,0,0,100,100,1.0,0,1,8,5,2,65,65,530,1
Style: TopBadge,Montserrat,50,&H0000D7FF,&H00FFFFFF,&H00101010,&HD2080808,-1,0,0,0,100,100,2.0,0,3,16,0,8,70,70,215,1

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
        clean_badge = _truncate_badge_words(_clean_word(badge), max_chars=22, max_words=4)
        if not clean_badge or clean_badge == last_badge:
            continue
        last_badge = clean_badge

        s_start = float(scene.get("start_sec", 0.0)) + 0.08
        s_end = min(float(scene.get("end_sec", s_start + 1.5)), s_start + 2.1)
        if s_end > s_start:
            events.append(
                f"Dialogue: 1,{_format_ass_time(s_start)},{_format_ass_time(s_end)},TopBadge,,0,0,0,,{{\\fad(120,160)}}  {clean_badge}  "
            )

    output_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"[CaptionBuilder] Generated {len(events)} Pro Creator 90pt word-synced subtitle events -> {output_ass}")
    return output_ass
