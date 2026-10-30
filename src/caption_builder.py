r"""
Builds dynamic word-synced ASS subtitles in the viral "Pro Creator V5 — Multi-Color Power-Word & Subscribe CTA" style:
- High-impact 90pt Bold Montserrat typography (with automatic step-down to 74pt for extra-long 14+ char lines)
  positioned in the lower-middle "Thumb-Stop Zone" (`MarginV: 530` on `1080x1920`).
- Dynamic 3-Color Active-Word Pop System:
  1. Danger / Action / Shock Power Words (`WAR`, `DEAD`, `EXPLODED`, `BOMB`, `NUCLEAR`, `SECRET`, `FAKE`, `DOOMED`, `LOST`, `CRAZIEST`, `KILL`, `FIRE`, `CRASHED`, `PLAGUE`)
     pop in **Neon Crimson Red (`#FF2E2E` -> ASS `&H2E2EFF&`)** at **118% elastic scale**!
  2. Numbers, Dates, Money & Stats (`$80M`, `1932`, `1944`, `20,000`, `MILLION`, `BILLION`, `100%`, `FIRST`)
     pop in **Electric Emerald Green (`#00FF7F` -> ASS `&H7FFF00&`)** at **116% elastic scale**!
  3. Standard Active Words pop in **Warm Viral Gold (`#FFD700` -> ASS `&H00D7FF&`)** at **114% elastic scale**!
- Animated "SUBSCRIBE FOR DAILY UNTOLD HISTORY" CTA pill in the final 3.0 seconds to convert viewers into subscribers.
"""
import re
from pathlib import Path
from typing import List
from voice_engine import WordTiming

PUNCTUATION_ONLY_RE = re.compile(r"^[,.;:!?—–\-\"'\)\]]+$")
LEADING_PUNCT_RE = re.compile(r"^([,.;:!?]+)(.+)$")

POWER_DANGER_WORDS = {
    "WAR", "DEAD", "DIED", "DEATH", "KILLED", "KILL", "EXPLODED", "EXPLOSION", "BOMB", "NUCLEAR",
    "ATOMIC", "DOOMED", "DESTROYED", "CRASHED", "CRASH", "FAKE", "CORPSE", "SECRET", "SPY", "SPIES",
    "HITLER", "NAZI", "TERROR", "PANIC", "BLOOD", "FIRE", "BURNED", "POISON", "PLAGUE", "CRAZIEST",
    "BIZARRE", "IMPOSSIBLE", "LOST", "VANISHED", "MISSING", "FLOOD", "WAVE", "CRUSHED", "TRAPPED",
    "NEVER", "REFUSED", "STOPPED", "SHOCKING", "INSANE", "DANGEROUS", "LETHAL", "FATAL", "ATTACK",
}

NUMBER_STAT_WORDS = {
    "MILLION", "BILLIONS", "BILLION", "THOUSAND", "HUNDRED", "FIRST", "ZERO", "SINGLE", "DOUBLE",
    "TONS", "MPH", "FEET", "MILES", "HOURS", "MINUTES", "SECONDS", "YEARS", "CENTURY",
}


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


def _get_word_pop_tag(word_str: str) -> str:
    """
    Returns the ASS override tag for an active word based on whether it is a Power/Danger word
    (Neon Red 118% pop), a Number/Money/Stat word (Electric Emerald Green 116% pop), or a standard
    word (Warm Viral Gold 114% pop).
    """
    bare = re.sub(r"[^A-Z0-9$%]", "", word_str.upper())
    if bare in POWER_DANGER_WORDS:
        # Neon Crimson Red (#FF3333 -> BGR &H3333FF&) + 118% elastic pop
        return r"{\c&H3333FF&\fscx100\fscy100\t(0,70,\fscx118\fscy118)}"
    if any(ch.isdigit() or ch in "$%" for ch in bare) or bare in NUMBER_STAT_WORDS:
        # Electric Emerald Green (#00FF7F -> BGR &H7FFF00&) + 116% elastic pop
        return r"{\c&H7FFF00&\fscx100\fscy100\t(0,70,\fscx116\fscy116)}"
    # Standard Warm Viral Gold (#FFD700 -> BGR &H00D7FF&) + 114% elastic pop
    return r"{\c&H00D7FF&\fscx100\fscy100\t(0,75,\fscx114\fscy114)}"


def _truncate_badge_words(text: str, max_chars: int = 22, max_words: int = 4) -> str:
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
    base_fs_tag = "\\fs74" if max_line_chars > 13 else "\\fs90"

    styled_words = []
    for j, w_str in enumerate(plain_words):
        if j == active_idx:
            pop_tag = _get_word_pop_tag(w_str)
            styled_words.append(
                f"{pop_tag}{w_str}{{\\c&HFFFFFF&\\fscx100\\fscy100}}"
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
Title: US History Short Word-Synced Captions (Pro Creator V5 Flashy Retention)
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: MainCaption,Montserrat,90,&H00FFFFFF,&H0000D7FF,&H00000000,&H96000000,-1,0,0,0,100,100,1.0,0,1,8,5,2,65,65,530,1
Style: TopBadge,Montserrat,50,&H0000D7FF,&H00FFFFFF,&H00101010,&HD2080808,-1,0,0,0,100,100,2.0,0,3,16,0,8,70,70,215,1
Style: SubscribeCTA,Montserrat,44,&H00FFFFFF,&H0000D7FF,&H001414CC,&HE61010B0,-1,0,0,0,100,100,2.0,0,3,16,0,8,70,70,310,1

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
    total_end_sec = 0.0
    for scene in scenes:
        s_end_val = float(scene.get("end_sec", 0.0))
        if s_end_val > total_end_sec:
            total_end_sec = s_end_val

        badge = (scene.get("on_screen_callout") or "").strip().upper()
        if not badge:
            continue
        clean_badge = _truncate_badge_words(_clean_word(badge), max_chars=22, max_words=4)
        if not clean_badge or clean_badge == last_badge:
            continue
        last_badge = clean_badge

        s_start = float(scene.get("start_sec", 0.0)) + 0.06
        s_end = min(s_end_val, s_start + 2.1)
        if s_end > s_start:
            events.append(
                f"Dialogue: 1,{_format_ass_time(s_start)},{_format_ass_time(s_end)},TopBadge,,0,0,0,,{{\\fad(90,140)\\fscx90\\fscy90\\t(0,90,\\fscx100\\fscy100)}}  {clean_badge}  "
            )

    # Add an animated Subscribe CTA pill in the final 2.8 seconds to convert viewers into subscribers!
    if total_end_sec > 6.0:
        cta_start = max(0.0, total_end_sec - 2.8)
        events.append(
            f"Dialogue: 3,{_format_ass_time(cta_start)},{_format_ass_time(total_end_sec)},SubscribeCTA,,0,0,0,,"
            r"{\fad(120,120)\fscx85\fscy85\t(0,140,\fscx104\fscy104)\t(140,260,\fscx100\fscy100)}  SUBSCRIBE FOR DAILY HISTORY  "
        )

    output_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"[CaptionBuilder] Generated {len(events)} Pro Creator V5 multi-color word-synced subtitle events -> {output_ass}")
    return output_ass
