"""
Synthesizes:
1. Emotional, uplifting "Interstellar / Viral Documentary" soundtrack (`interstellar_bgm.wav` & `mystery_bgm.wav`)
   inspired by Hans Zimmer's "Cornfield Chase" / "Time".
2. Netflix-Style 0.0s Hook Impact + Sub-Bass Pulse Track (`hook_impact.wav`):
   Gives the opening 1.2 seconds of the Short a deep cinematic sub-bass thud + subtle metallic shimmer riser
   so viewers physically stop scrolling on Frame 1, plus subtle low-end tension pulses at 12s, 24s, and 36s!
"""
from pathlib import Path
import wave
import numpy as np

SFX_DIR = Path("assets/sfx")
BGM_DIR = Path("assets/bgm")


def save_wav(filename: Path, samples: np.ndarray, sample_rate: int = 44100):
    filename.parent.mkdir(parents=True, exist_ok=True)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    if samples.ndim == 1:
        max_val = float(np.max(np.abs(samples)))
        if max_val > 0:
            samples = samples / max_val * 0.90
        int_samples = (samples * 32767).astype(np.int16)
        stereo = np.column_stack((int_samples, int_samples)).flatten()
    else:
        max_val = float(np.max(np.abs(samples)))
        if max_val > 0:
            samples = samples / max_val * 0.90
        int_samples = (samples * 32767).astype(np.int16)
        stereo = int_samples.flatten()

    with wave.open(str(filename), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


def generate_hook_impact_track(duration_sec: float = 60.0, sr: int = 44100) -> np.ndarray:
    """
    Synthesizes a 60-second stereo accent track containing:
    - A deep, theater-grade 0.0s sub-bass impact drop (90Hz -> 36Hz pitch sweep) + subtle high shimmer
      in the first 1.1 seconds to stop viewers from swiping away.
    - Subtle low-frequency documentary tension pulses at t=12s, t=24s, t=36s, and t=46s.
    """
    n_samples = int(sr * duration_sec)
    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)

    # 1. Opening 0.0s - 1.2s Hook Sub-Bass Impact Drop
    hit_len = int(1.2 * sr)
    t_hit = np.linspace(0, 1.2, hit_len, endpoint=False)
    freq_sweep = 36.0 + 62.0 * np.exp(-9.5 * t_hit)
    phase = 2 * np.pi * np.cumsum(freq_sweep) / sr
    sub_drop = np.sin(phase) * np.exp(-3.2 * t_hit) * np.minimum(1.0, t_hit / 0.012)
    shimmer_l = 0.14 * np.sin(2 * np.pi * 440.0 * t_hit) * np.exp(-5.5 * t_hit)
    shimmer_r = 0.14 * np.sin(2 * np.pi * 659.25 * t_hit) * np.exp(-5.5 * t_hit)

    left[:hit_len] += 0.75 * sub_drop + shimmer_l
    right[:hit_len] += 0.75 * sub_drop + shimmer_r

    # 2. Subtle documentary bass pulses at twist points (12s, 24s, 36s, 46s)
    for beat_sec in (12.0, 24.0, 36.0, 46.0):
        if beat_sec + 0.9 >= duration_sec:
            continue
        i0 = int(beat_sec * sr)
        p_len = int(0.85 * sr)
        tp = np.linspace(0, 0.85, p_len, endpoint=False)
        pf = 38.0 + 35.0 * np.exp(-10.0 * tp)
        p_phase = 2 * np.pi * np.cumsum(pf) / sr
        pulse = 0.42 * np.sin(p_phase) * np.exp(-4.0 * tp) * np.minimum(1.0, tp / 0.015)
        left[i0 : i0 + p_len] += pulse
        right[i0 : i0 + p_len] += pulse

    return np.column_stack((left, right))


def _piano_note(freq: float, dur_sec: float, sr: int = 44100) -> np.ndarray:
    n = int(sr * dur_sec)
    t = np.linspace(0, dur_sec, n, endpoint=False)

    attack = np.minimum(1.0, t / 0.008)
    decay = 0.68 * np.exp(-4.2 * t) + 0.32 * np.exp(-1.1 * t)
    env = attack * decay

    h1 = 1.00 * np.sin(2 * np.pi * freq * t)
    h2 = 0.42 * np.sin(2 * np.pi * (freq * 2.0005) * t) * np.exp(-3.0 * t)
    h3 = 0.16 * np.sin(2 * np.pi * (freq * 3.0015) * t) * np.exp(-6.0 * t)
    h4 = 0.06 * np.sin(2 * np.pi * (freq * 4.0030) * t) * np.exp(-10.0 * t)

    return (h1 + h2 + h3 + h4) * env


def _pipe_organ_voice(freq: float, t: np.ndarray) -> np.ndarray:
    return (
        0.35 * np.sin(2 * np.pi * (freq * 0.5) * t)
        + 0.55 * np.sin(2 * np.pi * freq * t)
        + 0.28 * np.sin(2 * np.pi * (freq * 2.0) * t)
        + 0.10 * np.sin(2 * np.pi * (freq * 3.0) * t)
        + 0.08 * np.sin(2 * np.pi * (freq * 4.0) * t)
    )


def generate_interstellar_uplifting_bgm(duration_sec: float = 68.0, sr: int = 44100) -> np.ndarray:
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    progression = [
        {
            "bass": 87.31,
            "organ": [174.61, 220.00, 261.63, 329.63],
            "arp": [523.25, 659.25, 440.00, 659.25, 523.25, 659.25, 440.00, 659.25],
            "string_lead": 659.25,
        },
        {
            "bass": 98.00,
            "organ": [196.00, 246.94, 293.66, 329.63],
            "arp": [493.88, 659.25, 392.00, 659.25, 493.88, 659.25, 392.00, 659.25],
            "string_lead": 587.33,
        },
        {
            "bass": 110.00,
            "organ": [220.00, 261.63, 329.63, 493.88],
            "arp": [523.25, 659.25, 440.00, 659.25, 493.88, 659.25, 440.00, 659.25],
            "string_lead": 659.25,
        },
        {
            "bass": 82.41,
            "organ": [164.81, 196.00, 261.63, 329.63],
            "arp": [523.25, 783.99, 659.25, 783.99, 523.25, 783.99, 659.25, 783.99],
            "string_lead": 783.99,
        },
    ]

    step_sec = 60.0 / 112.0 / 2.0
    notes_per_chord = 8
    chord_dur = step_sec * notes_per_chord

    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)

    total_chords = int(np.ceil(duration_sec / chord_dur)) + 1

    for c_idx in range(total_chords):
        chord = progression[c_idx % len(progression)]
        c_start = c_idx * chord_dur
        c_end = min(duration_sec, (c_idx + 1) * chord_dur)
        if c_start >= duration_sec:
            break

        i0 = int(c_start * sr)
        i1 = int(c_end * sr)
        seg_len = i1 - i0
        if seg_len <= 0:
            continue
        seg_t = t[i0:i1] - c_start

        cycle_swell = 0.72 + 0.28 * float(np.sin(np.pi * min(1.0, c_start / 45.0)))

        c_env = np.ones(seg_len, dtype=np.float64)
        fade_len = min(int(0.18 * sr), seg_len // 3)
        if fade_len > 0:
            c_env[:fade_len] = np.linspace(0.15, 1.0, fade_len)
            c_env[-fade_len:] = np.linspace(1.0, 0.15, fade_len)

        bass_f = chord["bass"]
        bass_wave = (
            0.55 * np.sin(2 * np.pi * bass_f * seg_t)
            + 0.30 * np.sin(2 * np.pi * (bass_f * 2.0) * seg_t)
        ) * c_env * 0.26 * cycle_swell

        org_l = np.zeros(seg_len, dtype=np.float64)
        org_r = np.zeros(seg_len, dtype=np.float64)
        for v_idx, freq in enumerate(chord["organ"]):
            detune = 1.0018 if v_idx % 2 == 0 else 0.9982
            breathe = 0.85 + 0.15 * np.sin(2 * np.pi * 0.35 * seg_t + v_idx)
            org_l += 0.14 * _pipe_organ_voice(freq, seg_t) * breathe
            org_r += 0.14 * _pipe_organ_voice(freq * detune, seg_t + 0.01) * breathe

        s_freq = chord["string_lead"]
        vib = 1.0 + 0.0025 * np.sin(2 * np.pi * 5.2 * seg_t)
        s_env = np.maximum(0.0, np.sin(np.pi * seg_t / max(c_end - c_start, 0.1))) ** 0.7
        str_l = 0.09 * (
            np.sin(2 * np.pi * s_freq * vib * seg_t)
            + 0.35 * np.sin(2 * np.pi * (s_freq * 2.0) * vib * seg_t)
        ) * s_env * cycle_swell
        str_r = 0.09 * (
            np.sin(2 * np.pi * (s_freq * 1.002) * vib * seg_t + 0.5)
            + 0.35 * np.sin(2 * np.pi * (s_freq * 2.004) * vib * seg_t + 0.5)
        ) * s_env * cycle_swell

        left[i0:i1] += bass_wave + org_l * c_env * cycle_swell + str_l
        right[i0:i1] += bass_wave + org_r * c_env * cycle_swell + str_r

        arp_notes = chord["arp"]
        note_ring_dur = step_sec * 2.4
        for n_idx, n_freq in enumerate(arp_notes):
            n_start_s = c_start + n_idx * step_sec
            if n_start_s >= duration_sec:
                break
            ni0 = int(n_start_s * sr)
            raw_note = _piano_note(n_freq, note_ring_dur, sr=sr)
            ni1 = min(n_samples, ni0 + len(raw_note))
            actual_len = ni1 - ni0
            if actual_len <= 0:
                continue

            vel = 0.34 if (n_idx % 4 == 0) else 0.24
            pan_l = 0.62 if (n_idx % 2 == 0) else 0.38
            pan_r = 1.0 - pan_l

            left[ni0:ni1] += raw_note[:actual_len] * vel * pan_l
            right[ni0:ni1] += raw_note[:actual_len] * vel * pan_r

    delay1 = int(0.268 * sr)
    delay2 = int(0.536 * sr)
    rev_l = np.zeros_like(left)
    rev_r = np.zeros_like(right)
    rev_l[delay1:] += right[:-delay1] * 0.24
    rev_r[delay1:] += left[:-delay1] * 0.24
    rev_l[delay2:] += left[:-delay2] * 0.14
    rev_r[delay2:] += right[:-delay2] * 0.14

    left = left + rev_l
    right = right + rev_r

    fi = int(0.5 * sr)
    fo = int(1.8 * sr)
    left[:fi] *= np.linspace(0.0, 1.0, fi)
    right[:fi] *= np.linspace(0.0, 1.0, fi)
    left[-fo:] *= np.linspace(1.0, 0.0, fo)
    right[-fo:] *= np.linspace(1.0, 0.0, fo)

    return np.column_stack((left, right))


def generate_all_sfx():
    sr = 44100
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    BGM_DIR.mkdir(parents=True, exist_ok=True)

    interstellar_stereo = generate_interstellar_uplifting_bgm(duration_sec=68.0, sr=sr)
    save_wav(SFX_DIR / "interstellar_bgm.wav", interstellar_stereo, sr)
    save_wav(SFX_DIR / "mystery_bgm.wav", interstellar_stereo, sr)
    save_wav(SFX_DIR / "ambient_bed.wav", interstellar_stereo, sr)

    hook_impact = generate_hook_impact_track(duration_sec=60.0, sr=sr)
    save_wav(SFX_DIR / "hook_impact.wav", hook_impact, sr)

    silent = np.zeros(int(sr * 0.10), dtype=np.float64)
    for legacy_name in ("whoosh.wav", "pop.wav", "boom.wav", "shutter.wav"):
        save_wav(SFX_DIR / legacy_name, silent, sr)

    print(
        "[AudioEngine] Generated Interstellar score + 0.0s Hook Impact sub-bass drop (assets/sfx/hook_impact.wav)."
    )


if __name__ == "__main__":
    generate_all_sfx()
