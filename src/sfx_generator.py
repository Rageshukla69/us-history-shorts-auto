"""
Synthesizes (v7.0 — Dual-Camera Documentary Sound Design: Scene Transition Whooshes + Axial Close-Up Shutter Clicks):
1. Emotional, uplifting "Interstellar / Viral Documentary" soundtrack (`interstellar_bgm.wav` & `mystery_bgm.wav`).
2. Timestamp-Synced Dual-Camera SFX Track (`generate_scene_transition_sfx_track`):
   - Fires a deep sub-bass impact drop + metallic shimmer on Frame 1 (`0.0s`) to stop scrolling.
   - Places a warm cinematic air-whoosh at the exact `start_sec` of every new story beat (`scenes[1:]`).
   - Places a subtle, crisp analog camera lens/shutter transient at the `52%` mark (`start_sec + dur * 0.52`)
     where `VideoComposer` cuts from Shot A (Wide Establishing) to Shot B (`1.32x` Tight Close-Up)!
"""
from pathlib import Path
from typing import List
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
            samples = samples / max_val * 0.92
        int_samples = (samples * 32767).astype(np.int16)
        stereo = np.column_stack((int_samples, int_samples)).flatten()
    else:
        max_val = float(np.max(np.abs(samples)))
        if max_val > 0:
            samples = samples / max_val * 0.92
        int_samples = (samples * 32767).astype(np.int16)
        stereo = int_samples.flatten()

    with wave.open(str(filename), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


def generate_hook_impact_track(duration_sec: float = 60.0, sr: int = 44100) -> np.ndarray:
    n_samples = int(sr * duration_sec)
    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)

    hit_len = int(1.25 * sr)
    t_hit = np.linspace(0, 1.25, hit_len, endpoint=False)
    freq_sweep = 38.0 + 68.0 * np.exp(-9.0 * t_hit)
    phase = 2 * np.pi * np.cumsum(freq_sweep) / sr
    sub_drop = np.sin(phase) * np.exp(-3.0 * t_hit) * np.minimum(1.0, t_hit / 0.010)
    shimmer_l = 0.16 * np.sin(2 * np.pi * 440.0 * t_hit) * np.exp(-5.0 * t_hit)
    shimmer_r = 0.16 * np.sin(2 * np.pi * 659.25 * t_hit) * np.exp(-5.0 * t_hit)

    left[:hit_len] += 0.82 * sub_drop + shimmer_l
    right[:hit_len] += 0.82 * sub_drop + shimmer_r

    for beat_sec in (12.0, 24.0, 36.0, 46.0):
        if beat_sec + 0.9 >= duration_sec:
            continue
        i0 = int(beat_sec * sr)
        p_len = int(0.85 * sr)
        tp = np.linspace(0, 0.85, p_len, endpoint=False)
        pf = 38.0 + 35.0 * np.exp(-10.0 * tp)
        p_phase = 2 * np.pi * np.cumsum(pf) / sr
        pulse = 0.45 * np.sin(p_phase) * np.exp(-4.0 * tp) * np.minimum(1.0, tp / 0.015)
        left[i0 : i0 + p_len] += pulse
        right[i0 : i0 + p_len] += pulse

    return np.column_stack((left, right))


def generate_scene_transition_sfx_track(
    scenes: List[dict], total_duration_sec: float, out_wav: Path, sr: int = 44100
) -> Path:
    dur = max(15.0, float(total_duration_sec) + 1.5)
    n_samples = int(sr * dur)
    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)
    rng = np.random.default_rng(1967)

    # 1. 0.0s Hook Impact Drop
    hit_len = min(n_samples, int(1.25 * sr))
    t_hit = np.linspace(0, 1.25, hit_len, endpoint=False)
    freq_sweep = 38.0 + 72.0 * np.exp(-8.5 * t_hit)
    phase = 2 * np.pi * np.cumsum(freq_sweep) / sr
    sub_drop = np.sin(phase) * np.exp(-2.8 * t_hit) * np.minimum(1.0, t_hit / 0.008)
    left[:hit_len] += 0.85 * sub_drop
    right[:hit_len] += 0.85 * sub_drop

    # 2. Story-Beat Transition Whooshes + Mid-Beat Axial Camera Cut Transients
    for idx, scene in enumerate(scenes):
        cut_sec = float(scene.get("start_sec", 0.0))
        scene_dur = float(scene.get("duration_sec", 3.5))

        # A. Story-beat transition air-whoosh (on scenes[1:])
        if idx > 0 and 0.3 < cut_sec < dur - 0.5:
            w_dur = 0.25
            w_len = int(w_dur * sr)
            w_start = max(0, int((cut_sec - 0.10) * sr))
            w_end = min(n_samples, w_start + w_len)
            actual_len = w_end - w_start
            if actual_len > 100:
                tw = np.linspace(0, w_dur, actual_len, endpoint=False)
                env = np.sin(np.pi * tw / w_dur) ** 1.6
                white = rng.standard_normal(actual_len)
                kernel_size = 28 + (idx % 3) * 10
                kernel = np.ones(kernel_size) / kernel_size
                filtered_air = np.convolve(white, kernel, mode="same")

                sweep_f = (110.0 + (idx % 3) * 20.0) + 150.0 * np.sin(np.pi * tw / w_dur)
                sweep_phase = 2 * np.pi * np.cumsum(sweep_f) / sr
                tonal_whoosh = np.sin(sweep_phase) * 0.30
                whoosh_sig = (filtered_air * 0.36 + tonal_whoosh) * env * 0.34

                pan_l = 0.58 if idx % 2 == 0 else 0.42
                left[w_start:w_end] += whoosh_sig * pan_l
                right[w_start:w_end] += whoosh_sig * (1.0 - pan_l)

            # Deep documentary bass pulse on midpoint & climax beats
            if idx in (4, 8) and cut_sec + 0.85 < dur:
                p_start = int(cut_sec * sr)
                p_len = int(0.80 * sr)
                tp = np.linspace(0, 0.80, p_len, endpoint=False)
                pf = 40.0 + 42.0 * np.exp(-9.0 * tp)
                p_phase = 2 * np.pi * np.cumsum(pf) / sr
                pulse = 0.50 * np.sin(p_phase) * np.exp(-3.6 * tp) * np.minimum(1.0, tp / 0.012)
                left[p_start : p_start + p_len] += pulse
                right[p_start : p_start + p_len] += pulse

        # B. Mid-Beat Axial Camera Cut Transient (`start_sec + scene_dur * 0.52`) when `scene_dur >= 2.6`
        if scene_dur >= 2.6:
            axial_sec = cut_sec + round(scene_dur * 0.52, 3)
            if 0.5 < axial_sec < dur - 0.3:
                a_start = max(0, int((axial_sec - 0.04) * sr))
                a_len = min(int(0.10 * sr), n_samples - a_start)
                if a_len > 40:
                    ta = np.linspace(0, 0.10, a_len, endpoint=False)
                    # Soft low-frequency lens thump + warm analog shutter click
                    lens_thump = np.sin(2 * np.pi * (85.0 * np.exp(-25.0 * ta)) * ta) * np.exp(-38.0 * ta) * 0.22
                    left[a_start : a_start + a_len] += lens_thump
                    right[a_start : a_start + a_len] += lens_thump

    stereo = np.column_stack((left, right))
    save_wav(out_wav, stereo, sample_rate=sr)
    return out_wav


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
    print(f"[SFX] Generated v7.0 Dual-Camera Documentary Score + Hook Impact -> {SFX_DIR}")


if __name__ == "__main__":
    generate_all_sfx()
