"""
Synthesizes a rich, dark, mysterious documentary background score (`mystery_bgm.wav` & `ambient_bed.wav`)
for US History Shorts — with ZERO cheesy cartoon transition sound effects (no zoop, whoosh, pop, or shutter).

Musical Architecture (D Minor Mysterious Cinematic Score @ 68 seconds):
- Deep Sub-Bass Pulse (D1 -> Bb0 -> G0 -> A0 progression with slow tension breath)
- Dark Haunting String / Synth Pad (D minor 9th / Eb major 7th / G minor / A7b9 tension chords)
- Eerie High Harmonic Shimmer & Bowed Glass Texture (slowly evolving detuned fifths)
- Subtle Heartbeat / Ticking Clock Sub-Pulse (60 BPM cinematic documentary tension)
"""
from pathlib import Path
import wave
import numpy as np

SFX_DIR = Path("assets/sfx")


def save_wav(filename: Path, samples: np.ndarray, sample_rate: int = 44100):
    filename.parent.mkdir(parents=True, exist_ok=True)
    if samples.ndim == 1:
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val * 0.88
        int_samples = (samples * 32767).astype(np.int16)
        stereo = np.column_stack((int_samples, int_samples)).flatten()
    else:
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val * 0.88
        int_samples = (samples * 32767).astype(np.int16)
        stereo = int_samples.flatten()

    with wave.open(str(filename), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


def _one_pole_lowpass(signal: np.ndarray, cutoff_hz: float, sr: int = 44100) -> np.ndarray:
    rc = 1.0 / (2.0 * np.pi * max(cutoff_hz, 20.0))
    dt = 1.0 / sr
    alpha = dt / (rc + dt)
    out = np.zeros_like(signal)
    prev = 0.0
    for i in range(len(signal)):
        prev = prev + alpha * (signal[i] - prev)
        out[i] = prev
    return out


def generate_mysterious_bgm(duration_sec: float = 68.0, sr: int = 44100) -> np.ndarray:
    """
    Synthesizes a stereo dark mysterious documentary background music track in D minor.
    """
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    np.random.seed(1944)

    # 4-chord progression cycling every 16 seconds (4 seconds per chord):
    # Dm9 -> Bbmaj7#11 -> Gm6 -> A7(b9)
    chords = [
        # (bass_hz, [pad_hz_1, pad_hz_2, pad_hz_3, shimmer_hz])
        (36.71, [146.83, 174.61, 220.00, 329.63, 587.33]),  # D minor 9
        (29.14, [116.54, 146.83, 174.61, 261.63, 523.25]),  # Bb major 7 (#11)
        (24.50, [98.00, 116.54, 146.83, 220.00, 440.00]),   # G minor 6
        (27.50, [110.00, 138.59, 164.81, 233.08, 466.16]),  # A7(b9) suspense
    ]

    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)

    chord_len_sec = 4.0
    total_chords = int(np.ceil(duration_sec / chord_len_sec))

    for idx in range(total_chords):
        start_s = idx * chord_len_sec
        end_s = min(duration_sec, (idx + 1) * chord_len_sec)
        i0 = int(start_s * sr)
        i1 = int(end_s * sr)
        if i1 <= i0:
            continue

        seg_t = t[i0:i1] - start_s
        seg_dur = end_s - start_s
        bass_hz, pad_freqs = chords[idx % len(chords)]

        # Smooth crossfade envelope per chord
        env = np.ones_like(seg_t)
        fade_n = min(int(0.45 * sr), len(seg_t) // 3)
        if fade_n > 0:
            env[:fade_n] = np.linspace(0.0, 1.0, fade_n) ** 1.2
            env[-fade_n:] = np.linspace(1.0, 0.0, fade_n) ** 1.2

        # 1. Deep dark sub-bass drone with subtle warm 2nd harmonic
        lfo_bass = 0.85 + 0.15 * np.sin(2 * np.pi * 0.25 * seg_t)
        sub_bass = (
            0.70 * np.sin(2 * np.pi * bass_hz * seg_t)
            + 0.30 * np.sin(2 * np.pi * (bass_hz * 2.0) * seg_t)
        ) * lfo_bass * env

        # 2. Stereo detuned dark cello/synth pad
        pad_l = np.zeros_like(seg_t)
        pad_r = np.zeros_like(seg_t)
        for f_idx, freq in enumerate(pad_freqs[:-1]):
            detune = 1.0025 if f_idx % 2 == 0 else 0.9975
            slow_mod = 0.75 + 0.25 * np.sin(2 * np.pi * (0.11 + 0.04 * f_idx) * seg_t + f_idx)
            weight = 0.28 / (1.0 + 0.25 * f_idx)
            pad_l += weight * np.sin(2 * np.pi * freq * seg_t) * slow_mod
            pad_r += weight * np.sin(2 * np.pi * (freq * detune) * seg_t + 0.35) * slow_mod

        # 3. Eerie high "bowed glass" mystery motif (plays softly on odd/even bars)
        shimmer_freq = pad_freqs[-1]
        shimmer_env = np.sin(np.pi * seg_t / max(seg_dur, 0.1)) ** 2
        vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 4.2 * seg_t)
        shimmer_l = 0.07 * np.sin(2 * np.pi * shimmer_freq * vibrato * seg_t) * shimmer_env
        shimmer_r = 0.07 * np.sin(2 * np.pi * (shimmer_freq * 1.003) * vibrato * seg_t + 0.8) * shimmer_env

        left[i0:i1] += 0.48 * sub_bass + pad_l * env + shimmer_l * env
        right[i0:i1] += 0.48 * sub_bass + pad_r * env + shimmer_r * env

    # 4. Subtle dark cinematic heartbeat / suspense pulse (every 1.0s = 60 BPM)
    pulse_period = 1.0
    n_pulses = int(duration_sec / pulse_period)
    pulse_track = np.zeros(n_samples, dtype=np.float64)
    for p in range(n_pulses):
        p_start = int(p * pulse_period * sr)
        p_len = int(0.28 * sr)
        if p_start + p_len >= n_samples:
            break
        pt = np.linspace(0, 0.28, p_len, endpoint=False)
        # Soft muffled low tom / heartbeat thump (58Hz -> 34Hz)
        pfreq = 58.0 * np.exp(-9.0 * pt) + 34.0
        pphase = 2 * np.pi * np.cumsum(pfreq) / sr
        penv = np.exp(-14.0 * pt) * (np.sin(np.pi * pt / 0.28) ** 0.5)
        pulse_track[p_start : p_start + p_len] += 0.16 * np.sin(pphase) * penv

    # 5. Subtle filtered vinyl / archival room air for documentary warmth
    raw_air = np.random.normal(0, 1, n_samples)
    filtered_air = _one_pole_lowpass(raw_air, cutoff_hz=320.0, sr=sr) * 0.025

    left = left + pulse_track + filtered_air
    right = right + pulse_track - filtered_air

    # Master fade-in (0.8s) and fade-out (1.5s)
    fi = int(0.8 * sr)
    fo = int(1.5 * sr)
    left[:fi] *= np.linspace(0.0, 1.0, fi)
    right[:fi] *= np.linspace(0.0, 1.0, fi)
    left[-fo:] *= np.linspace(1.0, 0.0, fo)
    right[-fo:] *= np.linspace(1.0, 0.0, fo)

    return np.column_stack((left, right))


def generate_all_sfx():
    sr = 44100
    SFX_DIR.mkdir(parents=True, exist_ok=True)

    # Generate the primary dark mysterious documentary score
    mystery_stereo = generate_mysterious_bgm(duration_sec=68.0, sr=sr)
    save_wav(SFX_DIR / "mystery_bgm.wav", mystery_stereo, sr)
    # Keep ambient_bed.wav pointing to the same mysterious score for backwards compatibility
    save_wav(SFX_DIR / "ambient_bed.wav", mystery_stereo, sr)

    # Generate silent 0.1s placeholders for legacy SFX filenames so any old scripts checking file existence never break,
    # while ensuring zero whoosh/zoop/pop noises can ever play.
    silent = np.zeros(int(sr * 0.10), dtype=np.float64)
    for legacy_name in ("whoosh.wav", "pop.wav", "boom.wav", "shutter.wav"):
        save_wav(SFX_DIR / legacy_name, silent, sr)

    print("[AudioEngine] Generated dark mysterious documentary score (assets/sfx/mystery_bgm.wav) with zero transition SFX.")


if __name__ == "__main__":
    generate_all_sfx()
