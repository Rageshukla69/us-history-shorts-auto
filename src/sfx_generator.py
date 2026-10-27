"""
Synthesizes royalty-free procedural sound effects (whoosh, pop, boom, shutter) and an ambient documentary drone bed.
"""
from pathlib import Path
import wave
import numpy as np

SFX_DIR = Path("assets/sfx")


def save_wav(filename: Path, samples: np.ndarray, sample_rate: int = 44100):
    filename.parent.mkdir(parents=True, exist_ok=True)
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val * 0.85
    int_samples = (samples * 32767).astype(np.int16)
    with wave.open(str(filename), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        stereo = np.column_stack((int_samples, int_samples)).flatten()
        wf.writeframes(stereo.tobytes())


def generate_all_sfx():
    sr = 44100
    SFX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fast Whoosh
    dur = 0.22
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    np.random.seed(42)
    noise = np.random.normal(0, 1, len(t))
    env = np.sin(np.pi * t / dur) ** 2
    sweep = np.sin(2 * np.pi * (180 + 900 * (t / dur) ** 1.5) * t)
    whoosh = (0.65 * noise + 0.35 * sweep) * env * 0.45
    save_wav(SFX_DIR / "whoosh.wav", whoosh, sr)

    # 2. Subtle Pop / Cut Tick
    dur = 0.08
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    freq = np.linspace(520, 110, len(t))
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.exp(-35 * t)
    pop = np.sin(phase) * env * 0.5
    save_wav(SFX_DIR / "pop.wav", pop, sr)

    # 3. Cinematic Bass Hit / Boom
    dur = 0.55
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    freq = 95 * np.exp(-4.5 * t) + 32
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.exp(-5.0 * t)
    boom = np.tanh(1.8 * np.sin(phase) * env) * 0.6
    save_wav(SFX_DIR / "boom.wav", boom, sr)

    # 4. Archival Camera Shutter
    dur = 0.10
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    click1 = np.exp(-150 * t) * np.random.normal(0, 1, len(t))
    click2 = np.roll(click1, int(0.035 * sr)) * 0.7
    click2[:int(0.035 * sr)] = 0
    shutter = (click1 + click2) * 0.55
    save_wav(SFX_DIR / "shutter.wav", shutter, sr)

    # 5. Ambient Documentary Bed (65s)
    dur = 65.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.12 * t)
    pad = (
        0.40 * np.sin(2 * np.pi * 55.0 * t)
        + 0.30 * np.sin(2 * np.pi * 82.41 * t)
        + 0.20 * np.sin(2 * np.pi * 110.0 * t) * lfo
        + 0.10 * np.sin(2 * np.pi * 130.81 * t) * (1.0 - 0.4 * lfo)
    )
    beat_env = 0.85 + 0.15 * (np.sin(2 * np.pi * 2.0 * t) ** 4)
    bed = pad * beat_env * 0.22
    save_wav(SFX_DIR / "ambient_bed.wav", bed, sr)
    print("Generated procedural SFX and ambient documentary bed in assets/sfx/")


if __name__ == "__main__":
    generate_all_sfx()
