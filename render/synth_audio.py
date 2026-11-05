"""Trilha sintetizada (numpy, sem arquivos externos) para o video: baixo pulsante, arpejo menor, hi-hat,
vento leve e motor em marcha lenta durante o plano do carro. A energia acompanha os tres planos."""
import wave

import numpy as np

SR = 44100


def build(duration=15.0, cut_ab=4.5, cut_bc=9.5, bpm=96, seed=5):
    n = int(SR * duration)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    buf = np.zeros(n)
    beat = 60.0 / bpm

    def energy(x):  # intensidade por plano
        return 0.55 if x < cut_ab else (0.8 if x < cut_bc else 1.0)

    def add(sig, start, gain=1.0):
        i = int(start * SR)
        if 0 <= i < n:
            j = min(n, i + len(sig))
            buf[i:j] += gain * sig[:j - i]

    kt = np.arange(int(0.32 * SR)) / SR
    kick = np.sin(2 * np.pi * (44 * kt + (62 / 26.0) * (1 - np.exp(-26 * kt)))) * np.exp(-kt * 9)
    ht = np.arange(int(0.05 * SR)) / SR
    hat = np.diff(rng.standard_normal(len(ht)), prepend=0.0) * np.exp(-ht * 80)
    notes = [220.0, 261.63, 329.63, 392.0, 329.63, 261.63, 440.0, 392.0]

    def pluck(f, dur=0.4):
        pt = np.arange(int(dur * SR)) / SR
        s = np.sin(2 * np.pi * f * pt) + 0.4 * np.sin(4 * np.pi * f * pt) + 0.2 * np.sin(6 * np.pi * f * pt)
        return s * np.exp(-pt * 9)

    b = 0.0
    while b < duration:
        add(kick, b, 0.85 * energy(b))
        add(hat, b + beat / 2, 0.16 * energy(b))
        b += beat
    k, p = 0, 0.0
    while p < duration:
        add(pluck(notes[k % len(notes)]), p, 0.09 * energy(p))
        k += 1
        p += beat / 2

    pad = sum(np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) for f in (110.0, 130.81, 164.81, 196.0))
    buf += pad * 0.035 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.2 * t))
    duck = 1 - 0.7 * np.exp(-((t % beat)) * 10)
    env = np.array([energy(x) for x in t[::441]]).repeat(441)[:n]
    buf += np.sin(2 * np.pi * 55 * t) * 0.15 * duck * env

    # motor em marcha lenta so no plano B (carro estacionado), com entrada/saida suaves
    gate = np.clip(np.minimum((t - cut_ab) / 0.3, (cut_bc - t) / 0.3), 0, 1)
    engine = (np.sin(2 * np.pi * 38 * t) + 0.5 * np.sin(2 * np.pi * 76 * t) + 0.25 * np.sin(2 * np.pi * 114 * t))
    buf += engine * (0.6 + 0.4 * np.sin(2 * np.pi * 11 * t)) * 0.09 * gate

    wind = np.convolve(rng.standard_normal(n), np.ones(60) / 60, mode="same")  # ruido grave
    buf += wind * 0.05

    buf *= np.minimum(1, t / 0.5) * np.minimum(1, (duration - t) / 0.8)
    buf *= 0.6 / np.max(np.abs(buf))
    delay = int(0.012 * SR)
    right = np.concatenate([np.zeros(delay), buf[:-delay]])
    return np.stack([buf, right], axis=1)


def write_wav(path, duration=15.0, **kw):
    pcm = (np.clip(build(duration, **kw), -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
