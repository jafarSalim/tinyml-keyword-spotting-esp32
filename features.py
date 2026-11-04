"""Log-mel features. firmware/tiny_kws.h computes the same thing in C++."""
import numpy as np

SR, CLIP, WIN, HOP, MELS = 8000, 6400, 256, 128, 20
FRAMES = 1 + (CLIP - WIN) // HOP


def mel_filters():
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(100), hz2mel(SR / 2), MELS + 2))
    bins = np.floor((WIN + 1) * pts / SR).astype(int)
    fb = np.zeros((MELS, WIN // 2 + 1), np.float32)
    for m in range(MELS):
        a, b, c = bins[m], bins[m + 1], bins[m + 2]
        for k in range(a, b):
            fb[m, k] = (k - a) / max(b - a, 1)
        for k in range(b, c):
            fb[m, k] = (c - k) / max(c - b, 1)
    return fb


FB = mel_filters()
HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)


def fit(x):
    """Centre the loudest part of the clip inside a fixed 0.8 s window."""
    x = x.astype(np.float32) / 32768
    if len(x) >= CLIP:
        e = np.convolve(x ** 2, np.ones(CLIP, np.float32), "valid")
        i = int(np.argmax(e))
        return x[i:i + CLIP]
    out = np.zeros(CLIP, np.float32)
    s = (CLIP - len(x)) // 2
    out[s:s + len(x)] = x
    return out


def logmel(x):
    x = x / max(float(np.abs(x).max()), 1e-3)
    fr = np.stack([x[i * HOP:i * HOP + WIN] * HANN for i in range(FRAMES)])
    p = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    return np.log(p @ FB.T + 1e-3).astype(np.float32)
