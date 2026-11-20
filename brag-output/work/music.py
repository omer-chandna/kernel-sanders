# 96 BPM, C major, 36 beats = 22.5s. Music and effects on one grid, mixed together.
import numpy as np, wave

SR = 44100; BPM = 96; BEAT = 60 / BPM; DUR = 22.5
N = int(SR * DUR) + SR * 2  # tail room, trimmed later
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(7)

def hz(m): return 440 * 2 ** ((m - 69) / 12)
def at(b): return int(b * BEAT * SR)

def add(sig, start, gain=1.0, pan=0.0):
    s = max(0, start); e = min(N, start + len(sig))
    if e <= s: return
    seg = sig[s - start:e - start] * gain
    L[s:e] += seg * np.sqrt(0.5 * (1 - pan)); R[s:e] += seg * np.sqrt(0.5 * (1 + pan))

def env(n, a=0.005, d=0.3):
    t = np.arange(n) / SR
    return np.minimum(1, t / a) * np.exp(-t / d)

def lowpass(x, fc):
    a = np.exp(-2 * np.pi * fc / SR); y = np.zeros_like(x); z = 0.0
    for i in range(len(x)): z = (1 - a) * x[i] + a * z; y[i] = z
    return y

# chords per bar (4 beats): C Am F G | Am F C G | C
CH = [[48, 52, 55, 60, 64], [45, 52, 57, 60, 64], [41, 53, 57, 60, 65], [43, 55, 59, 62, 67],
      [45, 52, 57, 60, 64], [41, 53, 57, 60, 65], [48, 52, 55, 60, 64], [43, 55, 59, 62, 67], [48, 55, 60, 64, 67]]
ROOT = [36, 33, 29, 31, 33, 29, 36, 31, 36]

# pad: warm detuned triangles, low-passed, per bar with soft swells
for bar, ch in enumerate(CH):
    n = int(4 * BEAT * SR) + (int(2.5 * SR) if bar == 8 else int(0.3 * SR))
    t = np.arange(n) / SR; sig = np.zeros(n)
    for m in ch[1:]:
        for det in (-0.06, 0.06):
            f = hz(m + det); sig += 2 / np.pi * np.arcsin(np.sin(2 * np.pi * f * t))
    a = np.minimum(1, t / 0.25); rel = np.clip((n / SR - t) / 0.3, 0, 1)
    sig = lowpass(sig * a * rel, 1400 if bar < 8 else 1000)
    g = 0.035 if bar == 0 else 0.05
    add(sig, at(bar * 4), g, 0.0)

# bass: round sine + a little 2nd harmonic on 1, the "and" of 2, and 3
for bar in range(9):
    for b in ([0, 1.5, 2, 3.5] if bar < 8 else [0]):
        if bar == 0 and b > 0: continue
        n = int((0.5 if bar < 8 else 2.5) * SR); t = np.arange(n) / SR
        f = hz(ROOT[bar]); s = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
        add(s * env(n, 0.008, 0.35 if bar < 8 else 1.0), at(bar * 4 + b), 0.22)

# kick: soft sine drop on every beat (lighter in the hook), skip the outro tail
for b in range(0, 33):
    n = int(0.35 * SR); t = np.arange(n) / SR
    f = 45 + 70 * np.exp(-t / 0.035); ph = 2 * np.pi * np.cumsum(f) / SR
    k = np.sin(ph) * np.exp(-t / 0.16)
    add(k, at(b), 0.32 if b >= 4 else 0.2)

# hats on the off-8ths, soft clap on 2 & 4 from the order scene
hat = rng.standard_normal(int(0.05 * SR)); hat = hat - lowpass(hat, 6000); hat *= env(len(hat), 0.001, 0.015)
for b in range(4, 32):
    add(hat, at(b + 0.5), 0.05, 0.3)
clap = rng.standard_normal(int(0.2 * SR)); clap = lowpass(clap - lowpass(clap, 900), 5000) * env(len(clap), 0.002, 0.06)
for b in range(10, 30):
    if b % 2 == 1: add(clap, at(b), 0.1, -0.15)

# marimba-ish pluck lead from beat 10, pentatonic motif per bar
def pluck(m, dur=0.5):
    n = int(dur * SR); t = np.arange(n) / SR; f = hz(m)
    return (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.28) + 0.35 * np.sin(2 * np.pi * 4 * f * t) * np.exp(-t / 0.05))
MEL = {10: 72, 10.5: 76, 11: 79, 11.75: 76,  # over Am-ish (bar 2 = F) — chord tones
       12: 72, 12.5: 77, 13: 81, 14: 79, 14.5: 74, 15.5: 79,
       16: 76, 16.5: 72, 17: 69, 17.5: 72,
       20: 72, 20.5: 77, 21: 81, 21.5: 77, 22: 72, 22.75: 74,
       24.5: 79, 26: 76, 26.5: 74, 27: 72,
       28: 74, 28.5: 79, 29: 83, 29.5: 79}
for b, m in MEL.items(): add(pluck(m), at(b), 0.08, 0.2)

# ---- effects, in key, on the grid ----
def tick(m, g):
    n = int(0.08 * SR); t = np.arange(n) / SR
    return np.sin(2 * np.pi * hz(m) * t) * np.exp(-t / 0.018) * g
for b in [5, 6, 7]: add(tick(84, 1), at(b), 0.09, -0.3)              # draft tokens (C6)
for b in [9.5, 10, 11]: add(tick(88, 1), at(b), 0.09, 0.2)           # accepted (E6)
add(tick(72, 1), at(10.5), 0.10, 0.2)                                 # rejected (C5, lower)
for b in [14, 15.5, 17]: add(tick(84, 1), at(b), 0.09, 0.0)          # parity rows
for b in range(20, 28): add(tick(84 + (b % 2) * 4, 1), at(b), 0.07, 0.0)   # speculative rows

def chime(notes, start_beat, g):
    for i, f in enumerate(notes):
        n = int(0.9 * SR); t = np.arange(n) / SR
        s = (np.sin(2 * np.pi * f * t) + 0.15 * np.sin(4 * np.pi * f * t)) * np.minimum(1, t / 0.01) * np.exp(-t / 0.35)
        add(s, at(start_beat + i * 0.25), g, 0.1)
chime([659.25, 880.0], 8, 0.08)                         # verify pass
chime([523.25, 659.25, 783.99, 1046.5], 28.5, 0.12)    # +18% to +44% lands

# noise swells into each cut
for b in [4, 13, 19, 28, 31]:
    n = int(0.5 * SR); t = np.arange(n) / SR
    w = lowpass(rng.standard_normal(n), 2500) * (t / t[-1]) ** 2
    add(w, at(b) - n, 0.12, 0.0)

# soft impact on the hook and the outro: low C + sub
for b in [0, 31]:
    n = int(1.6 * SR); t = np.arange(n) / SR
    s = np.sin(2 * np.pi * hz(24) * t) * np.exp(-t / 0.6) + 0.4 * np.sin(2 * np.pi * hz(48) * t) * np.exp(-t / 0.4)
    add(s, at(b), 0.25)

# mix: trim to length, gentle fade at the very end, soft clip, normalize
n = int(SR * DUR)
mix = np.stack([L[:n], R[:n]], 1)
fade = np.ones(n); fl = int(0.8 * SR); fade[-fl:] = np.linspace(1, 0, fl) ** 1.5
mix *= fade[:, None]
mix = np.tanh(mix * 1.4) / 1.4
mix /= np.max(np.abs(mix)) / 0.89
with wave.open("music.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
print("ok", n / SR)
