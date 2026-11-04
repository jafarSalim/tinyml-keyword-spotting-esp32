"""Trains the keyword model, tests it on unseen speakers, quantizes it to int8
and writes the C++ model plus a test set for the host check."""
import json, pathlib, sys, time, wave
import numpy as np, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import *

DATA = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "fsdd/recordings")
OUT = pathlib.Path("results")
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)


def load():
    X, y, spk, raw = [], [], [], []
    for f in sorted(DATA.glob("*.wav")):
        d, s, _ = f.stem.split("_")
        with wave.open(str(f)) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        c = fit(a)
        raw.append(c)
        X.append(logmel(c))
        y.append(int(d))
        spk.append(s)
    return np.stack(X), np.array(y), np.array(spk), np.stack(raw)


class Net(nn.Module):
    def __init__(self, w=(12, 16, 24)):
        super().__init__()
        ch = (1, *w)
        self.c = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, 2, 1) for i in range(3)])
        self.fc = nn.Linear(w[-1], 10)

    def forward(self, x):
        for c in self.c:
            x = torch.relu(c(x))
        return self.fc(x.mean((2, 3)))


def norm_stats(x):
    return float(x.mean()), float(x.std())


def fit_model(Xt, yt, epochs=40, seed=0, widths=(12, 16, 24), history=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(widths)
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=epochs * (len(Xt) // 64 + 1))
    X, y = torch.tensor(Xt)[:, None], torch.tensor(yt)
    for _ in range(epochs):
        p = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            b = p[i:i + 64]
            xb = X[b]
            shift = int(np.random.randint(-3, 4))
            xb = torch.roll(xb, shift, 2) + 0.1 * torch.randn_like(xb)
            loss = nn.functional.cross_entropy(net(xb), y[b], label_smoothing=.05)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if history is not None:
            history.append(loss.item())
    return net.eval()


@torch.no_grad()
def acc(net, X, y):
    return float((net(torch.tensor(X)[:, None]).argmax(1).numpy() == y).mean())


def quantize(net, Xcal, m, s):
    """Int8 weights, int32 biases and Q15 multipliers, all zero point 0."""
    q = dict(layers=[])
    x = torch.tensor((Xcal))[:, None]
    in_scale = float(x.abs().max()) / 127
    q["in_scale"] = in_scale
    cur = in_scale
    layers = list(net.c)
    with torch.no_grad():
        for c in layers:
            w = c.weight.numpy()
            sw = float(np.abs(w).max()) / 127
            x = torch.relu(c(x))
            so = float(x.max()) / 127
            q["layers"].append(dict(w=np.round(w / sw).astype(np.int8), b=np.round(c.bias.numpy() / (cur * sw)).astype(np.int32),
                                    mult=int(round(cur * sw / so * 32768))))
            cur = so
        w = net.fc.weight.numpy()
        sw = float(np.abs(w).max()) / 127
        logit = float(net.fc(x.mean((2, 3))).abs().max()) / 127
        q["fc"] = dict(w=np.round(w / sw).astype(np.int8), b=np.round(net.fc.bias.numpy() / (cur * sw)).astype(np.int32),
                       mult=int(round(cur * sw / logit * 32768)))
    return q


def conv_int(x, l):
    """x: (C,H,W) int32 in [0,127] or int8 input."""
    w, b, M = l["w"].astype(np.int64), l["b"].astype(np.int64), l["mult"]
    _, H, W = x.shape
    oh, ow = (H + 1) // 2, (W + 1) // 2
    pad = np.pad(x, ((0, 0), (1, 1), (1, 1))).astype(np.int64)
    out = np.zeros((w.shape[0], oh, ow), np.int64)
    win = np.lib.stride_tricks.sliding_window_view(pad, (3, 3), axis=(1, 2))[:, ::2, ::2]
    acc_ = np.einsum("fcij,chwij->fhw", w, win[:, :oh, :ow]) + b[:, None, None]
    return np.clip((acc_ * M + (1 << 14)) >> 15, 0, 127)


def predict_int(q, x):
    a = np.clip(np.round(x / q["in_scale"]), -127, 127).astype(np.int64)[None]
    for l in q["layers"]:
        a = conv_int(a, l)
    n = a.shape[1] * a.shape[2]
    pool = (a.sum((1, 2)) + n // 2) // n
    return int(np.argmax(q["fc"]["w"].astype(np.int64) @ pool + q["fc"]["b"]))


def carray(name, a, t):
    return f"static const {t} {name}[{a.size}] = {{{','.join(map(str, a.ravel().tolist()))}}};\n"


X, y, spk, raw = load()
print("clips", len(X), "frames x mels", X.shape[1:])
names = sorted(set(spk))
t0 = time.time()
loso = {}
conf = np.zeros((10, 10), int)
curves = {}
for sp in names:
    te = spk == sp
    m, s = norm_stats(X[~te])
    curve = []
    net = fit_model((X[~te] - m) / s, y[~te], history=curve)
    curves[sp] = curve
    pred = net(torch.tensor((X[te] - m) / s)[:, None]).argmax(1).numpy()
    loso[sp] = float((pred == y[te]).mean())
    np.add.at(conf, (y[te], pred), 1)
    print(sp, round(loso[sp], 4), int(time.time() - t0), "s", flush=True)

# Deployed model: one speaker kept out so the C++ check is on an unseen voice
hold = "theo"
te = spk == hold
m, s = norm_stats(X[~te])
Xtr, Xte = (X[~te] - m) / s, (X[te] - m) / s
net = fit_model(Xtr, y[~te])
fa = acc(net, Xte, y[te])
q = quantize(net, Xtr[::5], m, s)
ia = float(np.mean([predict_int(q, x) == t for x, t in zip(Xte, y[te])]))
params = sum(p.numel() for p in net.parameters())
print("float", fa, "int8", ia, "params", params)

with open("firmware/esp32_kws/model_data.h", "w") as f:
    f.write("// Generated by train.py. Int8 keyword model.\n#pragma once\n#include <stdint.h>\n")
    f.write(f"static const float KWS_MEAN = {m:.7f}f, KWS_STD = {s:.7f}f, KWS_IN_SCALE = {q['in_scale']:.9f}f;\n")
    for i, l in enumerate(q["layers"]):
        f.write(carray(f"W{i}", l["w"], "int8_t") + carray(f"B{i}", l["b"], "int32_t") + f"static const int32_t M{i} = {l['mult']};\n")
    f.write(carray("WF", q["fc"]["w"], "int8_t") + carray("BF", q["fc"]["b"], "int32_t"))
raw[te].astype(np.float32).tofile("tools/test_audio.bin")
y[te].astype(np.int32).tofile("tools/test_labels.bin")
flash = sum(l["w"].size + 4 * l["b"].size for l in q["layers"]) + q["fc"]["w"].size + 4 * q["fc"]["b"].size
json.dump(dict(clips=len(X), speakers=len(names), loso=loso, loso_mean=float(np.mean(list(loso.values()))),
               overall=float(np.trace(conf) / conf.sum()), held_out=hold, float_acc=fa, int8_python_acc=ia,
               params=int(params), weight_bytes=int(flash)), open(OUT / "train_summary.json", "w"), indent=2)
fig, a = plt.subplots(figsize=(5.6, 5), constrained_layout=True)
n = conf / conf.sum(1, keepdims=True)
a.imshow(n, cmap="Blues", vmin=0, vmax=1)
for i in range(10):
    for j in range(10):
        if conf[i, j]:
            a.text(j, i, f"{100 * n[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if n[i, j] > .5 else "black")
a.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="Spoken digit", title="Unseen speakers, leave-one-speaker-out (%)")
fig.savefig(OUT / "confusion.png", dpi=170)
plt.close(fig)
fig, a = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
k = list(loso)
a.bar(k, [100 * loso[i] for i in k], color="#1a73e8")
for i, v in enumerate(loso.values()):
    a.text(i, 100 * v + 1, f"{100 * v:.1f}%", ha="center")
a.set(ylabel="Accuracy (%)", ylim=(0, 108), title="Accuracy on a voice the model never heard")
a.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "per_speaker.png", dpi=170)
plt.close(fig)
fig, a = plt.subplots(figsize=(7, 2.8), constrained_layout=True)
a.imshow(X[0].T, origin="lower", aspect="auto", cmap="magma")
a.set(xlabel="Frame", ylabel="Mel band", title=f"What the model sees: spoken digit {y[0]}, {X.shape[1]} × {X.shape[2]} log-mel")
fig.savefig(OUT / "logmel.png", dpi=170)


# Model size against accuracy, leave one speaker out
WIDTHS = {"Tiny": (6, 8, 12), "Small": (8, 12, 16), "This model": (12, 16, 24), "Medium": (16, 32, 48), "Large": (32, 64, 96)}
size_rows = []
for name, w in WIDTHS.items():
    acc_ = []
    for sp in names:
        t_ = spk == sp
        m_, s_ = norm_stats(X[~t_])
        n_ = fit_model((X[~t_] - m_) / s_, y[~t_], widths=w)
        acc_.append(acc(n_, (X[t_] - m_) / s_, y[t_]))
    p_ = sum(i.numel() for i in n_.parameters())
    size_rows.append(dict(name=name, widths=w, params=int(p_), acc=float(np.mean(acc_)), std=float(np.std(acc_))))
    print(size_rows[-1], flush=True)

# Noise robustness of the deployed int8 model on the held-out voice
rng = np.random.default_rng(0)
SNRS = [30, 20, 15, 10, 5, 0]
noise_acc = []
clips, labels = raw[te], y[te]
for snr in SNRS:
    ok = 0
    for c, l in zip(clips, labels):
        p = np.mean(c ** 2)
        n = rng.standard_normal(len(c)).astype(np.float32) * np.sqrt(p / 10 ** (snr / 10))
        ok += predict_int(q, (logmel(np.clip(c + n, -1, 1)) - m) / s) == l
    noise_acc.append(ok / len(clips))
print("noise", dict(zip(SNRS, np.round(noise_acc, 3))))

recall = np.diag(conf) / conf.sum(1)
precision = np.diag(conf) / conf.sum(0)
extra = dict(widths=size_rows, noise=dict(zip(map(str, SNRS), map(float, noise_acc))), recall=recall.tolist(), precision=precision.tolist(),
             confusion=conf.tolist())
json.dump(extra, open(OUT / "experiments.json", "w"), indent=2)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

fig, a = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
a.errorbar([r["params"] for r in size_rows], [100 * r["acc"] for r in size_rows], yerr=[100 * r["std"] for r in size_rows],
           fmt="o-", c="#1a73e8", capsize=3, lw=2)
for r in size_rows:
    a.annotate(r["name"], (r["params"], 100 * r["acc"]), textcoords="offset points", xytext=(4, 8))
a.set_xscale("log")
a.set(xlabel="Parameters (log scale)", ylabel="Accuracy on unseen voices (%)", title="Accuracy against model size")
a.grid(alpha=.25)
fig.savefig(OUT / "size_vs_accuracy.png", dpi=170)
plt.close(fig)

fig, a = plt.subplots(figsize=(7.5, 3.9), constrained_layout=True)
a.plot(SNRS, 100 * np.array(noise_acc), "o-", c="#d01884", lw=2.4)
a.axhline(10, c="k", ls="--", label="Random guess")
a.invert_xaxis()
a.set(xlabel="Background white noise, SNR (dB)", ylabel="Int8 accuracy (%)", ylim=(0, 100), title=f"Noise robustness on held-out voice {hold}")
a.grid(alpha=.25)
a.legend(frameon=False)
fig.savefig(OUT / "noise_robustness.png", dpi=170)
plt.close(fig)

fig, a = plt.subplots(figsize=(7.5, 3.9), constrained_layout=True)
xx = np.arange(10)
a.bar(xx - .2, 100 * recall, .38, color="#1a73e8", label="Recall")
a.bar(xx + .2, 100 * precision, .38, color="#12b5cb", label="Precision")
a.set_xticks(xx)
a.set(xlabel="Digit", ylabel="%", ylim=(0, 105), title="Which digits are easy, on unseen voices")
a.legend(frameon=False, ncol=2)
fig.savefig(OUT / "per_digit.png", dpi=170)
plt.close(fig)

fig, a = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
for sp, c_ in curves.items():
    a.plot(c_, lw=1.6, label=sp)
a.set(xlabel="Epoch", ylabel="Last batch loss", title="Training, one curve per left-out speaker")
a.grid(alpha=.25)
a.legend(frameon=False, ncol=3)
fig.savefig(OUT / "training.png", dpi=170)
plt.close(fig)

fig, a = plt.subplots(figsize=(7.5, 3.4), constrained_layout=True)
labs = ["Weights and biases", "RAM buffers", "ESP32 flash (4 MB)", "ESP32 SRAM (520 KB)"]
vals = [flash / 1024, 10068 / 1024, 4096, 520]
a.barh(labs, vals, color=["#1a73e8", "#12b5cb", "#c4c7c5", "#c4c7c5"])
a.set_xscale("log")
a.invert_yaxis()
for i, v in enumerate(vals):
    a.text(v * 1.1, i, f"{v:,.1f} KB", va="center")
a.set(xlabel="KB (log scale)", title="The model against a typical ESP32", xlim=(1, 40000))
fig.savefig(OUT / "memory_footprint.png", dpi=170)
plt.close(fig)
