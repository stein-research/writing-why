"""Figure 1: one written word through the 16-pass model, drawn from the trained model's own data, as Reading Why's
overview: (a) on light, stage by stage, to the next word and each earlier word's exact evidence; (b) on qubits, the
same attention read by Hadamard tests on ibm_kingston, the next word, and each word's evidence with the device's row
sums.

Prompt "Lily went to the park . At the" (next word: park), the first device run of hw_word.py. Reads results/<tag>.pt
(the model), block 1's device job in the ledger (ibm_jobs/934582274897890b: measured against ideal overlaps),
results/hw_word_Lily_went_to_the_park_At_the_ibm_kingston_2000.json and, when present,
results/hw_word_bootstrap_Lily_went_to_the_park_At_the.json (shot-noise intervals).

python fig_overview.py [--out figures/overview]
"""
import glob
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle

import tinywords
from models import build

for f in glob.glob("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-*.otf"):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Latin Modern Sans", "DejaVu Sans"], "mathtext.fontset": "custom",
                     "mathtext.rm": "Latin Modern Sans", "mathtext.it": "Latin Modern Sans:italic",
                     "mathtext.bf": "Latin Modern Sans:bold", "pdf.fonttype": 3})
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "figures/overview"

INK, MUTED, LINE, ACCENT = "#2b2f36", "#495057", "#c9ced6", "#d9480f"
ev_cmap = LinearSegmentedColormap.from_list("evidence", ["#08306b", "#1f78d1", "#9cc9f5", "#f7f7f7", "#f9a77e",
                                                         "#e8431c", "#7f0d0d"])     # Seeing Why's: blue against, red for
att_cmap, cam_cmap = plt.get_cmap("magma"), plt.get_cmap("inferno")

tag, prompt, N = "words_born-therm16p2_s0", "Lily went to the park . At the", 16
slug = "Lily_went_to_the_park_At_the"
JOB0 = "ibm_jobs/934582274897890b"
run = json.load(open(f"results/{tag}.json"))
words, index = tinywords.vocab()
word = lambda t: words[int(t) - tinywords.C0]
model = build(run["model"], 8192, 128)
model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
model = model.double().eval()
toks = tinywords.tokens(prompt)
ids = torch.tensor([[index[t] for t in toks]])
L = len(toks)
amp = model.amplitudes(ids)
with torch.no_grad():
    scores = []
    u, _ = model.hidden(ids, amp, scores=scores)                       # block 2's 64 measured readings, [1, L, 64]
    inten, _ = model.intensities(ids, amp, at=torch.tensor([L - 1]))    # the camera at the last position, [1, 1, V]
    p = model.probs(ids)[0, -1].numpy()
    e = model._c(model.emb)[ids] + model._c(model.pos)[:L]
    ang = torch.arange(L, dtype=model.omega.dtype)[:, None] * model.omega[None, :]
    h0 = (e * torch.polar(torch.ones_like(ang), ang) * amp.unsqueeze(-1).to(e.dtype))[0]   # the fields, [L, 32]
    q0 = (h0 @ model._c(model.qkvo[0])[0].T).unflatten(-1, (model.heads, -1))             # block 1's queries
dev = json.load(open(f"results/hw_word_{slug}_ibm_kingston_2000.json"))
try:
    boot = json.load(open(f"results/hw_word_bootstrap_{slug}.json"))
except FileNotFoundError:
    boot = None
ev = np.array(dev["exact"]["evidence"])                               # BEW of the words before the last one


def device_overlaps(job):
    """Ideal and measured P0 - P1 of every overlap circuit of a ledger job (calibrations left out)."""
    from qiskit import qpy
    from qiskit.quantum_info import Statevector
    rec, counts = json.load(open(f"{job}/record.json")), json.load(open(f"{job}/counts.json"))
    circs = qpy.load(open(f"{job}/circuits.qpy", "rb"))
    ideal, meas = [], []
    for c, cnt, info in zip(circs, counts, rec["circuits"]):
        if "calibration" in info["note"]:
            continue
        pr = Statevector(c.remove_final_measurements(inplace=False)).probabilities([4])
        ideal.append(pr[0] - pr[1])
        meas.append((cnt.get("0", 0) - cnt.get("1", 0)) / sum(cnt.values()))
    return np.array(ideal), np.array(meas)


dev_ideal, dev_meas = device_overlaps(JOB0)

fig = plt.figure(figsize=(15.5, 10.2))
W, Hf = fig.get_size_inches()
HD, SUB, LAB = 17, 16.5, 15.5                                       # printed at the text width: captions ~6.5 pt


def axes(x, y, w, h, off=True):
    ax = fig.add_axes([x, y, w, h])
    if off:
        ax.axis("off")
    return ax


def head(ax, title, sub, above=0.08, below=-0.1):
    ax.text(0.5, 1 + above, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=HD,
            fontweight="bold", color=INK, linespacing=1.15)
    if sub:
        ax.text(0.5, below, sub, transform=ax.transAxes, ha="center", va="top", fontsize=SUB, color=MUTED,
                linespacing=1.25)


def arrow(x0, x1, y):
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, color=INK, lw=2.2,
                                   arrowstyle="-|>,head_length=6,head_width=3.5"))


def frame(ax):
    ax.set_xticks([]), ax.set_yticks([])
    [s.set_color(LINE) for s in ax.spines.values()]


def bars(ax, labels, probs, ref=None):
    ax.barh(range(3), probs, color=[ACCENT, LINE, LINE], height=0.6)
    ax.set_yticks(range(3), labels, fontsize=LAB, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.3), ax.set_xticks([])
    for c in range(3):
        ax.text(probs[c] + 0.04, c, f"{probs[c]:.2f}" if probs[c] >= 0.01 else f"{probs[c]:.3f}", va="center",
                fontsize=LAB, color=INK)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.tick_params(length=0)


def chips(ax, vals, lim, y_top=0.7, per=0.05, pad=0.05, last=None):
    xx, yy = 0.0, y_top
    for j, w in enumerate(toks):
        width = per * len(w) + pad
        if xx + width > 1.02:
            xx, yy = 0.0, yy - 0.25
        if j < len(vals):
            fc = ev_cmap(TwoSlopeNorm(0, -lim, lim)(vals[j]))
            ec, dark = "none", abs(vals[j]) > 0.55 * lim
        else:                                                         # the current word asks; it has no share
            fc, ec, dark = "#ffffff", MUTED, False
        ax.add_patch(FancyBboxPatch((xx, yy - 0.07), width - 0.025, 0.14, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=fc, ec=ec, lw=0.9, ls=(0, (2, 1.5)) if j >= len(vals) else "-",
                                    transform=ax.transAxes))
        ax.text(xx + (width - 0.025) / 2, yy, w, ha="center", va="center", fontsize=LAB, transform=ax.transAxes,
                color="white" if dark else INK)
        xx += width


# ---------------------------------------------------------------- (a) light
fig.text(0.008, 0.955, "a", fontsize=22, fontweight="bold", color=INK)
fig.text(0.033, 0.955, "On light (simulated): every nonlinearity on the light a quadratic detection", fontsize=20,
         fontweight="bold", color=INK)
SW, SH = 0.072, 2.1 / Hf
ys = 0.632
ym = ys + SH / 2
ax = axes(0.008, ys, 0.09, SH)
head(ax, "prompt", None, above=0.1)
for k, line in enumerate(("Lily went to", "the park .", "At the ...")):
    ax.text(0.0, 0.68 - 0.13 * k, line, fontsize=LAB + 1, color=INK, style="italic", transform=ax.transAxes)
arrow(0.093, 0.107, ym)
ax = axes(0.11, ys, SW, SH, off=False)
h = h0.numpy()
a = np.abs(h)[..., None]
ax.imshow(plt.cm.twilight((np.angle(h) / (2 * np.pi)) % 1)[..., :3] * np.clip(np.sqrt(a / np.percentile(a, 99)), 0, 1),
          aspect="auto", interpolation="nearest")
frame(ax)
head(ax, "modulator", f"{L} words $\\times$\n32 modes")
arrow(0.186, 0.2, ym)
fig.add_artist(FancyBboxPatch((0.203, ys - 0.09), 0.236, SH + 0.187, boxstyle="round,pad=0,rounding_size=0.012",
                              transform=fig.transFigure, fc="#f2f2f2", ec=INK, lw=1.2, zorder=0))
fig.text(0.439, ys - 0.117, "$\\times$2 blocks", ha="right", fontsize=SUB, color=MUTED, style="italic")
ax = axes(0.248, ys, SW, SH, off=False)
w = scores[1][0].mean(0).numpy()                                     # block 2, mean of heads, [L, L]
wfull = np.concatenate([w, (1 - w.sum(-1))[:, None]], 1)             # the reference key takes the rest of each row
future = np.triu(np.ones((L, L), bool), 1)                       # j > i: a later word, never read (causal)
masked = np.ma.masked_array(wfull, np.concatenate([future, np.zeros((L, 1), bool)], 1))
cm = att_cmap.copy()
cm.set_bad("#cfd4da")                                              # light grey: masked, not a small weight
ax.imshow(masked, cmap=cm, aspect="auto", interpolation="nearest")
ax.set_xticks([]), ax.set_yticks(range(L), toks, fontsize=15, color=MUTED)
ax.tick_params(length=0, pad=2)
[s.set_color(LINE) for s in ax.spines.values()]
ax.text(L, -0.75, "ref", ha="center", va="bottom", fontsize=14.5, color=MUTED)
head(ax, "Born\nattention", "$|\\langle q_i|k_j\\rangle|^2$\n/ row sum", above=0.1)
arrow(0.325, 0.34, ym)
ax = axes(0.355, ys, SW, SH, off=False)
m = u[0].numpy()
m = np.sign(m) * np.sqrt(np.abs(m))
lim = np.percentile(np.abs(m), 99)
ax.imshow(m, cmap=ev_cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", interpolation="nearest")
frame(ax)
head(ax, "measured\nfeed-forward", "balanced\nreading", above=0.1)
arrow(0.443, 0.457, ym)
ax = axes(0.46, ys, 0.1, SH, off=False)
I = inten[0, 0].numpy()
t = (I / I.max()) ** (1 / (2 * N))                                   # the mask's transmission |1 + g psi/16| / M
ax.imshow(t.reshape(64, 128), cmap="Purples", interpolation="nearest", aspect="auto")
frame(ax)
for k, dy in enumerate((0.25, 0.5, 0.75)):
    ax.annotate("", xy=(1.06, dy), xytext=(-0.06, dy), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.0, alpha=0.8))
head(ax, "one mask,\n16 passes", "transmission\n$|1 + g\\,\\psi_w/16|/M$", above=0.1)
arrow(0.565, 0.58, ym)
ax = axes(0.583, ys, 0.1, SH, off=False)
ax.imshow(np.log10(np.maximum(I / I.max(), 1e-12)).reshape(64, 128), cmap=cam_cmap, vmin=-12, vmax=0, interpolation="nearest", aspect="auto")   # log: one word holds 95 % of the light
frame(ax)
kp = int(np.argmax(I))                                                # the brightest pixel: "park"
ax.plot(kp % 128, kp // 128, "s", ms=6, mfc="none", mec="white", mew=1.4)
assert kp == int(np.argmax(p)), "a pixel is a word id"
ax.annotate(word(kp), (kp % 128, kp // 128), xytext=(kp % 128 - 22, 12), textcoords="data", ha="right", va="center",
            color="white", fontsize=LAB, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2, shrinkA=1, shrinkB=3))
head(ax, "camera", "8,192 pixels, one per\nword (log scale)", above=0.1)
arrow(0.688, 0.7, ym)
top3 = np.argsort(-p)[:3]
ax = axes(0.745, ys + 0.02, 0.045, SH - 0.04, off=False)
bars(ax, [word(i) for i in top3], p[top3])
head(ax, "next word", f"{word(top3[0])} {p[top3[0]]:.2f}", above=0.12, below=-0.12)
ax.texts[-1].set_color(ACCENT), ax.texts[-1].set_fontweight("bold")
arrow(0.83, 0.845, ym)
ax = axes(0.848, ys, 0.147, SH)
chips(ax, ev, np.abs(ev).max(), y_top=0.68, per=0.052, pad=0.05)
head(ax, "exact evidence", "sums to $P - 1/V - F_q$\nexactly; “the” asks", above=0.1)

# ---------------------------------------------------------------- (b) qubits
fig.add_artist(plt.Line2D([0.02, 0.985], [0.455, 0.455], transform=fig.transFigure, color=LINE, lw=0.9))
fig.text(0.008, 0.405, "b", fontsize=22, fontweight="bold", color=INK)
fig.text(0.033, 0.405, "On a quantum computer: the same weights, ibm_kingston", fontsize=20, fontweight="bold",
         color=INK)
yb, hb = 0.1195, 0.1845
qa = q0[L - 1, 0].numpy()
qa = qa / np.linalg.norm(qa)
ax = axes(0.02, yb + 0.028, 0.1, 0.052, off=False)
ax.bar(range(16), np.abs(qa) ** 2, color=plt.cm.twilight((np.angle(qa) / (2 * np.pi)) % 1), width=0.85)
ax.set_xlim(-0.8, 15.8), ax.set_ylim(0, (np.abs(qa) ** 2).max() * 1.15)
ax.set_xticks([0, 15], [r"$|0000\rangle$", r"$|1111\rangle$"], fontsize=LAB, color=INK)
ax.set_yticks([])
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(LINE)
ax.tick_params(length=0, pad=3)
ax.text(0.5, 1.9, f"“{toks[-1]}”", transform=ax.transAxes, ha="center", va="center", fontsize=SUB,
        color=INK, bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.3", fc="#f9dcdf", ec=INK, lw=1.0))
ax.annotate("", xy=(0.5, 1.08), xytext=(0.5, 1.6), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2", color=INK, lw=1.0))
head(ax, "a word as amplitudes", "its query: 16 complex\nnumbers, the amplitudes\nof 4 qubits (height $|a|^2$,\ncolour: phase)",
     above=1.45, below=-0.62)
arrow(0.132, 0.15, yb + hb / 2)
ax = axes(0.168, yb, 0.17, hb)
ax.set_xlim(0, 10), ax.set_ylim(0, 6)
for yw, x1 in ((4.6, 8.0), (2.2, 9.3)):
    ax.plot([0.0, x1], [yw, yw], color=INK, lw=1.3)
ax.text(-0.2, 4.6, r"$|0\rangle$", ha="right", va="center", fontsize=SUB, color=INK)
ax.text(-0.2, 2.2, "4", ha="right", va="center", fontsize=SUB, color=MUTED)
ax.plot([0.3, 0.6], [2.0, 2.4], color=INK, lw=1.0)
ax.add_patch(FancyBboxPatch((0.75, 1.3), 5.65, 4.1, boxstyle="round,pad=0,rounding_size=0.25", fc="#f9dcdf",
                            ec=INK, lw=1.1, zorder=3))
ax.text(3.58, 3.35, "prepare\n" + r"$|0\rangle|q_i\rangle+|1\rangle|k_j\rangle$", ha="center", va="center",
        fontsize=LAB - 1.5, color=INK, linespacing=1.4, zorder=4)
ax.add_patch(Rectangle((6.6, 4.15), 0.9, 0.9, fc="white", ec=INK, lw=1.0, zorder=3))
ax.text(7.05, 4.6, "H", ha="center", va="center", fontsize=LAB, fontweight="bold", color=INK, zorder=4)
ax.add_patch(Rectangle((8.0, 4.05), 1.3, 1.1, fc="white", ec=ACCENT, lw=1.2, zorder=3))
ax.add_patch(Arc((8.65, 4.3), 0.9, 0.9, theta1=0, theta2=180, color=ACCENT, lw=1.2, zorder=4))
ax.plot([8.65, 8.9], [4.3, 4.75], color=ACCENT, lw=1.1, zorder=4)
ax.text(5, 0.25, r"$P_0 - P_1 = \mathrm{Re}\,\langle q_i|k_j\rangle$", ha="center", fontsize=SUB, color=INK)
nc = dev["circuits_sent"] if "circuits_sent" in dev else None
head(ax, "Hadamard test", "two per overlap (Re, Im),\nevery row of block 1, the\nlast of block 2, and 32 as\ndeep per block to calibrate",
     above=0.03, below=-0.02)
arrow(0.342, 0.358, yb + hb / 2)
ax = axes(0.4, yb + 0.014, 0.1, hb - 0.028, off=False)
cal = dev["run"]["calibration"][0]
ax.plot([-1, 1], [-1, 1], color=LINE, lw=1.0, zorder=1)
ax.scatter(dev_ideal, dev_meas, s=6, color=ACCENT, alpha=0.7, lw=0, zorder=2)
ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_aspect("equal")
ax.set_xticks([-1, 0, 1]), ax.set_yticks([-1, 0, 1])
ax.tick_params(labelsize=LAB, colors=MUTED, length=2)
[ax.spines[s].set_color(LINE) for s in ax.spines]
ax.set_xlabel("ideal", fontsize=SUB, color=MUTED, labelpad=1)
ax.set_ylabel("measured", fontsize=SUB, color=MUTED, labelpad=1)
r = np.corrcoef(dev_ideal, dev_meas)[0, 1]
head(ax, "on the chip", f"block 1, {len(dev_ideal)} readings;\nr = {r:.3f} against ideal,\nvisibility {cal['slope']:.2f}",
     above=0.06, below=-0.34)
arrow(0.515, 0.535, yb + hb / 2)
ax = axes(0.6, yb + 0.028, 0.05, hb - 0.056, off=False)
dtop = dev["run"]["top"]
bars(ax, [w for w, _ in dtop], np.array([v for _, v in dtop]))
ptxt = f"device {dtop[0][1]:.3f}"
try:                                                                  # the probability's interval: 200 redraws
    pb = json.load(open(f"results/hw_word_bootstrap_{slug}_answer.json"))["p_word_95"]
    ptxt += f" [{pb[0]:.3f}, {pb[1]:.3f}]"
except FileNotFoundError:
    pass
head(ax, "the next word", f"{ptxt},\nexact {dev['exact']['top'][0][1]:.3f}; one of 8,192", above=0.12, below=-0.15)
arrow(0.695, 0.715, yb + hb / 2)
ax = axes(0.74, yb + 0.028, 0.245, hb - 0.056, off=False)
dn = json.load(open(f"results/hw_word_devnum_{slug}.json"))["device_numerators"]   # block 1's attention from the device
dv = np.array(dn["evidence"])
xw = np.arange(len(ev))
ax.bar(xw - 0.2, ev, 0.38, color=LINE, label="exact")
err = None
if "evidence_95" in dn:
    ci = np.array(dn["evidence_95"])
    err = np.vstack([dv - ci[:, 0], ci[:, 1] - dv]).clip(min=0)
ax.bar(xw + 0.2, dv, 0.38, color=ACCENT, label="block 1 from ibm_kingston", yerr=err, ecolor=INK,
       error_kw=dict(lw=0.9, capsize=1.6))
ax.set_xticks(xw, toks[:len(ev)], fontsize=LAB, color=INK, rotation=35, ha="right", rotation_mode="anchor")
ax.set_yticks([0, 0.1, 0.2]), ax.tick_params(labelsize=LAB, colors=MUTED, length=2)
ax.set_xlim(-0.6, len(ev) - 0.4)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
[ax.spines[sp].set_color(LINE) for sp in ("left", "bottom")]
ax.set_ylim(0, 0.37)
ax.legend(frameon=False, fontsize=LAB, loc="upper left", bbox_to_anchor=(0.02, 1.03), ncol=2, handlelength=0.9,
          handletextpad=0.5, columnspacing=0.9, borderaxespad=0.05)
inside = int(((ev >= ci[:, 0]) & (ev <= ci[:, 1])).sum()) if err is not None else None
head(ax, "evidence from the chip's readings", f"exact share within the 95 % shot-noise\n"
     f"interval for {inside} of {len(ev)} words",
     above=0.12, below=-0.42)
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}.png / .pdf")
