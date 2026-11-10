"""Figure: the readout ladder - accuracy and perplexity against the number of passes n of the readout
P(w) ∝ |1 + g psi_w / n|^(2n) (one mask passed n times), same body and recipe at every n (TinyStories V2 words,
1M windows, lr 3.2e-2, warm-up 0.02), scored on fresh_test.py's 2,000 fresh test windows. n = 16: mean of three seeds (bars: their range). The one-copy readout without
the reference beam (|psi|^2) is drawn at n = 1 for comparison: the reference beam alone changes nothing. The softmax
transformer of the same size is a separately trained model (three seeds), drawn as a reference line.

python fig_ladder.py [--out figures/ladder]
"""
import glob
import os
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

for f in glob.glob("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-*.otf"):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Latin Modern Sans", "DejaVu Sans"], "mathtext.fontset": "custom",
                     "mathtext.rm": "Latin Modern Sans", "mathtext.it": "Latin Modern Sans:italic",
                     "mathtext.bf": "Latin Modern Sans:bold", "pdf.fonttype": 3})
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "figures/ladder"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
INK, MUTED, LINE, ACCENT = "#2b2f36", "#495057", "#c9ced6", "#d9480f"

FRESH = json.load(open("results/fresh_test.json"))                    # fresh_test.py: test windows nothing was chosen on
get = lambda t: dict(test_acc=FRESH[t]["acc"], test_ppl=FRESH[t]["ppl"])
ns = [1, 2, 4, 8, 16]
runs = {n: [get(f"words_born-therm{n}p2_s0")] for n in ns}
runs[16] = [get(f"words_born-therm16p2_s{s}") for s in range(3)]
plain = get("words_born-p2r16_s0")
soft = [get(f"words_softmax-w96_s{s}") for s in range(3)]

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
for ax, key, lab, scale in ((axs[0], "test_acc", "test accuracy (%)", 100), (axs[1], "test_ppl", "perplexity", 1)):
    m = [np.mean([r[key] for r in runs[n]]) * scale for n in ns]
    lo = [m[i] - min(r[key] for r in runs[n]) * scale for i, n in enumerate(ns)]
    hi = [max(r[key] for r in runs[n]) * scale - m[i] for i, n in enumerate(ns)]
    ax.plot(ns, m, color=ACCENT, lw=2, zorder=3)
    ax.errorbar(ns, m, yerr=[lo, hi], fmt="o", ms=6, color=ACCENT, mec="white", mew=1.5, capsize=3, zorder=4)
    ax.plot([0.82], [plain[key] * scale], "o", ms=6, mfc="white", mec=ACCENT, mew=1.5, zorder=5)
    sm = np.mean([r[key] for r in soft]) * scale
    ax.axhline(sm, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.text(16, sm, "softmax transformer, same size", ha="right", va="bottom" if key == "test_acc" else "top",
            fontsize=7.5, color=MUTED)
    for n, v in zip(ns, m):
        acc = key == "test_acc"
        ax.annotate(f"{v:.1f}" if acc else f"{v:.2f}", (n, v), textcoords="offset points",
                    xytext=(0, -13) if acc else (6, 5), ha="center" if acc else "left", fontsize=7.5, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("passes through the mask, $n$", fontsize=8.5, color=INK)
    ax.set_ylabel(lab, fontsize=8.5, color=INK)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.grid(axis="y", color=LINE, lw=0.5, alpha=0.6)
axs[0].set_ylim(40.5, 52)
axs[1].set_ylim(7.5, 19)
axs[0].annotate(r"$|\psi|^2$", (0.82, plain["test_acc"] * 100), textcoords="offset points",
                xytext=(-4, 7), ha="center", fontsize=7, color=MUTED)
fig.tight_layout(w_pad=2.5)
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
print(f"wrote {out}.png / .pdf")
