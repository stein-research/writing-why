"""Figure: the three runs on ibm_kingston, exact against device, as Reading Why's fig_device.py - the kite prompt,
the same with "kite" deleted, and the park prompt. Each panel: the words most probable for the exact model or the
device (the union of both top-3s), their probabilities, and 95 % shot-noise intervals on the device's.

Reads results/hw_word_bootstrap_<prompt>[_del4]_answer.json (hw_word.py --bootstrap 200 --no-evidence).

python fig_device.py [--out figures/device_kingston]
"""
import glob
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
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "figures/device_kingston"
INK, MUTED, LINE, ACCENT = "#2b2f36", "#495057", "#c9ced6", "#d9480f"

RUNS = [("Sue_had_a_red_kite_She_flew_her", "Sue had a red kite . She flew her", "the word after the prompt: kite", INK),
        ("Sue_had_a_red_kite_She_flew_her_del4", "Sue had a red _ . She flew her",
         "“kite” deleted: the word that decides it", ACCENT),
        ("Lily_went_to_the_park_At_the", "Lily went to the park . At the", "the word after the prompt: park", INK)]

fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.4), sharey=True)   # printed at the text width: titles ~7 pt
for ax, (slug, title, foot, fc) in zip(axs, RUNS):
    b = json.load(open(f"results/hw_word_bootstrap_{slug}_answer.json"))
    W = list(b["words"])[:4]
    ex = np.array([b["words"][w]["exact"] for w in W])
    dv = np.array([b["words"][w]["device"] for w in W])
    ci = np.array([b["words"][w]["device_95"] for w in W])
    x = np.arange(len(W))
    ax.bar(x - 0.19, ex, 0.36, color=LINE, label="exact model")
    ax.bar(x + 0.19, dv, 0.36, color=ACCENT, label="ibm_kingston, 95 % shot noise",
           yerr=np.vstack([dv - ci[:, 0], ci[:, 1] - dv]).clip(min=0), ecolor=INK, error_kw=dict(lw=1.0, capsize=3))
    for k in range(len(W)):
        fsz = 9.5 if len(W) > 3 else 11.5                    # four words in a panel: smaller, still centred
        ax.text(x[k] - 0.19, ex[k] + 0.015, f"{ex[k]:.3f}", ha="center", va="bottom", fontsize=fsz, color=MUTED)
        ax.text(x[k] + 0.19, ci[k, 1] + 0.015, f"{dv[k]:.3f}", ha="center", va="bottom", fontsize=fsz, color=ACCENT)
    ax.set_xticks(x, W, fontsize=13.5, color=INK)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color=LINE, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    [ax.spines[sp].set_color(LINE) for sp in ("left", "bottom")]
    ax.tick_params(colors=MUTED, length=0, labelsize=12.5)
    ax.set_title(title, fontsize=14.5, style="italic", color=INK, pad=10)
    ax.text(0.5, -0.17, foot, transform=ax.transAxes, ha="center", va="top", fontsize=13.5, color=fc)
axs[0].set_ylabel("Born probability $P(w)$", fontsize=14, color=INK)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=13.5, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}.png / .pdf")
