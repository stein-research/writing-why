"""Figure: Writing Why's 16-pass model on qubits and light, its stages left to right, and under each stage what the
one-word runs on ibm_kingston (hw_word.py) did with it - the attention overlaps read on the device, the feed-forward
and the 16-pass readout simulated as light, the rest classical. After Reading Why's fig_born_qubits.py, box colours
as in Fig. 1 (fig_born_pair.py).

python fig_born_qubits.py [--out figures/born_qubits]
"""
import glob
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch

for f in glob.glob("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-*.otf"):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Latin Modern Sans", "DejaVu Sans"], "mathtext.fontset": "custom",
                     "mathtext.rm": "Latin Modern Sans", "mathtext.it": "Latin Modern Sans:italic",
                     "mathtext.bf": "Latin Modern Sans:bold", "pdf.fonttype": 3})
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "figures/born_qubits"
SUB = ("the same weights; the word after “Lily went to the park . At the” and “Sue had a red kite . She flew "
       "her”: every attention overlap read on the device, the evidence with exact numerators and the measured row sums")

INK, MUTED, ACCENT, LIGHT = "#2b2f36", "#495057", "#d9480f", "#b08900"
F = dict(emb="#f9dcdf", att="#fbe3c4", norm="#f3f5c6", ff="#cfe8f4", lin="#dcdcf2", out="#d6ecd2", blk="#f2f2f2")

fig = plt.figure(figsize=(17.5, 4.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.5, 35), ax.set_ylim(0.6, 8.9)
ax.axis("off")
Y, BW, BH = 5.0, 4.3, 1.6


def arrow(x0, x1, y=Y, color=INK):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>,head_length=4.5,head_width=2.6", color=color, lw=1.1,
                                 shrinkA=0, shrinkB=0, zorder=4))


def meter(cx, cy, r=0.2):
    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=0, theta2=180, color=ACCENT, lw=1.2, zorder=7))
    ax.plot([cx - r, cx + r], [cy, cy], color=ACCENT, lw=1.2, zorder=7)
    ax.plot([cx, cx + 0.62 * r], [cy, cy + 0.78 * r], color=ACCENT, lw=1.1, zorder=7)


def box(x, kind, title, sub, meas=False, fs=16.5, sfs=15.0):
    ax.add_patch(FancyBboxPatch((x - BW / 2 + 0.05, Y - BH / 2 - 0.07), BW, BH, boxstyle="round,pad=0,rounding_size=0.15",
                                fc="#000000", alpha=0.07, ec="none", zorder=2))
    ax.add_patch(FancyBboxPatch((x - BW / 2, Y - BH / 2), BW, BH, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=F[kind], ec=INK, lw=1.2, zorder=3))
    ax.text(x - 0.15, Y + 0.28, title, ha="center", va="center", fontsize=fs, color=INK, zorder=5)
    ax.text(x, Y - 0.33, sub, ha="center", va="center", fontsize=sfs, color=MUTED, zorder=5)
    if meas:
        meter(x + BW / 2 - 0.38, Y + BH / 2 - 0.32)


def note(x, text, how="classical"):
    color, weight = dict(device=(ACCENT, "bold"), light=(LIGHT, "bold"), classical=(MUTED, "normal"))[how]
    ax.text(x, Y - BH / 2 - 0.45, text, ha="center", va="top", fontsize=14, linespacing=1.25, color=color,
            fontweight=weight)


XS = [2.55, 8.2, 14.55, 21.0, 26.45, 31.9]          # preparation, attention, feed-forward, controller, mask, camera
L2 = XS[2] + BW / 2 + 0.6                             # the block's output joins the field
ax.add_patch(FancyBboxPatch((XS[1] - BW / 2 - 0.35, Y - BH / 2 - 0.35), L2 + 0.4 - (XS[1] - BW / 2 - 0.35),
                            BH + 0.7, boxstyle="round,pad=0,rounding_size=0.25", fc=F["blk"], ec=INK, lw=1.1, zorder=1))
ax.text(L2 + 0.4, Y + BH / 2 + 0.62, r"$\times 2$ blocks", ha="right", va="center", fontsize=16, color=INK)
box(XS[0], "emb", "State preparation", "a word's amplitudes")
box(XS[1], "att", "Hadamard tests", r"Re, Im $\langle q_i|k_j\rangle$, $+\,k_\mathrm{ref}$", meas=True)
box(XS[2], "ff", "Balanced reading", r"$\Sigma_j\,$Re$[(W_jh)\overline{(V_jh)}]$", meas=True)
box(XS[3], "lin", "Controller", r"$\psi=\Sigma_k A_ku\otimes B_ku$")
box(XS[4], "lin", "One mask, 16 passes", r"$|1+g\,\psi/16|/M$", fs=15)
box(XS[5], "out", "Camera", r"$|1+g\,\psi/16|^{32}$", meas=True)
for a, b in ((XS[0] + BW / 2, XS[1] - BW / 2), (XS[1] + BW / 2, XS[2] - BW / 2), (XS[2] + BW / 2, L2 + 0.4),
             (L2 + 0.4, XS[3] - BW / 2), (XS[3] + BW / 2, XS[4] - BW / 2), (XS[4] + BW / 2, XS[5] - BW / 2)):
    arrow(a, b)
ax.text(XS[0], Y + BH / 2 + 0.55, "Lily went to the park . At the", ha="center", va="center", fontsize=14.5,
        style="italic", color=INK)
arrow(XS[5] + BW / 2, XS[5] + BW / 2 + 0.45)
ax.text(XS[5] + BW / 2 + 0.55, Y, r"$P(w)$", ha="left", va="center", fontsize=18, color=INK)
# the word drawn goes back in: sample, append, repeat (as Fig. 1)
yb, xl = Y - BH / 2 - 2.55, XS[0] - BW / 2 - 0.45      # under the notes, then up the left side into the input
ax.plot([XS[5] + BW / 2 + 0.95, XS[5] + BW / 2 + 0.95, xl, xl], [Y - 0.3, yb, yb, Y], color=ACCENT, lw=1.1,
        ls=(0, (4, 2)), zorder=1)
arrow(xl, XS[0] - BW / 2, color=ACCENT)
ax.text((XS[0] + XS[5]) / 2, yb - 0.32, "sample, append, repeat", ha="center", va="center", fontsize=14,
        color=ACCENT)
ax.text(XS[1], Y + BH / 2 + 0.5, "run on ibm_kingston", ha="center", va="bottom", fontsize=14.5, color="white",
        fontweight="bold", zorder=8, bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.12", fc=ACCENT, ec="none"))
ax.text((XS[4] + XS[5]) / 2, Y + BH / 2 + 0.5, "simulated light", ha="center", va="bottom", fontsize=14.5,
        color="white", fontweight="bold", zorder=8,
        bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.12", fc=LIGHT, ec="none"))
note(XS[0], "query and key computed\nclassically, loaded as\n4 qubits' amplitudes")
note(XS[1], "5 qubits, 2,000 shots;\n32 calibration circuits\nas deep per block", how="device")
note(XS[2], "the feed-forward,\nas compiled for light", how="light")
note(XS[3], "computed classically: the\nrow sums of the measured\noverlaps, and the state $\\psi$")
note(XS[4], "the 16-pass readout,\nas compiled for light", how="light")
note(XS[5], "a photon lands on\nword $w$ with $P(w)$", how="light")
ax.text(0.25, 8.45, "Writing Why on qubits and light", ha="left", va="center", fontsize=19, fontweight="bold", color=INK)
ax.text(0.25, 7.7, SUB, ha="left", va="center", fontsize=14, color=MUTED, style="italic")
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}.png / .pdf")
