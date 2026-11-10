"""Figure: Writing Why's two Born Transformers side by side, drawn as optics, bottom to top: the one-copy readout (every
step a single Born measurement) and the 16-copy readout, identical up to the readout, where the light passes one mask 16 times
before the camera (|1 + g psi / 16|^32: still a Born probability, of 16 copies). As compile_optics.py builds it:
the controller computes psi from block 2's 64 measured numbers and writes one SLM - psi itself (one pass) or
1 + g psi / 16 (16 passes through a multipass cell). Both write one word at a time.
Style and helpers from fig_arch.py (Reading Why's fig_compare3.py). Numbers: mean and sd of three seeds on fresh_test.py's fresh test windows (TinyStories words, 1M windows).

python fig_born_pair.py [--out figures/born_pair]
"""
import glob
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge

for f in glob.glob("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-*.otf"):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Latin Modern Sans", "DejaVu Sans"], "mathtext.fontset": "custom",
                     "mathtext.rm": "Latin Modern Sans", "mathtext.it": "Latin Modern Sans:italic",
                     "mathtext.bf": "Latin Modern Sans:bold", "pdf.fonttype": 3})
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "figures/born_pair"

INK, MUTED, LINE, ACCENT, BEAM = "#2b2f36", "#495057", "#c9ced6", "#d9480f", "#ffd43b"
# box colours of the original transformer figure (Vaswani et al. 2017)
F = dict(emb="#f9dcdf", att="#fbe3c4", norm="#f3f5c6", ff="#cfe8f4", lin="#dcdcf2", out="#d6ecd2", blk="#f2f2f2")

fig = plt.figure(figsize=(12.2, 12.8))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16.3), ax.set_ylim(-0.4, 16.4)
ax.axis("off")

Y = dict(inp=0.55, emb=1.65, pos=2.75, att=4.25, an1=5.55, ff=6.95, an2=8.1, lin=9.85, mask=11.0, out=12.15, prob=13.1)
BLOCK = (3.5, 8.75)
BANDS = [("embedding", 1.05, 3.25), ("attention", 3.5, 6.55), ("feed-forward", 6.55, 8.75), ("readout", 9.4, 13.55)]
C1, C2, BW, BH = 4.0, 11.7, 4.4, 0.86
PROMPT = "Once upon a time"

def arrow(x0, y0, x1, y1, lw=1.1, color=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=4.5,head_width=2.6", color=color,
                                 lw=lw, shrinkA=0, shrinkB=0, zorder=4))


def meter(cx, cy, r=0.14, lw=1.2):
    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=0, theta2=180, color=ACCENT, lw=lw, zorder=7))
    ax.plot([cx - r, cx + r], [cy, cy], color=ACCENT, lw=lw, solid_capstyle="round", zorder=7)
    ax.plot([cx, cx + 0.62 * r], [cy, cy + 0.78 * r], color=ACCENT, lw=lw * 0.9, solid_capstyle="round", zorder=7)


# ---- small optical icons (drawn at the left of a box)
def i_slm(x, y):
    for r in range(3):
        for c in range(4):
            ax.add_patch(Rectangle((x - 0.2 + c * 0.1, y - 0.15 + r * 0.1), 0.09, 0.09,
                                   fc=plt.cm.twilight((r * 4 + c) / 12), ec="none", zorder=6))


def i_bs(x, y, s=0.17):
    ax.add_patch(Rectangle((x - s, y - s), 2 * s, 2 * s, fc="#e7f5ff", ec=INK, lw=0.9, zorder=6))
    ax.plot([x - s, x + s], [y - s, y + s], color=INK, lw=0.9, zorder=7)


def i_det(x, y):
    ax.add_patch(Wedge((x, y), 0.11, 90, 270, fc=INK, ec="none", zorder=7))


def i_interf(x, y):
    """A beam splitter and two detectors: balanced detection."""
    i_bs(x - 0.12, y)
    i_det(x + 0.22, y + 0.12), i_det(x + 0.22, y - 0.12)


def i_lens(x, y):
    ax.add_patch(Ellipse((x, y), 0.14, 0.5, fc="#d0ebff", ec="#74c0fc", lw=1.0, zorder=6))


def i_cam(x, y):
    ax.add_patch(Rectangle((x - 0.18, y - 0.18), 0.36, 0.36, fc="#1b1b1b", ec=INK, lw=0.8, zorder=6))
    for k in range(9):
        r, c = divmod(k, 3)
        ax.add_patch(Rectangle((x - 0.15 + c * 0.1, y - 0.15 + r * 0.1), 0.08, 0.08,
                               fc=plt.cm.inferno(0.2 + 0.08 * ((k * 5) % 9)), ec="none", zorder=7))


def i_plate(x, y):
    ax.add_patch(Rectangle((x - 0.06, y - 0.22), 0.12, 0.44, fc="#e5dbff", ec="#7048e8", lw=0.9, zorder=6))


def box(x, y, kind, title, sub=None, w=BW, h=BH, fs=15.5, meas=False, icon=None, sfs=15, sdx=0.0):
    ax.add_patch(FancyBboxPatch((x - w / 2 + 0.04, y - h / 2 - 0.05), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="#000000", alpha=0.07, ec="none", zorder=2))
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=F[kind], ec=INK, lw=1.2, zorder=3))
    tx = x                                              # centred (icons sit at the left edge)
    if sub:
        ax.text(tx, y + 0.17, title, ha="center", va="center", fontsize=fs, color=INK, zorder=5)
        ax.text(tx + sdx, y - 0.21, sub, ha="center", va="center", fontsize=sfs, color=MUTED, zorder=5)
    else:
        ax.text(tx, y, title, ha="center", va="center", fontsize=fs, color=INK, zorder=5)
    if icon:
        icon(x - w / 2 + 0.38, y)
    if meas:
        meter(x + w / 2 - 0.3, y + h / 2 - 0.22)


def plus(x, y, r=0.17):
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK, lw=1.1, zorder=5))
    ax.plot([x - r * 0.6, x + r * 0.6], [y, y], color=INK, lw=1.0, zorder=6)
    ax.plot([x, x], [y - r * 0.6, y + r * 0.6], color=INK, lw=1.0, zorder=6)


def residual(x, y_from, y_to, side=-1, color=INK):
    xs = x + side * (BW / 2 + 0.3)
    ax.plot([x, xs, xs], [y_from, y_from, y_to], color=color, lw=1.0, zorder=1)
    arrow(xs, y_to, x + side * BW / 2, y_to, lw=1.0, color=color)



def i_mesh(x, y):
    """A readout mesh: a lattice of small beam splitters."""
    for r in range(2):
        for c in range(3):
            i_bs(x - 0.13 + c * 0.13, y - 0.07 + r * 0.14, s=0.055)


for name, y0, y1 in BANDS:
    ax.add_patch(Rectangle((0.15, y0), 16.0, y1 - y0, fc="#fafafa" if name != "feed-forward" else "#f6f6f6",
                           ec="none", zorder=0))
    ax.text(0.3, (y0 + y1) / 2, name, rotation=90, ha="center", va="center", fontsize=16, color=MUTED,
            style="italic")

HEAD = [(C1, "Born Transformer, one copy", "every nonlinearity a quadratic detection;\nthe camera reads the field once"),
        (C2, "Born Transformer, 16 copies", "the same up to the readout: one mask passed\n16 times, as 16 copies in distribution")]
for x, t, s in HEAD:
    ax.text(x, 16.0, t, ha="center", va="center", fontsize=19.5, fontweight="bold", color=INK)
    ax.text(x, 15.7, s, ha="center", va="top", fontsize=16, color=MUTED, style="italic", linespacing=1.25)


def ghost(x, y, text, w=BW, h=BH):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="none", ec=LINE, lw=1.3, ls=(0, (4, 3)), zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=14.5, color=MUTED, style="italic", zorder=5)


def i_passes(x, y):
    """One mask, the light through it again and again (n passes)."""
    i_slm(x + 0.02, y)
    for k, dy in enumerate((-0.2, 0.0, 0.2)):
        ax.annotate("", xy=(x + 0.28, y + dy), xytext=(x - 0.28, y + dy),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.8, shrinkA=0, shrinkB=0), zorder=8)


def i_chip(x, y):
    """The controller: electronics computing psi from the measured numbers."""
    ax.add_patch(Rectangle((x - 0.15, y - 0.15), 0.3, 0.3, fc="#495057", ec=INK, lw=0.8, zorder=6))
    for k in range(3):
        o = -0.09 + k * 0.09
        for (dx, dy, w, h) in ((o - 0.015, 0.15, 0.03, 0.07), (o - 0.015, -0.22, 0.03, 0.07),
                               (0.15, o - 0.015, 0.07, 0.03), (-0.22, o - 0.015, 0.07, 0.03)):
            ax.add_patch(Rectangle((x + dx, y + dy), w, h, fc=INK, ec="none", zorder=6))


def column(x, thermal):
    # the light: modulator to block 2's homodyne, then written again by the mask (the controller is electronics)
    for y0, y1 in ((Y["emb"], BLOCK[1]), (Y["mask"], Y["out"])):
        ax.add_patch(Rectangle((x - 0.32, y0), 0.64, y1 - y0, fc=BEAM, alpha=0.32, ec="none", zorder=1.5))
    ax.text(x, Y["inp"], PROMPT, ha="center", va="center", fontsize=15, style="italic", color=INK)
    arrow(x, Y["inp"] + 0.2, x, Y["emb"] - BH / 2)
    box(x, Y["emb"], "emb", "Modulator", "32 modes per word", icon=i_slm, sfs=14.2)
    arrow(x, Y["emb"] + BH / 2, x, Y["pos"] - 0.17)
    plus(x, Y["pos"])
    sx = x - 1.0
    i_plate(sx, Y["pos"])
    ax.plot([sx + 0.06, x - 0.17], [Y["pos"], Y["pos"]], color=INK, lw=1.0, zorder=4)
    ax.text(sx - 0.2, Y["pos"], "position field,\nphase " + r"$e^{i\omega_k t}$", ha="right",
            va="center", fontsize=15, color=INK, linespacing=1.2)
    arrow(x, Y["pos"] + 0.17, x, Y["att"] - BH / 2)
    ax.add_patch(FancyBboxPatch((x - BW / 2 - 0.45, BLOCK[0]), BW + 0.9, BLOCK[1] - BLOCK[0],
                                boxstyle="round,pad=0,rounding_size=0.25", fc=F["blk"], ec=INK, lw=1.2, zorder=1))
    ax.text(x + BW / 2 + 0.58, (BLOCK[0] + BLOCK[1]) / 2, r"$\times 2$", ha="left", va="center", fontsize=17,
            color=INK)
    box(x, Y["att"], "att", "Interferometers", r"$|\langle q_i|k_j\rangle|^2$ / row sum, $j \leq i$, $+\,k_\mathrm{ref}$", meas=True,
        icon=i_interf, sfs=13.6, sdx=0.15)
    box(x, Y["an1"], "norm", "Beam splitter", h=0.56, fs=15, icon=i_bs)
    box(x, Y["ff"], "ff", "Homodyne", r"$\Sigma_{j=1}^{4}\,$Re$[(W_j h)\overline{(V_j h)}]$", meas=True,
        sfs=14.0, sdx=0.1, icon=i_interf)
    box(x, Y["an2"], "norm", "Beam splitter", h=0.56, fs=15, icon=i_bs)
    arrow(x, Y["att"] + BH / 2, x, Y["an1"] - 0.25)
    arrow(x, Y["an1"] + 0.25, x, Y["ff"] - BH / 2)
    arrow(x, Y["ff"] + BH / 2, x, Y["an2"] - 0.25)
    residual(x, Y["att"] - BH / 2 - 0.19, Y["an1"])
    residual(x, Y["an1"] + 0.25 + 0.2, Y["an2"])
    arrow(x, Y["an2"] + 0.25, x, Y["lin"] - BH / 2)
    ax.text(x + 0.42, (BLOCK[1] + Y["lin"] - BH / 2) / 2, "64 numbers measured\nby block 2's homodyne", ha="left",
            va="center", fontsize=12, linespacing=1.15, color=MUTED, style="italic", zorder=6)
    box(x, Y["lin"], "lin", "Controller", r"$\psi = \Sigma_k\, A_k u \otimes B_k u$, rank 16",
        icon=i_chip, sfs=13.8)
    arrow(x, Y["lin"] + BH / 2, x, Y["mask"] - BH / 2)
    if thermal:
        box(x, Y["mask"], "lin", "One mask, 16 passes", r"transmission $1 + g\,\psi/16$", icon=i_passes, sfs=14.2)
        box(x, Y["out"], "out", "Camera", r"$|1 + g\,\psi/16|^{32}$, 8,192 pixels", meas=True,
            icon=i_cam, sfs=14.2)
    else:
        box(x, Y["mask"], "lin", "One mask, one pass", r"field $\psi$", icon=i_slm, sfs=14.2)
        box(x, Y["out"], "out", "Camera", r"$|\psi|^2$, 8,192 pixels", meas=True, icon=i_cam,
            sfs=14.2)
    arrow(x, Y["mask"] + BH / 2, x, Y["out"] - BH / 2)
    arrow(x, Y["out"] + BH / 2, x, Y["prob"] - 0.12)
    ax.text(x, Y["prob"] + 0.1, r"$P(w)$  Born probabilities of the next word", ha="center", va="center",
            fontsize=15, color=INK)
    lx = x + BW / 2 + 1.25
    yl = Y["prob"] + 0.1                                     # from the end of the label: the sampled word goes back in
    ax.plot([x + 2.5, lx, lx], [yl, yl, Y["inp"]], color=ACCENT, lw=1.1, zorder=1, ls=(0, (4, 2)))
    arrow(lx, Y["inp"], x + 1.55, Y["inp"], lw=1.1, color=ACCENT)
    ax.text(lx + 0.12, (Y["emb"] + Y["out"]) / 2, "sample, append, repeat", rotation=90, ha="left", va="center",
            fontsize=14, color=ACCENT)


column(C1, False)
column(C2, True)

FOOT = [(C1, "TinyStories 43.0 $\\pm$ 0.1 %, perplexity 16.9 (3 seeds)\nBEW exact: degree 400, gap $10^{-13}$"),
        (C2, "TinyStories 49.0 $\\pm$ 0.1 %, perplexity 9.80 (3 seeds)\nBEW exact: degree 6,400, gap $5 \\times 10^{-11}$")]
for x, t in FOOT:
    ax.text(x, 14.25, t, ha="center", va="center", fontsize=15, color=INK, linespacing=1.35)
meter(0.6, -0.15)
ax.text(0.9, -0.15, "a measurement (detectors)", ha="left", va="center", fontsize=16, color=MUTED)
ax.add_patch(Rectangle((5.6, -0.3), 0.5, 0.3, fc=BEAM, alpha=0.32, ec="none"))
ax.text(6.2, -0.15, "the light: detectors read split-off parts; the main path runs on to block 2", ha="left", va="center", fontsize=16,
        color=MUTED)

for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}.png / .pdf")
