"""Figure: a story the 16-copy Born Transformer wrote, with the exact evidence (BEW) for one word it chose.

The story is sample.py's third draw from words_born-therm16p2_s0 (prompt "Once upon a time", seed 0, T 0.5,
top-p 0.9, up to 120 words), regenerated here on the CPU in float32, as it was first drawn. The explained word is "park" in "At the park" (position 41); BEW in float64 with the model's own
node count (degree // 2), every normaliser held: the shares sum to P(park) - 1/V - F_q, F_q the held
function with only the query word sent (a query-only baseline term; the path scales the context words only). Style: Reading Why's
evidence chips (red raised, blue lowered), Latin Modern Sans.

python fig_story.py [--out figures/story] [--target 41]
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
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

import evidence as E
import stories
import tinywords
from models import build

for f in glob.glob("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-*.otf"):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Latin Modern Sans", "DejaVu Sans"], "mathtext.fontset": "custom",
                     "mathtext.rm": "Latin Modern Sans", "mathtext.it": "Latin Modern Sans:italic",
                     "mathtext.bf": "Latin Modern Sans:bold", "pdf.fonttype": 3})
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "figures/story"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
T = int(sys.argv[sys.argv.index("--target") + 1]) if "--target" in sys.argv else 41
INK, MUTED, LINE, ACCENT = "#2b2f36", "#495057", "#c9ced6", "#d9480f"
ev_cmap = LinearSegmentedColormap.from_list("evidence", ["#08306b", "#1f78d1", "#9cc9f5", "#f7f7f7", "#f9a77e",
                                                         "#e8431c", "#7f0d0d"])        # Reading Why's

tag = "words_born-therm16p2_s0"
words, index = tinywords.vocab()
run = json.load(open(f"results/{tag}.json"))
model = build(run["model"], tinywords.V, stories.L)
model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
model.eval()
gen = torch.Generator().manual_seed(0)
for _ in range(3):                                            # sample.py's draws 1-3 from one generator
    ids = [index.get(t, tinywords.UNK) for t in tinywords.tokens("Once upon a time")]
    for _ in range(120):
        with torch.no_grad():
            pv = model.probs(torch.tensor([ids[-stories.L:]]))[0, -1].double().clamp(min=0)
        pv[[0, tinywords.UNK]] = 0
        pv = pv ** 2                                          # T = 0.5
        srt, idx = pv.sort(descending=True)
        cum = srt.cumsum(0) / srt.sum()
        srt[(cum - srt / srt.sum()) >= 0.9] = 0
        pv = torch.zeros_like(pv).scatter_(0, idx, srt)
        ids.append(int(torch.multinomial(pv / pv.sum(), 1, generator=gen)))
        if ids[-1] == tinywords.EOS:
            break
toks = [words[i - tinywords.C0] if i >= tinywords.C0 else "" for i in ids]
assert toks[T] == "park", f"the story changed: position {T} is {toks[T]!r}"

dev = "cuda" if torch.cuda.is_available() else "cpu"           # BEW on the GPU; the story was drawn on the CPU
model = model.double().to(dev)
x = torch.tensor([ids[:T]], device=dev)
ev, full, zero = E.born(model, x, torch.tensor([T - 1], device=dev), torch.tensor([[ids[T]]], device=dev),
                        n_nodes=model.degree() // 2)
ev, full, zero = ev[0, 0].cpu().numpy(), full[0, 0].item(), zero[0, 0].item()
with torch.no_grad():
    P = model.probs(x)[0, T - 1, ids[T]].item()
gap = abs(ev.sum() - (full - zero))
print(f"'park': P {P:.4f}, P - 1/V {full:.6f}; query alone F_q {zero:.2e}; sum of shares {ev.sum():.6f} "
      f"= P - 1/V - F_q {full - zero:.6f}, gap {gap:.1e}")

HF = 4.6
fig = plt.figure(figsize=(7.0, HF))
k = 3.25 / HF                                                  # story layout in inches, as at 3.25 in
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
ax.set_xlim(0, 1), ax.set_ylim(0, 1)
r = fig.canvas.get_renderer()
lim = np.abs(ev[:T - 1]).max()
L0, R0, y, lh = 0.02, 0.98, 1 - 0.07 * k, 0.085 * k
NOSP = {",", ".", "!", "?", ";", ":"}
xc, prev, qo = L0, None, False
for t, w in enumerate(toks):
    if w == "":
        break
    if w == "\n":
        xc, y, prev = L0, y - lh * 1.25, None
        continue
    if w == '"':
        qo = not qo
        w = "“" if qo else "”"
    space = not (prev is None or w in NOSP or w == "”" or prev == "“")
    style = dict(fontsize=9.5, va="baseline", ha="left", color=MUTED if t > T else INK, zorder=3)
    if t == T:
        style.update(fontweight="bold", bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=INK, lw=1.1))
    elif t < T - 1 and w not in NOSP and abs(ev[t]) > 0.04 * lim:
        style["bbox"] = dict(boxstyle="round,pad=0.15", lw=0, fc=ev_cmap(TwoSlopeNorm(0, -lim, lim)(ev[t])))
        style["color"] = "white" if abs(ev[t]) > 0.55 * lim else INK
    txt = ax.text(xc + (0.008 if space else 0.001), y, w, **style)
    bb = txt.get_window_extent(renderer=r).transformed(ax.transData.inverted())
    if bb.x1 > R0:
        xc, y = L0, y - lh
        txt.set_position((xc, y))
        bb = txt.get_window_extent(renderer=r).transformed(ax.transData.inverted())
    xc, prev = bb.x1, w
# the five largest shares as bars, each word with its context
CTX = {39: "At (just before)", 34: "park (\u201cwent to the park\u201d)", 31: "went", 1: "upon (\u201cOnce upon\u201d)",
       26: "One (\u201cOne day\u201d)"}
top = [t for t in np.argsort(-ev[:T - 1]) if toks[t] not in ("", "\n")][:5]
yb, row, bx, bw = y - 0.14 * k, 0.072 * k, 0.33, 0.48
for n, t in enumerate(top):
    yy = yb - n * row
    ax.text(L0, yy, CTX.get(int(t), toks[t]), fontsize=8.5, color=INK, va="center")
    ax.add_patch(plt.Rectangle((bx, yy - 0.022 * k), bw * ev[t] / ev[top[0]], 0.044 * k, lw=0,
                               fc=ev_cmap(TwoSlopeNorm(0, -lim, lim)(ev[t]))))
    ax.text(bx + bw * ev[t] / ev[top[0]] + 0.012, yy, f"+{ev[t]:.3f}", fontsize=8, color=MUTED, va="center")
y_end = yb - 4 * row
from matplotlib.transforms import Bbox
W, H = fig.get_size_inches()
crop = Bbox([[0, max(0.0, y_end - 0.07 * k) * H], [W, H]])          # down to the last bar
fig.savefig(f"{out}.png", dpi=300, bbox_inches=crop)
fig.savefig(f"{out}.pdf", bbox_inches=crop)
print(f"wrote {out}.png / .pdf")
