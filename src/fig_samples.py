"""Appendix figure: exact evidence (BEW) on the selected samples of the 16-pass model, one TinyStories story and one
WikiText-103 passage (sample.py --dump results/appendix_draws_long.json and results/appendix_draws.json; draws chosen by eye from the first five, see
paper Appendix B). Each panel explains the model's probability of one copied word at its second occurrence, with
every normaliser held (float64, degree // 2 nodes, GPU); chips as Fig. 2 (red raised, blue lowered), the current word
boxed, later words grey; bars: the five largest shares. Prints P, F_q and the gap.

python fig_samples.py   ->  figures/sample_story.{png,pdf}, figures/sample_wiki.{png,pdf}
"""
import glob
import importlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.transforms import Bbox

import evidence as E
import stories
from models import build

for f in glob.glob("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-*.otf"):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Latin Modern Sans", "DejaVu Sans"], "pdf.fonttype": 3})
INK, MUTED = "#2b2f36", "#495057"
ev_cmap = LinearSegmentedColormap.from_list("evidence", ["#08306b", "#1f78d1", "#9cc9f5", "#f7f7f7", "#f9a77e",
                                                         "#e8431c", "#7f0d0d"])
DRAWS = {**json.load(open("results/appendix_draws.json")), **{k + "|long": v for k, v in json.load(open("results/appendix_draws_long.json")).items()}}
NOSP = {",", ".", "!", "?", ";", ":", ")"}


def panel(tag, task, prompt, k, word, out, long=False):
    mod = importlib.import_module(task)
    words, _ = mod.vocab()
    ids = DRAWS[f"{tag}|{prompt}|T0.7|p0.9|s0" + ("|long" if long else "")][k]
    toks = [words[i - mod.C0] if i >= mod.C0 else "" for i in ids]
    T = [t for t, w in enumerate(toks) if w == word][1]                   # its second occurrence
    run = json.load(open(f"results/{tag}.json"))
    model = build(run["model"], mod.V, stories.L)
    model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
    model = model.double().cuda().eval()
    x = torch.tensor([ids[:T]], device="cuda")
    ev, full, fq = E.born(model, x, torch.tensor([T - 1], device="cuda"), torch.tensor([[ids[T]]], device="cuda"),
                          n_nodes=model.degree() // 2, k=128)
    ev, full, fq = ev[0, 0].cpu().numpy(), full[0, 0].item(), fq[0, 0].item()
    with torch.no_grad():
        P = model.probs(x)[0, T - 1, ids[T]].item()
    print(f"{tag} '{word}' at {T}: P {P:.4f}, F_q {fq:.1e}, sum of shares {ev.sum():.6f} = P - 1/V - F_q "
          f"{full - fq:.6f}, gap {abs(ev.sum() - (full - fq)):.1e}")
    fig = plt.figure(figsize=(7.0, 4.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    r = fig.canvas.get_renderer()
    lim = np.abs(ev[:T - 1]).max()
    L0, R0, y, lh = 0.02, 0.98, 0.95, 0.06
    xc, prev, pp, qo = L0, None, None, False
    for t, w in enumerate(toks):
        if w == "":
            if ids[t] == mod.EOS:
                break
            continue
        if w == '"':
            qo = not qo
            w = "“" if qo else "”"
        space = not (prev is None or w in NOSP or w == "”" or prev in ("“", "(")
                     or (w.isdigit() and prev.isdigit())                       # 1 2 0 -> 120
                     or (w.isdigit() and prev in (",", ".") and (pp or "").isdigit())   # 3, 5 0 0 -> 3,500
                     or w == "-" or prev == "-")                               # four - mile -> four-mile
        style = dict(fontsize=9.5, va="baseline", ha="left", color=MUTED if t > T else INK)
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
        xc, pp, prev = bb.x1, prev, w
    top = [t for t in np.argsort(-np.abs(ev[:T - 1])) if toks[t]][:5]
    yb, row, bx, bw = y - 0.1, 0.05, 0.25, 0.55
    big = abs(ev[top[0]])
    for n, t in enumerate(top):
        yy = yb - n * row
        ctx = " ".join(w for w in toks[max(0, t - 2):t + 2] if w)
        ax.text(L0, yy, f"{toks[t]}  ({ctx})", fontsize=8.5, color=INK, va="center")
        ax.add_patch(plt.Rectangle((bx + 0.12, yy - 0.015), bw * abs(ev[t]) / big, 0.03, lw=0,
                                   fc=ev_cmap(TwoSlopeNorm(0, -lim, lim)(ev[t]))))
        ax.text(bx + 0.12 + bw * abs(ev[t]) / big + 0.012, yy, f"{ev[t]:+.3f}", fontsize=8, color=MUTED, va="center")
    W, H = fig.get_size_inches()
    crop = Bbox([[0, max(0.0, yb - 4 * row - 0.05) * H], [W, H]])
    fig.savefig(f"{out}.png", dpi=300, bbox_inches=crop)
    fig.savefig(f"{out}.pdf", bbox_inches=crop)
    plt.close(fig)
    del model
    torch.cuda.empty_cache()
    return dict(P=P, Fq=fq, T=T, gap=float(abs(ev.sum() - (full - fq))))


res = {"story": panel("words_born-therm16p2_s0", "tinywords", "Once upon a time", 9, "ball", "figures/sample_story", long=True),
       "wiki": panel("wiki_born-therm16p2_s0", "wikiwords", "The river", 3, "suburbs", "figures/sample_wiki")}
json.dump(res, open("results/fig_samples.json", "w"), indent=1)
