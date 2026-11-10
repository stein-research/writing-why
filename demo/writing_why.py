"""Writing Why, the notebooks' engine: load a trained model, let it write, and explain any word it chose with the
exact Born evidence per word (BEW).

    import writing_why as ww
    m = ww.load("words_born-therm16p2_s0")            # the 16-copy model (TinyStories)
    story = ww.write(m, "Once upon a time", seed=0, draws=10, only=10)
    print(story.text, story.checksum)                  # the paper's Appendix B story, checksum 9b8ef1d6e5ba
    r = ww.explain(m, story.ids, story.find("ball", 2)) # the second "ball"
    ww.show(story, r)                                  # the story with each earlier word's share as a chip

The sampling loop is src/sample.py's, step for step, so a seed and a draw number give the same story there and here.
BEW: integrated gradients of P(w) - 1/V with every normaliser held, by degree // 2 Gauss-Legendre nodes, which is
exact for that polynomial; the current word stays at its amplitude, so the earlier words' shares sum to
P(w) - 1/V - F_q (F_q, the query-only baseline term, is tiny for the 16-copy model).
"""
import hashlib
import html
import importlib
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import evidence as E                    # noqa: E402
import stories                          # noqa: E402
from models import build                # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(tag):
    """A trained model from results/<tag>.pt, in float64 (a Born readout needs the range), on the GPU if there is one."""
    run = json.load(open(os.path.join(ROOT, "results", f"{tag}.json")))
    task = importlib.import_module(run["task"])            # tinywords or wikiwords: words, tokens, decode
    model = build(run["model"], task.V, stories.L)
    model.load_state_dict(torch.load(os.path.join(ROOT, "results", f"{tag}.pt"), map_location="cpu"))
    model = model.double().to(DEV).eval()
    model.tag, model.task, model.run = tag, task, run
    model.words, model.index = task.vocab()
    return model


@dataclass
class Story:
    ids: list
    text: str
    checksum: str
    ended: bool
    words: list

    def find(self, word, nth=1):
        """Position of the nth occurrence of a word in the story."""
        hits = [t for t, w in enumerate(self.words) if w == word]
        if len(hits) < nth:
            raise ValueError(f"{word!r} occurs {len(hits)} times")
        return hits[nth - 1]


def _probs(model, x):
    with torch.no_grad():
        if model.family == "B":
            return model.probs(x)[0, -1]
        return torch.softmax(model(x)[0, -1], -1)


def write(model, prompt, words=250, temp=0.7, top_p=0.9, seed=0, draws=1, only=None):
    """Draw `draws` stories from one generator seeded with `seed` (as src/sample.py) and return draw `only` (1-based;
    the last when None). Sampling: p^(1/temp), renormalised, the smallest set of words holding top_p."""
    task = model.task
    banned, eos = [0, task.UNK], task.EOS
    gen = torch.Generator(device="cpu").manual_seed(seed)
    keep = None
    for k in range(1, draws + 1):
        ids = [model.index.get(t, task.UNK) for t in task.tokens(prompt)]
        for _ in range(words):
            x = torch.tensor([ids[-stories.L:]], device=DEV)
            pv = _probs(model, x).double().clamp(min=0).cpu()
            pv[banned] = 0
            pv = pv ** (1 / temp)
            if top_p < 1:
                srt, idx = pv.sort(descending=True)
                cum = srt.cumsum(0) / srt.sum()
                srt[(cum - srt / srt.sum()) >= top_p] = 0
                pv = torch.zeros_like(pv).scatter_(0, idx, srt)
            ids.append(int(torch.multinomial(pv / pv.sum(), 1, generator=gen)))
            if ids[-1] == eos:
                break
        if only is None or k == only:
            keep = ids
    ids = keep
    text = task.decode(ids, model.words).replace("\n\n", " ").strip()
    digest = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:12]
    toks = [model.words[i - task.C0] if i >= task.C0 else "" for i in ids]
    return Story(ids, text, digest, ids[-1] == eos, toks)


def explain(model, ids, T):
    """Why the model chose the word at position T, given the words before it: BEW over the earlier words (exact, a Born
    model only). Past 128 words the model sees only the last 128, as when it wrote; earlier words get 0. Returns
    shares [T] (the current word T - 1 and later get 0), P, F_q and the completeness gap."""
    if model.family != "B":
        raise ValueError("exact evidence needs a Born model")
    start = max(0, T - stories.L)                           # the model sees the last L words, as when it wrote
    x = torch.tensor([ids[start:T]], device=DEV)
    pos, tgt = torch.tensor([T - 1 - start], device=DEV), torch.tensor([[ids[T]]], device=DEV)
    ev, full, fq = E.born(model, x, pos, tgt, n_nodes=model.degree() // 2, k=128 if DEV == "cuda" else 16)
    ev, full, fq = ev[0, 0].cpu().numpy(), full[0, 0].item(), fq[0, 0].item()
    with torch.no_grad():
        P = model.probs(x)[0, T - 1 - start, ids[T]].item()
    shares = np.zeros(T)                                    # words before the window have no share
    shares[start:] = ev[:T - start]
    return dict(T=T, shares=shares, P=P, F_q=fq, gap=abs(ev.sum() - (full - fq)), nodes=model.degree() // 2)


def _chip(v, lim):
    """Red raised, blue lowered (Reading Why's evidence colours), white near zero."""
    a = min(abs(v) / lim, 1.0) ** 0.6 if lim > 0 else 0.0
    c = np.array([232, 67, 28]) if v > 0 else np.array([31, 120, 209])
    rgb = (255 - a * (255 - c)).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", "white" if a > 0.6 else "#2b2f36"


def show(story, r, top=5):
    """The story as HTML: earlier words coloured by their share, the explained word boxed, later words grey; and the
    `top` largest shares. Returns the HTML (displayed in a notebook)."""
    from IPython.display import HTML
    T, ev = r["T"], r["shares"]
    lim = np.abs(ev[:T - 1]).max() if T > 1 else 1.0
    out = []
    for t, w in enumerate(story.words):
        if not w:
            continue
        w_html = html.escape(w).replace("\n", "<br>")
        if t == T:
            out.append(f'<b style="border:2px solid #2b2f36;border-radius:4px;padding:0 3px">{w_html}</b>')
        elif t < T - 1 and abs(ev[t]) > 0.04 * lim:
            bg, fg = _chip(ev[t], lim)
            out.append(f'<span style="background:{bg};color:{fg};border-radius:3px;padding:0 2px" '
                       f'title="{ev[t]:+.4f}">{w_html}</span>')
        else:
            out.append(f'<span style="color:{"#9aa0a6" if t > T else "#2b2f36"}">{w_html}</span>')
    best = [t for t in np.argsort(-np.abs(ev[:T - 1])) if story.words[t].strip()][:top]
    rows = "".join(f"<tr><td>{html.escape(story.words[t])}</td><td style='text-align:right'>{ev[t]:+.4f}</td></tr>"
                   for t in best)
    head = (f"<p>Why <b>{html.escape(story.words[T])}</b>: P = {r['P']:.4f}; the earlier words' shares sum to "
            f"P - 1/V - F_q = {ev.sum():.6f} (gap {r['gap']:.1e}, {r['nodes']} nodes, F_q = {r['F_q']:.1e})</p>")
    return HTML(head + "<p style='line-height:1.9;font-size:15px'>" + " ".join(out) + "</p>"
                + "<table><tr><th>word</th><th>share</th></tr>" + rows + "</table>")
