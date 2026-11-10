"""Evidence on the word model where the text says which earlier words decide the next one: word copy positions.

A copy position i of a test window (current word x_i, next word y = x_(i+1)) is one where
  - y is a word (letters) and x_i occurred earlier in the window, at j < i - 1, followed by y there (x_(j+1) = y);
  - every earlier occurrence of x_i is followed by y (the copy is unambiguous inside the context), and
  - a bigram model of the training windows gives P(y | x_i) < 0.5 (the context, not habit, decides).
Deciding words D = {j (the latest earlier key), j + 1 (the copied word)}. Up to one position per window (the first),
on the first --n windows that have one.

Only the context words t < i are ranked and deleted; the current word i always stays. It is the query: deleting it
leaves no question to answer (the Born readout at i goes to uniform), so any measure that may delete it is decided by
that one word. (Ranked with the others, the current word is the Born evidence's lowest in nine positions of ten, and
its deletion changes the prediction every time: a bottom-5 of 0.93 that says nothing about the context.) Its own
evidence is reported apart (curr_lowest: the share of positions where it is the lowest).

What is explained: the model's own predicted word at i (Born: P - 1/V, BEW exact with every denominator held;
softmax: its probability, integrated gradients over one gate per word field, the path BEW uses). Per method, over
the context words t < i:
  top1     the largest evidence is the copied word j + 1
  topk     share of D among the 2 largest
  flip1    deleting the word ranked first changes the prediction (the real model: denominators recomputed)
  bottom5  deleting the 5 words ranked last changes it (evidence only)
and the ceiling: any single context deletion changes it. Every mean has a bootstrap 95% interval (2,000 resamples
of the positions, as Reading Why). Methods: Born BEW, grad x input, attention (last block, mean of
heads), leave-one-out, random; softmax IG, grad x input, attention, leave-one-out, random. Measures are counted on
every position and on the positions where the prediction is the copy (y_hat = y).

    python word_evidence.py <born tag> <softmax tag> [--n 300] [--nodes 256] [--k 128]
"""
import argparse
import json
import time

import numpy as np
import torch

import evidence as E
import tasks
from models import build

p = argparse.ArgumentParser()
p.add_argument("born")
p.add_argument("softmax")
p.add_argument("--n", type=int, default=300, help="copy positions (one per window)")
p.add_argument("--nodes", type=int, default=512, help="softmax IG nodes (Reading Why: 512)")
p.add_argument("--k", type=int, default=128, help="BEW nodes per pass")
p.add_argument("--bigram-windows", type=int, default=300_000)
p.add_argument("--born-only", action="store_true", help="skip the softmax model (its numbers do not depend on the Born one)")
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

d = tasks.get("tinywords", n_train=args.bigram_windows)
words = d["words"]
V = d["vocab"]
is_word = np.zeros(V, bool)
for i, w in enumerate(words):
    is_word[3 + i] = w.isalpha() or ("'" in w and w.replace("'", "").isalpha())

# bigram P(y | x) from the training windows
xa, ya = d["x_train"].astype(np.int64).ravel(), d["y_train"].astype(np.int64).ravel()
key, cnt = np.unique(xa * V + ya, return_counts=True)
tot = np.bincount(xa, minlength=V)


def bigram(x, y):
    i = np.searchsorted(key, x * V + y)
    c = cnt[i] if i < len(key) and key[i] == x * V + y else 0
    return c / max(tot[x], 1)


xt, yt = d["x_test"].astype(np.int64), d["y_test"].astype(np.int64)
found = []                                                     # (window, i, j)
for r in range(len(xt)):
    x, y = xt[r], yt[r]
    for i in range(2, x.shape[0]):
        if not is_word[y[i]]:
            continue
        prev = [j for j in range(i - 1) if x[j] == x[i]]       # j < i - 1: the copied word j + 1 lies before i
        if not prev or any(x[j + 1] != y[i] for j in prev):
            continue
        if bigram(x[i], y[i]) >= 0.5:
            continue
        found.append((r, i, prev[-1]))
        break
    if len(found) == args.n:
        break
print(f"{len(found)} copy positions in the first {r + 1} test windows "
      f"(bigram counts from {args.bigram_windows:,} training windows)", flush=True)


def load(tag):
    run = json.load(open(f"results/{tag}.json"))
    m = build(run["model"], V, xt.shape[1])
    m.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
    return m.double().to(dev).eval()


def score(model, ids, keep, i, tgt):
    """The explained score at position i for a batch of deletions keep [n, L] (False = deleted): Born P - 1/V with
    the denominators recomputed (the real model), readout at i only; softmax its probability. Also the prediction."""
    n = keep.shape[0]
    idr = ids.expand(n, -1)
    if model.family == "B":
        amp = model.amplitudes(idr) * keep.to(torch.float64)
        num, z = model.numerators(idr, amp, at=torch.tensor([i], device=ids.device))
        pr = num[:, 0] / (z[:, 0, None] + 1e-300) + model.share
    else:
        pr = torch.softmax(soft_logits_at(model, model.embed(idr), torch.tensor([i], device=ids.device), keep)[:, 0], -1)
    return pr.gather(-1, tgt.expand(n)[:, None])[:, 0] - (model.share[0] if model.family == "B" else 0), pr.argmax(-1)


def soft_logits_at(model, x, at, keep=None):
    """CausalSoftmax.from_embedded with the head applied only at the positions `at` ([B, P, V], not [B, L, V])."""
    L = x.shape[1]
    allowed = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device)).expand(x.shape[0], L, L)
    if keep is not None:
        x = x * keep[..., None].to(x.dtype)
        allowed = (allowed & keep[:, None, :]) | torch.eye(L, dtype=torch.bool, device=x.device)
    h = x
    for b in model.blocks:
        h, _ = b(h, allowed)
    return model.head(model.ln(h[:, at]))


def soft_prob(model, x, pos, target):
    """softmax: the probability of the target (Reading Why's softmax evidence explains the probability)."""
    return torch.softmax(soft_logits_at(model, x, pos), -1).gather(-1, target[..., None])[..., 0]


def soft_ig(model, ids, pos, target, n_nodes):
    x0 = model.embed(ids).detach()
    with torch.no_grad():
        full = soft_prob(model, x0, pos, target)
    tgt_p = target.repeat(len(pos), 1)
    f = lambda x: soft_prob(model, x, pos, tgt_p.expand(x.shape[0], -1) if x.shape[0] != tgt_p.shape[0] else tgt_p)
    return E._ig(f, x0, pos, n_nodes, k=64).sum(-1), full, E._base(f, x0, pos)


def soft_gxi(model, ids, pos, target):
    x0 = model.embed(ids).detach().requires_grad_(True)
    (g,) = torch.autograd.grad(soft_prob(model, x0, pos, target).sum(), x0)
    return (x0 * g).sum(-1).detach()[:, None]                         # [1, 1, L]


born, soft = load(args.born), (None if args.born_only else load(args.softmax))
NAMES = ("born",) if args.born_only else ("born", "softmax")
gen = torch.Generator().manual_seed(0)
rows = []
t0 = time.time()
for n, (r, i, j) in enumerate(found):
    ids = torch.tensor(xt[r:r + 1, :i + 1], device=dev)        # causal: the prefix decides position i
    pos = torch.tensor([i], device=dev)
    D = {j, j + 1}
    row = dict(window=r, i=i, j=j, y=int(yt[r, i]))
    for name, model in [m for m in (("born", born), ("softmax", soft)) if m[0] in NAMES]:
        with torch.no_grad():
            pred = E.predict(model, ids)[:, pos]                # [1, 1]
        tgt, tgt2 = pred[:, 0], pred                          # [1] for score, [B, P] = [1, 1] for evidence.py
        row[f"{name}_pred"] = int(tgt)
        if name == "born":
            ev, full, base = E.born(model, ids, pos, tgt2, n_nodes=model.degree() // 2, k=args.k)
            row["born_gap"] = float((ev.sum(-1) - (full - base)).abs().max())
            main = ev[0, 0]
        else:
            ev, full, base = soft_ig(model, ids, pos, tgt2, args.nodes)
            row["softmax_ig_gap"] = float((ev.sum(-1) - (full - base)).abs().max())
            main = ev[0, 0]
        meth = {"evidence": main,
                "gradxinput": (E.grad_input(model, ids, pos, tgt2) if name == "born" else soft_gxi(model, ids, pos, tgt2))[0, 0],
                "attention": E.attention(model, ids, pos)[0, 0],
                "random": torch.rand(i + 1, generator=gen).to(dev)}
        # leave-one-out with the denominators recomputed (the real model), and each deletion's prediction
        with torch.no_grad():                                  # all single deletions in one batch
            base_s, _ = score(model, ids, torch.ones_like(ids, dtype=torch.bool), i, tgt)
            dl = ~torch.eye(i + 1, dtype=torch.bool, device=dev)[:i]          # one context word deleted per row
            sc, pd = score(model, ids, dl, i, tgt)
            lo = (base_s - sc).double()
            flips = (pd != tgt[0]).cpu()
        meth["leaveoneout"] = lo
        row[f"{name}_ceiling"] = bool(flips.any())
        evc = meth["evidence"][: i + 1].double().cpu()
        row[f"{name}_curr_lowest"] = bool(int(torch.argmin(evc)) == i)
        row[f"{name}_evidence"] = evc.tolist()
        for mname, s in meth.items():
            s = s[:i].double().cpu()                                   # context words only
            order = torch.argsort(s, descending=True)
            row[f"{name}_{mname}_top1"] = int(order[0]) == j + 1
            row[f"{name}_{mname}_topk"] = len(set(order[:len(D)].tolist()) & D) / len(D)
            row[f"{name}_{mname}_flip1"] = bool(flips[order[0]])
        keep = torch.ones_like(ids, dtype=torch.bool)
        keep[:, torch.argsort(meth["evidence"][:i].double().cpu())[:5].to(dev)] = False
        with torch.no_grad():
            row[f"{name}_bottom5"] = bool(score(model, ids, keep, i, tgt)[1][0] != tgt[0])
    rows.append(row)
    if (n + 1) % 20 == 0:
        print(f"  {n + 1} positions ({time.time() - t0:.0f} s)", flush=True)

def ci(v, rng=np.random.default_rng(0)):
    v = np.asarray(v, float)
    m = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


out = {}
for subset in ("all", "copied"):
    for name in NAMES:
        sel = [rw for rw in rows if subset == "all" or rw[f"{name}_pred"] == rw["y"]]
        if not sel:
            continue
        res = dict(n=len(sel), **{q: ci([rw[f"{name}_{q}"] for rw in sel]) for q in ("ceiling", "bottom5", "curr_lowest")})
        for m in ("evidence", "gradxinput", "attention", "leaveoneout", "random"):
            for q in ("top1", "topk", "flip1"):
                res[f"{m}_{q}"] = ci([rw[f"{name}_{m}_{q}"] for rw in sel])
        out[f"{name}_{subset}"] = res
        f = lambda c: f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]"
        print(f"{name} ({subset}, n {len(sel)}): ceiling {f(res['ceiling'])}, bottom5 {f(res['bottom5'])}, "
              f"current word lowest {f(res['curr_lowest'])}")
        for m in ("evidence", "gradxinput", "attention", "leaveoneout", "random"):
            print(f"  {m:12s} top1 {f(res[m + '_top1'])}  topk {f(res[m + '_topk'])}  flip1 {f(res[m + '_flip1'])}")
gaps = [rw["born_gap"] for rw in rows]
igg = [rw["softmax_ig_gap"] for rw in rows] if not args.born_only else [float("nan")]
print(f"born BEW gap: max {max(gaps):.1e}, median {np.median(gaps):.1e}; softmax IG gap: max {max(igg):.1e}, "
      f"median {np.median(igg):.1e}")
json.dump(dict(born=args.born, softmax=args.softmax, positions=len(rows), results=out, rows=rows,
               born_gap_max=max(gaps), softmax_ig_gap_max=max(igg)),
          open(f"results/{args.born}_word_evidence.json", "w"), indent=1)
