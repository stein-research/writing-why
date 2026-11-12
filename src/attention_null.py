"""What the device tests can and cannot see: the three device prompts with the attention made uninformative
(referee round 5), on the trained 16-pass model, no job.

answer      the model's top words with the attention overlaps replaced: uniform in both blocks, in block 1 only, in
            block 2 only (every magnitude 1), random magnitudes (200 draws), and each exact overlap times
            exp(sigma N(0,1)) (200 draws per sigma). Does the device's 'same top word' need the attention at all?
evidence    function (iii) of Section 4.5 (hw_word.py --devnum: block 1's numerators and row sums from the readings)
            rebuilt with block 1's overlaps made uniform (each row's magnitudes equal, its sum of squares and the
            phases kept) or permuted within each row (10 draws); the device's own 95 % half-widths
            (results/hw_word_devnum_*.json) put about these shares, as --devpower does, and the exact shares outside
            counted. Would the test notice an uninformative block-1 attention?

    python src/attention_null.py      Writes results/attention_null.json (about 15 minutes on a laptop GPU).
"""
import json
import math

import numpy as np
import torch

import evidence as E
import tinywords
from models import Frozen, build

dev = "cuda" if torch.cuda.is_available() else "cpu"
words, index = tinywords.vocab()
run = json.load(open("results/words_born-therm16p2_s0.json"))
model = build(run["model"], tinywords.V, 128)
model.load_state_dict(torch.load("results/words_born-therm16p2_s0.pt", map_location="cpu"))
model = model.double().to(dev).eval()
name = lambda i: words[i - 3] if i >= 3 else str(i)
PROMPTS = [("park", "Lily went to the park . At the", None, "Lily_went_to_the_park_At_the"),
           ("kite", "Sue had a red kite . She flew her", None, "Sue_had_a_red_kite_She_flew_her"),
           ("kite deleted", "Sue had a red kite . She flew her", 4, None)]


def overlaps(q, k, kr):
    return torch.einsum("bihe,bjhe->bhij", q.conj(), k), (q.conj() * kr).sum(-1).transpose(1, 2)


def top_words(ids, amp, mode, rng=None, sigma=0.0):
    def hook(blk, q, k, kr):
        ov, orr = overlaps(q, k, kr)
        one = lambda: (torch.ones_like(ov), torch.ones_like(orr))
        if mode == "uniform" or (mode == "uniform block 1" and blk == 0) or (mode == "uniform block 2" and blk == 1):
            return one()
        if mode == "random":
            g = lambda t: torch.tensor(rng.random(t.shape), device=dev).to(t.dtype)
            return g(ov), g(orr)
        if mode == "noise":
            g = lambda t: t * torch.tensor(np.exp(sigma * rng.standard_normal(t.shape)), device=dev).to(t.dtype)
            return g(ov), g(orr)
        return ov, orr
    model.overlap_hook = hook
    try:
        with torch.no_grad():
            p = model.probs(ids, amp)[0, -1]
    finally:
        model.overlap_hook = None
    top = torch.topk(p, 3)
    return [(name(int(i)), round(float(v), 4)) for v, i in zip(top.values, top.indices)]


def tally(tops, n):
    v, c = np.unique(tops, return_counts=True)
    return {str(a): int(b) for b, a in sorted(zip(c, v), reverse=True)[:5]} | {"draws": n}


def block1_exact(ids):
    rec = {}
    def hook(blk, q, k, kr):
        ov, orr = overlaps(q, k, kr)
        if blk == 0:
            rec["ov"], rec["orr"] = ov.clone(), orr.clone()
        return ov, orr
    model.overlap_hook = hook
    try:
        with torch.no_grad():
            p = model.probs(ids)[0, -1]
    finally:
        model.overlap_hook = None
    return rec["ov"], rec["orr"], p


def bew_with(ids, target, ovd, orrd, ov1, orr1):
    """BEW of function (iii) with block 1's overlaps ovd, orrd in place of the exact ov1, orr1 (as hw_born_dev)."""
    L = ids.shape[1]; i = L - 1
    rat = torch.where(ov1.abs() > 0, ovd / torch.where(ov1.abs() > 0, ov1, 1), 0)
    ratr = torch.where(orr1.abs() > 0, orrd / torch.where(orr1.abs() > 0, orr1, 1), 0)
    s1, sd = ov1[0, :, i, :i].sum(-1), ovd[0, :, i, :i].sum(-1)
    def hook(blk, q, k, kr):
        ov, orr = overlaps(q, k, kr)
        if blk > 0:
            return ov, orr
        out_r = orr * ratr
        out_r[:, :, i] = orrd[0, :, i] + (orr[:, :, i] - orr1[0, :, i]) * sd / s1
        return ov * rat, out_r
    pos = torch.tensor([i], device=dev)
    a0 = model.amplitudes(ids).detach()
    fz = Frozen()
    model.overlap_hook = hook
    try:
        with torch.no_grad():
            E.born_scores(model, ids, pos, target, a0, fz)
        def f(x):
            r = x.shape[0] // ids.shape[0]
            fz.start_replay().k = r
            return E.born_scores(model, ids.repeat(r, 1), pos, target.repeat(r, 1), x, fz)
        e = E._ig(f, a0, pos, math.ceil(model.degree() / 2), k=16)
    finally:
        model.overlap_hook = None
    return e[0, 0, :i].cpu().numpy()


out = {}
for label, prompt, delete, slug in PROMPTS:
    ids = torch.tensor([[index[w] for w in prompt.split()]], device=dev)
    amp = model.amplitudes(ids)
    if delete is not None:
        amp = amp.clone(); amp[0, delete] = 0
    o = {m: top_words(ids, amp, m) for m in ("exact", "uniform", "uniform block 1", "uniform block 2")}
    rng = np.random.default_rng(0)
    o["random magnitudes"] = tally([top_words(ids, amp, "random", rng)[0][0] for _ in range(200)], 200)
    for s in (0.1, 0.3, 0.5):
        rng = np.random.default_rng(1)
        o[f"noise sigma {s}"] = tally([top_words(ids, amp, "noise", rng, s)[0][0] for _ in range(200)], 200)
    if slug:
        ov1, orr1, p = block1_exact(ids)
        target = p.argmax().view(1, 1)
        dn = json.load(open(f"results/hw_word_devnum_{slug}.json"))
        a = np.array(dn["exact"]); e_dev = np.array(dn["device_numerators"]["evidence"])
        c = np.array(dn["device_numerators"]["evidence_95"]); lo, hi = e_dev - c[:, 0], c[:, 1] - e_dev
        check = float(np.abs(bew_with(ids, target, ov1.clone(), orr1.clone(), ov1, orr1) - a).max())
        outside = lambda e: int(((a < e - lo) | (a > e + hi)).sum())
        L = ids.shape[1]
        causal = torch.tril(torch.ones(L, L, device=dev, dtype=torch.float64))
        u = torch.sqrt(((ov1.abs() * causal) ** 2).sum(-1) + orr1.abs() ** 2) / torch.sqrt(causal.sum(-1) + 1)
        ph = lambda z: torch.where(z.abs() > 0, z / torch.where(z.abs() > 0, z.abs(), 1), 1)
        e_u = bew_with(ids, target, ph(ov1) * u[..., None] * causal, ph(orr1) * u, ov1, orr1)
        rng = np.random.default_rng(0)
        perm = []
        for _ in range(10):
            ovp, orrp = ov1.clone(), orr1.clone()
            for h in range(model.heads):
                for r_ in range(L):
                    m_ = torch.cat([ov1[0, h, r_, :r_ + 1].abs(), orr1[0, h, r_:r_ + 1].abs()])
                    m_ = m_[torch.tensor(rng.permutation(len(m_)), device=dev)]
                    ovp[0, h, r_, :r_ + 1] = ph(ov1[0, h, r_, :r_ + 1]) * m_[:-1]
                    orrp[0, h, r_] = ph(orr1[0, h, r_]) * m_[-1]
            e_p = bew_with(ids, target, ovp, orrp, ov1, orr1)
            perm.append((float(np.corrcoef(a, e_p)[0, 1]), outside(e_p)))
        perm = np.array(perm)
        o["evidence"] = dict(words=len(a), rebuild_check_max_abs=check,
                             device=dict(outside=outside(e_dev), r=float(np.corrcoef(a, e_dev)[0, 1])),
                             uniform_block1=dict(outside=outside(e_u), r=float(np.corrcoef(a, e_u)[0, 1]),
                                                 shares=np.round(e_u, 4).tolist()),
                             permuted_block1=dict(outside_median=float(np.median(perm[:, 1])),
                                                  outside_min=int(perm[:, 1].min()), outside_max=int(perm[:, 1].max()),
                                                  r_median=float(np.median(perm[:, 0])), draws=10))
    out[label] = o
    print(label, json.dumps(o), flush=True)
json.dump(out, open("results/attention_null.json", "w"), indent=1)
