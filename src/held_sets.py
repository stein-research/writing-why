"""Which held normaliser makes the held F fall away from P along BEW's path (referee round 6, M1), 16-pass model.

The 16-pass readout holds, in order, the hidden normalisers (prefix mass, prenorms, each row's largest overlap, the
attention row sums), the norm of u (|u|, the readout's input), the readout's common scale (top) and Z. Per copy
position of Table 3 (results/words_born-therm16p2_s0_word_evidence.json), along x(alpha) = (alpha a_<i, a_i):
  sets      median F(alpha)/F(1) with: every normaliser held (BEW's F); only |u| recomputed; |u| and top recomputed;
            only the hidden normalisers recomputed; everything but Z recomputed; everything (the real P - 1/V);
  |u|       |u|(0.99)/|u|(1) with the hidden normalisers held, and its local power log(ratio)/log(0.99);
  rankings  per position, the Spearman correlation with the leave-one-out drop (every normaliser recomputed) of
            BEW, held gradient x input, integrated gradients of the real P, and integrated gradients of the held F
            with only |u| recomputed (512 Gauss-Legendre nodes; not a polynomial, so its completeness gap against
            F_u(1) - F_u(alpha -> 0+) is reported). Median over positions. Float64.

    python src/held_sets.py [--n N]     Writes results/held_sets.json (minutes on a laptop GPU).
"""
import json
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

import evidence as E
import tasks
from models import Frozen, build

dev = "cuda" if torch.cuda.is_available() else "cpu"
args = sys.argv[1:]
NPOS = int(args[args.index("--n") + 1]) if "--n" in args else 0
TAG = "words_born-therm16p2_s0"
d = tasks.get("tinywords", n_train=10)
xt = d["x_test"].astype(np.int64)
rows = json.load(open(f"results/{TAG}_word_evidence.json"))["rows"]
rows = rows[:NPOS] if NPOS else rows
run = json.load(open(f"results/{TAG}.json"))
model = build(run["model"], d["vocab"], xt.shape[1])
model.load_state_dict(torch.load(f"results/{TAG}.pt", map_location="cpu"))
model = model.double().to(dev).eval()
assert model.pair and model.thermal
GRID = [0.999, 0.99, 0.95, 0.8]


class Partial:
    """Replay the recorded normalisers except those whose call index is in `free` (recomputed); k repeats a batch."""
    def __init__(self, vals, free, k=1):
        self.vals, self.free, self.k, self.i = vals, free, k, 0

    def __call__(self, v):
        idx = self.i
        self.i += 1
        if idx in self.free:
            return v
        w = self.vals[idx]
        return w.repeat((self.k,) + (1,) * (w.dim() - 1)) if self.k > 1 else w


def probs_at(ids, keep, i):
    idr = ids.expand(keep.shape[0], -1)
    num, z = model.numerators(idr, model.amplitudes(idr) * keep.to(torch.float64), at=torch.tensor([i], device=dev))
    return num[:, 0] / (z[:, 0, None] + 1e-300) + model.share


def ig_of(f, a0, pos, i):
    ig = E._ig(f, a0, pos, 512, k=64)
    while ig.dim() > 3:
        ig = ig.sum(-1)
    return ig[0, 0, :i].double().cpu().numpy()


ratios, uratio, rho, gap_u = {}, [], {}, []
for rw in rows:
    i = rw["i"]
    ids = torch.tensor(xt[rw["window"]][None, :i + 1], device=dev)
    pos = torch.tensor([i], device=dev)
    a0 = model.amplitudes(ids).detach()
    shp = (1, i + 1) + (1,) * (a0.dim() - 2)
    with torch.no_grad():
        p0 = probs_at(ids, torch.ones(1, i + 1, dtype=torch.bool, device=dev), i)[0]
        tgt = int(p0.argmax())
        assert tgt == rw["born_pred"]
        drop = (p0[tgt] - probs_at(ids, ~torch.eye(i + 1, dtype=torch.bool, device=dev)[:i], i)[:, tgt]).cpu().numpy()
    tgt2 = torch.tensor([[tgt]], device=dev)
    base = Frozen()
    with torch.no_grad():
        F1 = float(E.born_scores(model, ids, pos, tgt2, a0, base))
    K = len(base.values)                      # the last three: |u|, top, Z
    U = K - 3
    sets = {"all held (BEW's F)": set(), "|u| recomputed": {U}, "|u| and top recomputed": {U, K - 2},
            "hidden normalisers recomputed": set(range(U)), "all but Z recomputed": set(range(K - 1)),
            "all recomputed (real P - 1/V)": set(range(K))}
    with torch.no_grad():
        for name, free in sets.items():
            out = []
            for al in GRID:
                s = torch.ones(i + 1, dtype=torch.float64, device=dev); s[:i] = al
                out.append(float(E.born_scores(model, ids, pos, tgt2, a0 * s.view(shp), Partial(base.values, free))) / F1)
            ratios.setdefault(name, []).append(out)
        h1 = model.hidden(ids, a0, None)[0][0, i]
        s = torch.ones(i + 1, dtype=torch.float64, device=dev); s[:i] = 0.99
        h2 = model.hidden(ids, a0 * s.view(shp), Partial(base.values, set()))[0][0, i]
        uratio.append(float((h2 ** 2).sum().sqrt() / (h1 ** 2).sum().sqrt()))
    fu = lambda x: E.born_scores(model, ids.repeat(x.shape[0], 1), pos, tgt2.repeat(x.shape[0], 1), x,
                                 Partial(base.values, {U}, x.shape[0]))
    fr = lambda x: E.born_scores(model, ids.repeat(x.shape[0], 1), pos, tgt2.repeat(x.shape[0], 1), x, Frozen())
    igu, igr = ig_of(fu, a0, pos, i), ig_of(fr, a0, pos, i)
    with torch.no_grad():
        s = torch.ones(i + 1, dtype=torch.float64, device=dev); s[:i] = 1e-12
        fu1, fu0 = float(fu(a0)), float(fu(a0 * s.view(shp)))
    gap_u.append(abs(igu.sum() - (fu1 - fu0)))
    rank = {"BEW": np.asarray(rw["born_evidence"][:i]),
            "held gradient x input": E.grad_input(model, ids, pos, tgt2)[0, 0, :i].detach().cpu().numpy(),
            "IG of the real P": igr, "IG of the held F with |u| recomputed": igu}
    for name, v in rank.items():
        rho.setdefault(name, []).append(spearmanr(v, drop).statistic)
    rho.setdefault("Spearman(IG |u| recomputed, IG real P)", []).append(spearmanr(igu, igr).statistic)
out = dict(tag=TAG, positions=len(rows), grid=GRID,
           held_over_F_median={k: np.median(np.array(v), 0).tolist() for k, v in ratios.items()},
           u_ratio_at_0_99_median=float(np.median(uratio)),
           u_ratio_at_0_99_q10_q90=np.quantile(uratio, [0.1, 0.9]).tolist(),
           u_local_power_median=float(np.median(np.log(uratio) / np.log(0.99))),
           spearman_with_leave_one_out_median={k: float(np.nanmedian(v)) for k, v in rho.items()},
           ig_u_recomputed_gap_max=float(np.max(gap_u)), ig_u_recomputed_gap_median=float(np.median(gap_u)))
print(json.dumps(out, indent=1))
json.dump(out, open("results/held_sets.json", "w"), indent=1)
