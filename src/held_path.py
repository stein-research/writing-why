"""How the held function behaves along BEW's path, and what integrated gradients of the real probability give
(referee round 4: the held F falls away from the real P as soon as the context is scaled).

On the copy positions of Table 3 (results/<born tag>_word_evidence.json, the model's own predicted word), per position:
  path     along x(alpha) = (alpha a_<i, a_i): the held F(alpha) / F(1) and the real (P(alpha) - 1/V) / (P(1) - 1/V),
           denominators recomputed, at a grid of alpha; alpha_half, the largest alpha where each falls below half;
           and the end points: F_q = F(0) held against the real P(0) - 1/V (the deletion of every earlier word).
  real IG  integrated gradients of the real P - 1/V (denominators recomputed, not a polynomial) along the same
           path; the real P jumps at alpha = 0 (all context deleted), so its gap is taken against alpha -> 0+;
           path, --nodes Gauss-Legendre nodes; its completeness gap against P(1) - P(0); and the measures of Table 3
           (copied word first, deciding pair in the top 2, flip@1, bottom-5), deletions in the real model.
  held-free check: gradient x input rescaled to sum to F - F_q, the ratio of its shares to BEW's (k_eff).
Bootstrap 95 % intervals over positions as word_evidence.py. Float64.

    python src/held_path.py words_born-therm16p2_s0 words_born-p2r16_s0 [--nodes 512]
Writes results/held_path.json.
"""
import argparse
import json
import time

import numpy as np
import torch

import evidence as E
import tasks
from models import Frozen, build
from scipy.stats import spearmanr

p = argparse.ArgumentParser()
p.add_argument("tags", nargs="+")
p.add_argument("--nodes", type=int, default=512)
p.add_argument("--k", type=int, default=64)
p.add_argument("--n", type=int, default=0, help="first n positions only (a quick check)")
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = tasks.get("tinywords", n_train=10)
xt = d["x_test"].astype(np.int64)
GRID = [1.0, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.5, 1e-12, 0.0]   # 1e-12: the real P jumps at 0 (below)


def ci(v, rng=None):
    rng = rng or np.random.default_rng(0)
    v = np.asarray(v, float)
    m = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def real_score(model, ids, keep, i, tgt):
    """Real P - 1/V at i (denominators recomputed) and the prediction, for a batch of deletions keep [n, L]."""
    n = keep.shape[0]
    idr = ids.expand(n, -1)
    num, z = model.numerators(idr, model.amplitudes(idr) * keep.to(torch.float64), at=torch.tensor([i], device=ids.device))
    pr = num[:, 0] / (z[:, 0, None] + 1e-300) + model.share
    return pr.gather(-1, tgt.expand(n)[:, None])[:, 0] - model.share[0], pr.argmax(-1)


out = {}
for tag in args.tags:
    rows = json.load(open(f"results/{tag}_word_evidence.json"))["rows"]
    rows = rows[:args.n] if args.n else rows
    run = json.load(open(f"results/{tag}.json"))
    model = build(run["model"], d["vocab"], xt.shape[1])
    model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
    model = model.double().to(dev).eval()
    per = dict(held=[], real=[], ahalf_held=[], ahalf_real=[], fq_held=[], fq_real=[], F=[], gap=[], rel_gap=[],
               top1=[], topk=[], flip1=[], bottom5=[], keff=[], rho=[], same1=[], jump=[])
    t0 = time.time()
    for n, rw in enumerate(rows):
        i, j, tgtv = rw["i"], rw["j"], rw["born_pred"]
        ids = torch.tensor(xt[rw["window"]][None, :i + 1], device=dev)
        pos, tgt2 = torch.tensor([i], device=dev), torch.tensor([[tgtv]], device=dev)
        a0 = model.amplitudes(ids).detach()
        ctx_shape = (1, i + 1) + (1,) * (a0.dim() - 2)
        fz = Frozen()
        with torch.no_grad():
            F1 = float(E.born_scores(model, ids, pos, tgt2, a0, fz))
            held, real = [], []
            for al in GRID:
                s = torch.ones(i + 1, dtype=torch.float64, device=dev); s[:i] = al
                x = a0 * s.view(ctx_shape)
                fz.start_replay().k = 1
                held.append(float(E.born_scores(model, ids, pos, tgt2, x, fz)))
                real.append(float(E.born_scores(model, ids, pos, tgt2, x, Frozen())))
        held, real = np.array(held), np.array(real)
        per["held"].append((held / held[0]).tolist()); per["real"].append((real / real[0]).tolist())
        ga = np.array(GRID)
        per["ahalf_held"].append(float(ga[np.argmax(held / held[0] < 0.5)]))
        per["ahalf_real"].append(float(ga[np.argmax(real / real[0] < 0.5)]) if (real / real[0] < 0.5).any() else 0.0)
        per["fq_held"].append(held[-1]); per["fq_real"].append(real[-1]); per["F"].append(F1)
        # integrated gradients of the real P - 1/V
        f = lambda x: E.born_scores(model, ids.repeat(x.shape[0], 1), pos, tgt2.repeat(x.shape[0], 1), x, Frozen())
        ig = E._ig(f, a0, pos, args.nodes, k=args.k)
        while ig.dim() > 3:
            ig = ig.sum(-1)
        ev = ig[0, 0, :i].double().cpu()
        # the real P is not continuous at alpha = 0 (an absent context changes the normalisers discontinuously):
        # the integral reaches the limit alpha -> 0+, not the deletion of every earlier word
        gap = float(ev.sum()) - (real[0] - real[-2])
        per["gap"].append(abs(gap)); per["rel_gap"].append(abs(gap) / max(abs(real[0] - real[-2]), 1e-300))
        per["jump"].append(abs(real[-1] - real[-2]) / abs(real[0]))
        D = {j, j + 1}
        order = torch.argsort(ev, descending=True)
        per["top1"].append(int(order[0]) == j + 1)
        per["topk"].append(len(set(order[:2].tolist()) & D) / 2)
        with torch.no_grad():
            tgt = torch.tensor([tgtv], device=dev)
            keep = torch.ones(1, i + 1, dtype=torch.bool, device=dev); keep[0, int(order[0])] = False
            per["flip1"].append(bool(real_score(model, ids, keep, i, tgt)[1][0] != tgtv))
            keep = torch.ones(1, i + 1, dtype=torch.bool, device=dev); keep[0, order[-5:].to(dev)] = False
            per["bottom5"].append(bool(real_score(model, ids, keep, i, tgt)[1][0] != tgtv))
        # BEW against gradient x input rescaled to sum to F - F_q (the held function's completeness)
        b = np.asarray(rw["born_evidence"][:i])
        g = E.grad_input(model, ids, pos, tgt2)[0, 0, :i].cpu().numpy()
        per["keff"].append(float(g.sum() / b.sum()))
        gr = g * b.sum() / g.sum()
        per["rho"].append(float(np.abs(gr - b).sum() / np.abs(b).sum()))
        per["same1"].append(float(spearmanr(ev.numpy(), b).statistic))
        if (n + 1) % 25 == 0:
            print(f"  {tag}: {n + 1}/{len(rows)} ({time.time() - t0:.0f} s)", flush=True)
    H, R = np.array(per["held"]), np.array(per["real"])
    o = dict(positions=len(rows), nodes=args.nodes, grid=GRID,
             held_over_F_median=np.median(H, 0).tolist(), real_over_P_median=np.median(R, 0).tolist(),
             alpha_half_held_median=float(np.median(per["ahalf_held"])),
             alpha_half_real_median=float(np.median(per["ahalf_real"])),
             fq_held_median=float(np.median(per["fq_held"])), fq_real_median=float(np.median(per["fq_real"])),
             fq_real_over_F_median=float(np.median(np.array(per["fq_real"]) / np.array(per["F"]))),
             real_jump_at_0_over_P_median=float(np.median(per["jump"])),
             real_jump_at_0_over_P_q90=float(np.quantile(per["jump"], 0.9)),
             real_ig_gap_max=float(np.max(per["gap"])), real_ig_gap_median=float(np.median(per["gap"])),
             real_ig_rel_gap_max=float(np.max(per["rel_gap"])), real_ig_rel_gap_median=float(np.median(per["rel_gap"])),
             real_ig=dict(top1=ci(per["top1"]), topk=ci(per["topk"]), flip1=ci(per["flip1"]),
                          bottom5=ci(per["bottom5"])),
             spearman_real_ig_vs_bew_median=float(np.nanmedian(per["same1"])),
             keff_median=float(np.median(per["keff"])),
             rescaled_gxi_l1_over_bew_median=float(np.median(per["rho"])),
             rescaled_gxi_l1_over_bew_q90=float(np.quantile(per["rho"], 0.9)))
    out[tag] = o
    print(tag, json.dumps({k: v for k, v in o.items() if k not in ("grid",)}, indent=0), flush=True)
json.dump(out, open("results/held_path.json", "w"), indent=1)
