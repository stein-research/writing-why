"""How alike do exact BEW and held gradient x input rank the context words at the word copy positions?

Reads the BEW vectors that word_evidence.py stored (results/<tag>_word_evidence.json), computes held gradient x input
for the same positions and the same explained word (the model's prediction), over the context words t < i, and
reports per position: Spearman rank correlation, whether the first-ranked word is the same, and
k_eff = sum(gxi) / sum(BEW). For a polynomial homogeneous of degree D in the context amplitudes, BEW_t = gxi_t / D
exactly (Euler), so the rankings agree and k_eff = D; the current word held at its amplitude and the residual paths
make F inhomogeneous, and k_eff shows how far. Float64.

    python word_rank_agree.py words_born-therm16p2_s0 words_born-p2r16_s0
"""
import argparse
import json

import numpy as np
import torch
from scipy.stats import spearmanr

import evidence as E
import tasks
from models import build

p = argparse.ArgumentParser()
p.add_argument("tags", nargs="+")
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = tasks.get("tinywords", n_train=1000)
xt = d["x_test"].astype(np.int64)
out = {}
for tag in args.tags:
    rows = json.load(open(f"results/{tag}_word_evidence.json"))["rows"]
    run = json.load(open(f"results/{tag}.json"))
    model = build(run["model"], d["vocab"], xt.shape[1])
    model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
    model = model.double().to(dev).eval()
    rho, same, keff = [], [], []
    for rw in rows:
        i = rw["i"]
        x = torch.tensor(xt[rw["window"]][None], device=dev)
        g = E.grad_input(model, x, torch.tensor([i], device=dev), torch.tensor([[rw["born_pred"]]], device=dev))
        g = g[0, 0, :i].cpu().numpy()
        b = np.asarray(rw["born_evidence"][:i])
        rho.append(spearmanr(b, g).statistic)
        same.append(int(np.argmax(b) == np.argmax(g)))
        keff.append(g.sum() / b.sum())
    rho, keff = np.array(rho), np.array(keff)
    out[tag] = dict(positions=len(rows), spearman_median=float(np.median(rho)), spearman_q10=float(np.quantile(rho, 0.1)),
                    spearman_min=float(rho.min()), top1_same=float(np.mean(same)), keff_median=float(np.median(keff)),
                    keff_q10=float(np.quantile(keff, 0.1)), keff_q90=float(np.quantile(keff, 0.9)), degree=model.degree())
    o = out[tag]
    print(f"{tag}: {o['positions']} positions; Spearman median {o['spearman_median']:.4f} (10th pct {o['spearman_q10']:.4f}, "
          f"min {o['spearman_min']:.4f}); same first word {o['top1_same']:.3f}; k_eff median {o['keff_median']:.1f} "
          f"[{o['keff_q10']:.1f}, {o['keff_q90']:.1f}] (degree {o['degree']})")
json.dump(out, open("results/word_rank_agree.json", "w"), indent=1)
