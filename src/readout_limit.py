"""Read a trained n-pass model at other pass counts with the same psi: how far is it from its n -> infinity limit?

For N in --ns the readout P_N(w) = |1 + g psi_w / N|^(2N) / sum_v (.) is taken with the trained body and gain; the
limit exp(2 g Re psi_w) / sum_v (.) is stood in for by N = 10^6 (n log|1 + x/n|^2 differs from 2 Re x by O(|x|^2 / n)).
Reports, over every position of the first --windows test windows: the median total-variation distance between P_N
and the limit, and accuracy and perplexity at each N. Float64 throughout.

    python readout_limit.py words_born-therm16p2_s0 [--windows 400] [--ns 16 32 64 256 1024]
"""
import argparse
import json

import numpy as np
import torch

import tasks
from models import build

p = argparse.ArgumentParser()
p.add_argument("tag")
p.add_argument("--windows", type=int, default=400)
p.add_argument("--ns", type=int, nargs="+", default=[16, 32, 64, 256, 1024])
p.add_argument("--batch", type=int, default=4)
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

run = json.load(open(f"results/{args.tag}.json"))
d = tasks.get(run["task"], n_train=1000)
xt, yt = d["x_test"][:args.windows].astype(np.int64), d["y_test"][:args.windows].astype(np.int64)
model = build(run["model"], d["vocab"], xt.shape[1])
model.load_state_dict(torch.load(f"results/{args.tag}.pt", map_location="cpu"))
model = model.double().to(dev).eval()
trained = model.thermal
LIMIT = 10 ** 6

tv = {n: [] for n in args.ns}
hits = {n: 0 for n in args.ns + [LIMIT]}
nll = {n: 0.0 for n in args.ns + [LIMIT]}
count = 0
with torch.no_grad():
    for s in range(0, len(xt), args.batch):
        x = torch.tensor(xt[s:s + args.batch], device=dev)
        y = torch.tensor(yt[s:s + args.batch], device=dev)
        m = y != -100
        probs = {}
        for n in args.ns + [LIMIT]:
            model.thermal = n
            probs[n] = model.probs(x).clamp(min=0)
            pm = probs[n][m]
            assert torch.allclose(pm.sum(-1), torch.ones_like(pm[:, 0]), atol=1e-9), f"P_{n} does not sum to 1"
            hits[n] += (pm.argmax(-1) == y[m]).sum().item()
            nll[n] -= torch.log(pm.gather(-1, y[m][:, None]).clamp(min=1e-300)).sum().item()
        for n in args.ns:
            tv[n] += (0.5 * (probs[n] - probs[LIMIT]).abs().sum(-1))[m].tolist()
        count += m.sum().item()
model.thermal = trained

res = dict(tag=args.tag, trained_n=trained, windows=args.windows, positions=count, limit_n=LIMIT,
           tv_median={n: float(np.median(tv[n])) for n in args.ns},
           acc={n: hits[n] / count for n in args.ns + [LIMIT]},
           ppl={n: float(np.exp(nll[n] / count)) for n in args.ns + [LIMIT]})
for n in args.ns + [LIMIT]:
    t = f"median TV to the limit {res['tv_median'][n]:.3f}" if n in tv else "(the limit)"
    print(f"N = {n:>7}: acc {100 * res['acc'][n]:.2f} %, ppl {res['ppl'][n]:.2f}  {t}")
print(f"{count} positions")
json.dump(res, open(f"results/{args.tag}_readout_limit.json", "w"), indent=1)
