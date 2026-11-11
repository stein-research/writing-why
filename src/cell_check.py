"""The multipass cell, not ideal (referee round 3): the 16-pass readout with the two cell errors the paper's ideal cell
leaves out, on the trained 16-pass model.

(a) Pixel-dependent loss. Pixel w transmits, besides the mask, an intensity share l_w = 1 + sigma * N(0,1) per pass,
    the same at every pass (a fixed non-uniformity of the plate or mirror), so the camera reads l_w^n |1+g psi_w/n|^2n:
    the error is raised to the n-th power.
(b) Misregistration. After every pass a share eta of each pixel's field goes to each of its four grid neighbours
    (64 x 128 grid), coherently: the field is the uniform beam times the mask, real and nonnegative, so the leaked
    light adds in amplitude; zero outside the grid.

For each setting: next-word accuracy on the 64 validation windows of Table 4 (the code's 'test' split, positions
with a target), its change from the ideal cell computed by the same code, and the median total-variation distance
to the ideal P_16. Two draws of the loss pattern per sigma.

Run from the repository root:  python src/cell_check.py   (about 4 GB; minutes)   Writes results/cell_check.json.
"""
import json
import math
import os
import sys

import torch

torch.set_num_threads(4)
import tasks
from compile_optics import inner
from models import build

TAG, WINDOWS, CHUNK, N = "words_born-therm16p2_s0", 64, 4, 16
run = json.load(open(f"results/{TAG}.json")); d = tasks.get(run["task"], n_train=10)
model = build(run["model"], d["vocab"], d["x_test"].shape[1])
model.load_state_dict(torch.load(f"results/{TAG}.pt", map_location="cpu")); model = model.double().eval()
R, C = model.pair
assert R * C == d["vocab"] and model.thermal == N


def mask(x):
    """t_w in [0, 1] for every position of the windows x: [P, V], and the targets [P]."""
    with torch.no_grad():
        mid, _ = inner(model, x)
        u = mid / (mid ** 2).sum(-1, keepdim=True).sqrt()
        A, B = model._c(model.pair_a), model._c(model.pair_b)
        psi = torch.einsum("blki,blkj->blij", torch.einsum("blw,kiw->blki", u.to(A.dtype), A),
                           torch.einsum("blw,kjw->blkj", u.to(A.dtype), B)).flatten(-2)
        t = (1 + torch.exp(model.log_gain) * psi / N).abs().flatten(0, 1)
    return t / t.amax(-1, keepdim=True)


def camera(t, loss=None, eta=0.0):
    """P over the words after N passes: field a starts uniform, each pass a <- blur(a * t * sqrt(loss))."""
    per = t if loss is None else t * loss.clamp(min=0).sqrt()
    a = torch.ones_like(t)
    for _ in range(N):
        a = a * per
        if eta:
            g = a.view(-1, R, C)
            nb = torch.zeros_like(g)
            nb[:, 1:] += g[:, :-1]; nb[:, :-1] += g[:, 1:]; nb[:, :, 1:] += g[:, :, :-1]; nb[:, :, :-1] += g[:, :, 1:]
            a = ((1 - 4 * eta) * g + eta * nb).view_as(a)
        a = a / a.amax(-1, keepdim=True)                                    # rescale (P is a ratio)
    I = a * a
    return I / I.sum(-1, keepdim=True)


SIGMAS, ETAS = [0.001, 0.003, 0.01, 0.03], [0.001, 0.003, 0.01, 0.03]
gen = torch.Generator().manual_seed(2611)
patterns = [torch.randn(R * C, generator=gen, dtype=torch.float64) for _ in range(2)]
acc = {"ideal": [0, 0]}
tv = {}
for c0 in range(0, WINDOWS, CHUNK):
    x = torch.tensor(d["x_test"][c0:c0 + CHUNK]); y = torch.tensor(d["y_test"][c0:c0 + CHUNK]).flatten()
    keep = y > 0
    t = mask(x)[keep]; y = y[keep]
    P0 = camera(t)
    hit = lambda P: int((P.argmax(-1) == y).sum())
    acc["ideal"][0] += hit(P0); acc["ideal"][1] += len(y)
    settings = [(f"loss sigma {s} draw {k}", dict(loss=1 + s * patterns[k])) for s in SIGMAS for k in range(2)]
    settings += [(f"misregistration eta {e}", dict(eta=e)) for e in ETAS]
    for name, kw in settings:
        P = camera(t, **kw)
        a_ = acc.setdefault(name, [0, 0]); a_[0] += hit(P); a_[1] += len(y)
        tv.setdefault(name, []).append(0.5 * (P - P0).abs().sum(-1))
    print(f"windows {c0 + CHUNK}/{WINDOWS}", flush=True)

ideal = 100 * acc["ideal"][0] / acc["ideal"][1]
out = dict(tag=TAG, windows=WINDOWS, positions=acc["ideal"][1], passes=N, ideal_accuracy=ideal, settings={})
for name in acc:
    if name == "ideal":
        continue
    a = 100 * acc[name][0] / acc[name][1]
    out["settings"][name] = dict(accuracy=a, change=a - ideal, tv_median=torch.cat(tv[name]).median().item())
    print(f"{name:32s} accuracy {a:6.2f} ({a - ideal:+.2f})  median TV to the ideal cell {out['settings'][name]['tv_median']:.4f}")
print(f"ideal cell: {ideal:.2f} % on {acc['ideal'][1]} positions")
json.dump(out, open("results/cell_check.json", "w"), indent=1)
