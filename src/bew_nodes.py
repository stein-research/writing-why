"""How many Gauss-Legendre nodes BEW actually needs (referee round 7): the a-priori count D/2 (3,200 for 16 passes,
200 for the plain readout) is exact for a polynomial of degree D, but along BEW's path the held F is close to one of
much lower degree. Per copy position of Table 3, BEW with 32, 64, 100 and 200 nodes against the stored shares
(results/<tag>_word_evidence.json, D/2 nodes): the largest share error, the completeness gap, and the time per word.

    python src/bew_nodes.py [--n N]      Writes results/bew_nodes.json (minutes on a laptop GPU).
"""
import json
import sys
import time

import numpy as np
import torch

import evidence as E
import tasks
from models import build

dev = "cuda" if torch.cuda.is_available() else "cpu"
args = sys.argv[1:]
NPOS = int(args[args.index("--n") + 1]) if "--n" in args else 0
d = tasks.get("tinywords", n_train=10)
xt = d["x_test"].astype(np.int64)
out = {}
for tag in ("words_born-therm16p2_s0", "words_born-p2r16_s0"):
    rows = json.load(open(f"results/{tag}_word_evidence.json"))["rows"]
    rows = rows[:NPOS] if NPOS else rows
    run = json.load(open(f"results/{tag}.json"))
    model = build(run["model"], d["vocab"], xt.shape[1])
    model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
    model = model.double().to(dev).eval()
    res = {}
    for rw in rows:
        i = rw["i"]
        ids = torch.tensor(xt[rw["window"]][None, :i + 1], device=dev)
        pos, tgt2 = torch.tensor([i], device=dev), torch.tensor([[rw["born_pred"]]], device=dev)
        b = np.asarray(rw["born_evidence"][:i])
        for nn in (32, 64, 100, 200):
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            ev, full, base = E.born(model, ids, pos, tgt2, n_nodes=nn, k=min(nn, 50))
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            ev = ev[0, 0, :i]
            while ev.dim() > 1:
                ev = ev.sum(-1)
            ev = ev.double().cpu().numpy()
            o = res.setdefault(nn, dict(err=[], gap=[], t=[]))
            o["err"].append(float(np.abs(ev - b).max()))
            o["gap"].append(abs(float(ev.sum()) - (float(full) - float(base))))
            o["t"].append(dt)
    out[tag] = dict(degree=int(model.degree()), a_priori_nodes=int(model.degree() // 2), positions=len(rows),
                    nodes={str(nn): dict(max_share_error=max(o["err"]), max_gap=max(o["gap"]),
                                         median_seconds=float(np.median(o["t"]))) for nn, o in res.items()})
    print(tag, json.dumps(out[tag]), flush=True)
json.dump(out, open("results/bew_nodes.json", "w"), indent=1)
