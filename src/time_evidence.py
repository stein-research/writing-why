"""Wall-clock per explained word at the word copy positions: BEW (degree // 2 nodes), held gradient x input, and the
softmax model's integrated gradients (512 nodes). First --n positions of results/<born>_word_evidence.json, one position
at a time, after one warm-up; GPU, float64 for Born (as word_evidence.py), CUDA synchronised around each call.

    python time_evidence.py words_born-therm16p2_s0 words_born-p2r16_s0 words_softmax-w96_s0 [--n 5]
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
p.add_argument("born", nargs=2)
p.add_argument("softmax")
p.add_argument("--n", type=int, default=5)
args = p.parse_args()
d = tasks.get("tinywords", n_train=1000)
xt = d["x_test"].astype(np.int64)
rows = json.load(open(f"results/{args.born[0]}_word_evidence.json"))["rows"][:args.n + 1]


def load(tag, dtype):
    run = json.load(open(f"results/{tag}.json"))
    m = build(run["model"], d["vocab"], xt.shape[1])
    m.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
    return m.to(dtype).cuda().eval()


def clock(fn):
    ts = []
    for k, rw in enumerate(rows):
        x = torch.tensor(xt[rw["window"]][None], device="cuda")
        pos, tgt = torch.tensor([rw["i"]], device="cuda"), torch.tensor([[rw["born_pred"]]], device="cuda")
        torch.cuda.synchronize(); t0 = time.time()
        fn(x, pos, tgt)
        torch.cuda.synchronize()
        if k:                                                 # the first is a warm-up
            ts.append(time.time() - t0)
    return float(np.median(ts))


out = {}
for tag in args.born:
    m = load(tag, torch.float64)
    out[f"{tag} BEW ({m.degree() // 2} nodes)"] = clock(lambda x, pos, tgt: E.born(m, x, pos, tgt, n_nodes=m.degree() // 2, k=128))
    out[f"{tag} grad x input (held)"] = clock(lambda x, pos, tgt: E.grad_input(m, x, pos, tgt))
    del m; torch.cuda.empty_cache()
m = load(args.softmax, torch.float64)
out[f"{args.softmax} IG (512 nodes)"] = clock(lambda x, pos, tgt: E.softmax_ig(m, x, pos, tgt, n_nodes=512))
out[f"{args.softmax} grad x input"] = clock(lambda x, pos, tgt: E.grad_input(m, x, pos, tgt))
for k, v in out.items():
    print(f"{k}: {v:.3f} s per explained word (median of {args.n})")
json.dump(out, open("results/time_evidence.json", "w"), indent=1)
