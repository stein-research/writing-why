"""Score TinyStories word models on fresh test windows that nothing was chosen on.

The learning rates were chosen on the 2,000 test windows train.py reports (tinywords.get, seed 0), so those windows
are a validation set. This draws --n new windows of 129 tokens from the same official validation file, none of them
sharing a story with the old windows (not only no token position), and scores every run given on them: accuracy and perplexity, Born
models in float64 (as train.evaluate), softmax by its log-softmax. Results to results/fresh_test.json (merged).

    python fresh_test.py words_softmax-w96_s0 words_born-therm16p2_s0 ... [--n 2000]
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import stories
import tinywords
from models import build

p = argparse.ArgumentParser()
p.add_argument("tags", nargs="+")
p.add_argument("--n", type=int, default=2000)
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
L = stories.L

_, test, _ = tinywords._load()
old = np.random.default_rng(0 + 1_000_003).integers(0, len(test) - L, 2000)   # tinywords.get's test windows
# whole stories apart: a story any old window touches is out, and a new window may only touch untouched stories
story = np.concatenate([[0], np.cumsum(test[:-1] == tinywords.EOS)])   # story index of every token (EOS ends one)
bad = np.zeros(story[-1] + 1, bool)
for s0 in old:
    bad[story[s0:s0 + L + 1]] = True
used = bad[story]                                                     # tokens of stories the old windows touch
free = np.convolve(used, np.ones(L + 1, int), "valid") == 0          # start s: test[s:s+L+1] touches no such story
starts = np.flatnonzero(free)
rng = np.random.default_rng(20261110)
new = []
for s in rng.permutation(starts):                                    # greedy, no overlap among the new windows either
    if not used[s:s + L + 1].any():
        new.append(s)
        used[s:s + L + 1] = True
    if len(new) == args.n:
        break
assert len(new) == args.n, f"only {len(new)} fresh windows fit"
wt = test[np.array(new)[:, None] + np.arange(L + 1)]
x_all, y_all = torch.tensor(wt[:, :L]), torch.tensor(wt[:, 1:])
print(f"{args.n} fresh windows from a test stream of {len(test):,} tokens ({story[-1] + 1:,} stories, {bad.sum():,} touched "
      f"by the old windows); no new window shares a story with an old one", flush=True)

path = "results/fresh_test.json"
res = json.load(open(path)) if os.path.exists(path) else {}
for tag in args.tags:
    run = json.load(open(f"results/{tag}.json"))
    model = build(run["model"], tinywords.V, L)
    model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
    born = model.family == "B"
    model = (model.double() if born else model).to(dev).eval()
    step = 8 if born else 32
    hits = nll = n = 0
    with torch.no_grad():
        for i in range(0, args.n, step):
            x, y = x_all[i:i + step].to(dev), y_all[i:i + step].to(dev)
            lp = torch.log(model.probs(x).clamp(min=1e-12)) if born else F.log_softmax(model(x), -1)
            hits += (lp.argmax(-1) == y).sum().item()
            nll += F.nll_loss(lp.reshape(-1, lp.shape[-1]), y.reshape(-1), reduction="sum").item()
            n += y.numel()
    res[tag] = dict(acc=hits / n, ppl=math.exp(nll / n), old_acc=run["test_acc"], old_ppl=run["test_ppl"], windows=args.n)
    print(f"{tag}: fresh acc {100 * hits / n:.2f} % ppl {math.exp(nll / n):.3f}   (old windows {100 * run['test_acc']:.2f} % "
          f"{run['test_ppl']:.3f})", flush=True)
    json.dump(res, open(path, "w"), indent=1)
    del model
    torch.cuda.empty_cache()
