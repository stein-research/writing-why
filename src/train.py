"""Train one of the paper's models on TinyStories V2 or WikiText-103 words (tasks.py) and report accuracy and perplexity.

Adam, cosine learning rate (after an optional linear warmup), gradient norm clipped at 1, loss only at the scored positions. Born trains on the
cross-entropy of P_v itself, softmax on that of its logits. Last epoch's numbers, no model selection.
Writes results/<tag>.json and .pt; with --ckpt-every, also results/<tag>_ckpt.json and .pt (the latest, overwritten, for
sampling while it trains: python sample.py <tag>_ckpt).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import compile_optics
import tasks
from models import build, count

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--task", required=True, choices=tasks.TASKS)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--n-train", type=int, default=100_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="")
p.add_argument("--ckpt-every", type=int, default=0, help="every this many steps: test, keep a checkpoint")
p.add_argument("--opt", default="adam", choices=("adam", "cadam"), help="cadam: phase-blind Adam for Born (optim.py)")
p.add_argument("--light", type=float, default=0.0,
               help="Born thermal pair models: add this per decade of light lost inside the compiled optics "
                    "(compile_optics.inner, on the first --light-windows windows of each batch)")
p.add_argument("--light-windows", type=int, default=16)
p.add_argument("--phase-noise", type=float, default=0.0,
               help="Born: train through meshes with this phase error (rad): every fixed matrix e^{iK_L} M e^{iK_R} "
                    "(models.mesh_noise), drawn afresh each step; never at test")
p.add_argument("--init", default="", help="start from results/<init>.pt (a fine-tune)")
p.add_argument("--warmup", type=float, default=0.0, help="share of the steps with the lr rising linearly from 0")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = tasks.get(args.task, n_train=args.n_train)
xtr, ytr = torch.from_numpy(d["x_train"]), torch.from_numpy(d["y_train"])   # tinystories: uint8
xte, yte = torch.tensor(d["x_test"], device=dev), torch.tensor(d["y_test"], device=dev)
model = build(args.model, d["vocab"], xtr.shape[1]).to(dev)
if args.init:
    model.load_state_dict(torch.load(f"results/{args.init}.pt", map_location=dev))
if args.phase_noise:
    model.phase_sigma = args.phase_noise
if args.opt == "cadam":
    from optim import ComplexAdam
    opt = ComplexAdam(model.named_parameters(), lr=args.lr)
else:
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
steps = args.epochs * math.ceil(len(xtr) / args.batch)
warm = int(args.warmup * steps)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) if warm and s < warm else
                                          0.5 * (1 + math.cos(math.pi * min(s - warm, steps - warm) / (steps - warm))))


def log_probs(ids):
    if model.family == "B":
        return torch.log(model.probs(ids).clamp(min=1e-12))
    return F.log_softmax(model(ids), -1)


chunk = max(1, min(500, 40_000_000 // (xtr.shape[1] * d["vocab"])))   # test windows at once: a word vocab's [n, L, V]


def evaluate(x, y):
    """Test accuracy and perplexity. A Born model is read out in float64 here: a plain (not thermal) readout's
    intensities can pass float32's range late in training (inf / inf = NaN) while the log-domain training loss stays
    finite (words_born-p2r16_s0 read NaN from step 6,000; its weights were fine)."""
    model.eval()
    born = model.family == "B"
    if born:
        model.double()
    hits = nll = n = 0
    step = max(1, chunk // 2) if born else chunk
    with torch.no_grad():
        for i in range(0, len(x), step):
            lp, yy = log_probs(x[i:i + step]), y[i:i + step]
            m = yy != -100
            hits += (lp.argmax(-1)[m] == yy[m]).sum().item()
            nll += F.nll_loss(lp[m], yy[m], reduction="sum").item()
            n += m.sum().item()
    if born:
        model.float()
    model.train()
    return hits / n, math.exp(nll / n)


tag = args.tag or f"{args.task}_{args.model}_s{args.seed}"
print(f"{tag}: params {count(model)}, {dev}", flush=True)
t0, hist, marks, step = time.time(), [], [], 0
os.makedirs("results", exist_ok=True)


def light_now():
    """Median log10 light at the last homodyne of the compiled optics, first 64 test windows (thermal pair only)."""
    if not getattr(model, "thermal", 0):
        return None
    with torch.no_grad():
        return float(compile_optics.inner(model, xte[:64])[1].median())


def checkpoint():
    acc, ppl = evaluate(xte, yte)
    marks.append(dict(step=step, epoch=step * args.batch / len(xtr), test_acc=acc, test_ppl=ppl))
    torch.save(model.state_dict(), f"results/{tag}_ckpt.pt")
    with open(f"results/{tag}_ckpt.json", "w") as f:
        json.dump(dict(vars(args), tag=tag, step=step, test_acc=acc, test_ppl=ppl, marks=marks), f, indent=1)
    marks[-1]["light_log10"] = light_now()
    print(f"  step {step} (epoch {marks[-1]['epoch']:.2f}) test acc {acc:.4f} ppl {ppl:.3f}"
          + (f" light 10^{marks[-1]['light_log10']:.1f}" if marks[-1]["light_log10"] is not None else "")
          + f" ({time.time() - t0:.0f} s)", flush=True)


for ep in range(args.epochs):
    perm = torch.randperm(len(xtr))
    run = 0.0
    for i in range(0, len(xtr), args.batch):
        idx = perm[i:i + args.batch]
        ids, y = xtr[idx].to(dev).long(), ytr[idx].to(dev).long()
        if model.family == "B":                     # log I_y - log Z: no [B, L, V] probabilities (word vocab)
            loss = -model.log_prob(ids, y).sum() / (y >= 0).sum()
            if args.light > 0:                      # light is log10 of a share <= 1: maximise it
                light = compile_optics.inner(model, ids[:args.light_windows])[1].mean()
                loss = loss - args.light * light
        else:
            loss = F.nll_loss(log_probs(ids).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run += loss.item() * len(idx)
        step += 1
        if args.ckpt_every and step % args.ckpt_every == 0:
            checkpoint()
    acc, ppl = evaluate(xte, yte)
    hist.append(dict(epoch=ep + 1, loss=run / len(xtr), test_acc=acc, test_ppl=ppl))
    print(f"epoch {ep + 1:3d} loss {run / len(xtr):.4f} test acc {acc:.4f} ppl {ppl:.3f} ({time.time() - t0:.0f} s)",
          flush=True)

extra = {}
if "copy_test" in d:                            # stories: the copy positions on their own
    yc = yte.masked_fill(~torch.tensor(d["copy_test"], device=dev), -100)
    extra["copy_acc"], extra["copy_ppl"] = evaluate(xte, yc)
    print(f"copy positions: acc {extra['copy_acc']:.4f} ppl {extra['copy_ppl']:.3f}")
if args.task.endswith("words"):                 # word tasks: also without the positions whose target is <unk> (id 1)
    yk = yte.masked_fill(yte == 1, -100)
    extra["known_acc"], extra["known_ppl"] = evaluate(xte, yk)
    extra["unk_share"] = float(((yte == 1).sum() / (yte != -100).sum()).item())
    print(f"targets other than <unk> ({1 - extra['unk_share']:.2%} of them): acc {extra['known_acc']:.4f} "
          f"ppl {extra['known_ppl']:.3f}")
res = dict(vars(args), tag=tag, params=count(model), light_log10=light_now(), test_acc=hist[-1]["test_acc"], test_ppl=hist[-1]["test_ppl"],
           seconds=time.time() - t0, history=hist, **extra, **(dict(marks=marks) if marks else {}))
with open(f"results/{tag}.json", "w") as f:
    json.dump(res, f, indent=1)
torch.save(model.state_dict(), f"results/{tag}.pt")
print(f"{tag}: test acc {res['test_acc']:.4f} ppl {res['test_ppl']:.3f}")
