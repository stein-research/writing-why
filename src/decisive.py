"""Model-grounded checks of the word rankings at the copy positions of Table 3 (referee round 5).

The copy positions name two 'deciding words' from the text (the earlier key j and the copied word j+1). This script
asks the model itself, deleting words with every normaliser recomputed (the real model):
  decisive    how often deleting the copied word, the earlier key, or the previous word changes the prediction
              (all positions, and the positions whose prediction is the copy); the copied word's rank by the drop;
  agreement   per position, the Spearman correlation of each ranking with the leave-one-out drop of the predicted
              word's probability, median over positions;
  recency     two positional baselines: delete the previous word (flip@1) and delete the five earliest (bottom-5).
Rankings: the model's main evidence (BEW; integrated gradients for the softmax model, 4,096 nodes), gradient x input,
attention (last block, mean of heads), integrated gradients of the real P (Born models, 512 nodes, as held_path.py),
and recency (the later word first). Bootstrap 95 % intervals over positions as word_evidence.py. Float64.

    python src/decisive.py           Writes results/decisive.json (minutes on a laptop GPU).
"""
import json

import numpy as np
import torch
from scipy.stats import spearmanr

import evidence as E
import tasks
from models import Frozen, build

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = tasks.get("tinywords", n_train=10)
xt = d["x_test"].astype(np.int64)
MODELS = {"16 passes": ("words_born-therm16p2_s0", "results/words_born-therm16p2_s0_word_evidence.json", "born"),
          "plain readout": ("words_born-p2r16_s0", "results/words_born-p2r16_s0_word_evidence.json", "born"),
          "softmax": ("words_softmax-w96_lr0.02w_s0",
                      "results/words_born-p2r16_s0_word_evidence_softmax_lr0.02w_n4096.json", "softmax")}


def ci(v):
    v = np.asarray(v, float)
    m = v[np.random.default_rng(0).integers(0, len(v), (2000, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def load(tag):
    run = json.load(open(f"results/{tag}.json"))
    m = build(run["model"], d["vocab"], xt.shape[1])
    m.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
    return m.double().to(dev).eval()


def soft_logits_at(model, x, at, keep=None):
    L = x.shape[1]
    allowed = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device)).expand(x.shape[0], L, L)
    if keep is not None:
        x = x * keep[..., None].to(x.dtype)
        allowed = (allowed & keep[:, None, :]) | torch.eye(L, dtype=torch.bool, device=x.device)
    h = x
    for b in model.blocks:
        h, _ = b(h, allowed)
    return model.head(model.ln(h[:, at]))


def probs_at(model, ids, keep, i):
    """The real model's distribution at i for a batch of deletions keep [n, i+1] (every normaliser recomputed)."""
    idr = ids.expand(keep.shape[0], -1)
    if model.family == "B":
        num, z = model.numerators(idr, model.amplitudes(idr) * keep.to(torch.float64), at=torch.tensor([i], device=dev))
        return num[:, 0] / (z[:, 0, None] + 1e-300) + model.share
    return torch.softmax(soft_logits_at(model, model.embed(idr), torch.tensor([i], device=dev), keep)[:, 0], -1)


def soft_gxi(model, ids, pos, target):
    x0 = model.embed(ids).detach().requires_grad_(True)
    p = torch.softmax(soft_logits_at(model, x0, pos), -1).gather(-1, target[..., None])[..., 0]
    (g,) = torch.autograd.grad(p.sum(), x0)
    return (x0 * g).sum(-1).detach()[0]


out = {}
for label, (tag, evf, key) in MODELS.items():
    rows = json.load(open(evf))["rows"]
    model = load(tag)
    R = {k: [] for k in ("copied", "key", "prev", "early5", "copied_pred", "key_pred", "copied_rank")}
    sp = {}
    for rw in rows:
        r, i, j = rw["window"], rw["i"], rw["j"]
        ids = torch.tensor(xt[r][None, :i + 1], device=dev)
        pos = torch.tensor([i], device=dev)
        with torch.no_grad():
            p0 = probs_at(model, ids, torch.ones(1, i + 1, dtype=torch.bool, device=dev), i)[0]
            tgt = int(p0.argmax())
            assert tgt == rw[f"{key}_pred"], (tag, tgt, rw[f"{key}_pred"])
            P = probs_at(model, ids, ~torch.eye(i + 1, dtype=torch.bool, device=dev)[:i], i)
            flips = (P.argmax(-1) != tgt).cpu().numpy()
            drop = (p0[tgt] - P[:, tgt]).cpu().numpy()
            keep = torch.ones(1, i + 1, dtype=torch.bool, device=dev); keep[0, :5] = False
            R["early5"].append(bool(probs_at(model, ids, keep, i)[0].argmax() != tgt))
        R["copied"].append(bool(flips[j + 1])); R["key"].append(bool(flips[j])); R["prev"].append(bool(flips[i - 1]))
        R["copied_rank"].append(int((drop > drop[j + 1]).sum()) + 1)
        if tgt == rw["y"]:
            R["copied_pred"].append(bool(flips[j + 1])); R["key_pred"].append(bool(flips[j]))
        tgt2 = torch.tensor([[tgt]], device=dev)
        rank = {"evidence": np.asarray(rw[f"{key}_evidence"][:i]), "recency": np.arange(i, dtype=float)}
        if key == "born":
            rank["gradient x input"] = E.grad_input(model, ids, pos, tgt2)[0, 0, :i].detach().cpu().numpy()
            f = lambda x: E.born_scores(model, ids.repeat(x.shape[0], 1), pos, tgt2.repeat(x.shape[0], 1), x, Frozen())
            ig = E._ig(f, model.amplitudes(ids).detach(), pos, 512, k=64)
            while ig.dim() > 3:
                ig = ig.sum(-1)
            rank["IG of the real P"] = ig[0, 0, :i].double().cpu().numpy()
        else:
            rank["gradient x input"] = soft_gxi(model, ids, pos, tgt2)[:i].cpu().numpy()
        rank["attention"] = E.attention(model, ids, pos)[0, 0, :i].double().cpu().numpy()
        for name, s in rank.items():
            sp.setdefault(name, []).append(spearmanr(s, drop).statistic)
    o = dict(tag=tag, positions=len(rows), positions_predicting_copy=len(R["copied_pred"]),
             flip_delete_copied=ci(R["copied"]), flip_delete_key=ci(R["key"]),
             flip_delete_copied_on_copy_predictions=ci(R["copied_pred"]),
             flip_delete_key_on_copy_predictions=ci(R["key_pred"]),
             copied_word_rank_by_drop_median=float(np.median(R["copied_rank"])),
             recency_flip1_previous_word=ci(R["prev"]), recency_bottom5_five_earliest=ci(R["early5"]),
             spearman_with_leave_one_out_median={k: float(np.nanmedian(v)) for k, v in sp.items()})
    out[label] = o
    print(label, json.dumps(o), flush=True)
json.dump(out, open("results/decisive.json", "w"), indent=1)
