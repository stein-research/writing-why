"""Per-token evidence for next-token prediction, at the scored positions of a batch.

The baseline keeps the current token: for position i the path scales only the context,
x_t(alpha) = alpha x_t for t < i, the token at i (and, by causality without effect, those after it) stays whole. So
e_i(t), t < i, is what each context token adds to the prediction made from the current token alone, and
sum_{t < i} e_i(t) = F_i(x) - F_i(baseline). (With a zero baseline every term of Born's F_i carries the current
token's amplitude, which then took most of the evidence: README, "First round".)

Born (exact): with every denominator held at the input (models.Frozen), F_i = P_{y_i} - b is a polynomial of the
context amplitudes of degree at most D = model.degree() (2 * 18^B for the first models), so ceil(D / 2) Gauss-Legendre nodes give the integrated gradients
e_i(t) = a_t int_0^1 dF_i/da_t(path) d alpha exactly; the baseline value is F_i with the denominators still held.
Softmax: the same integrated gradients of the logit of y_i over the embedded context (summed over dimensions), with a
fixed number of nodes; it is not a polynomial, so its gap is reported.
Also for both: attention (last block, mean of heads), gradient x input and leave-one-out (delete one token: Born sends
amplitude 0, softmax zeroes its field and lets no query read it; the drop of the explained score).
"""
import math

import numpy as np
import torch

from models import Frozen


def nodes01(n):
    t, w = np.polynomial.legendre.leggauss(n)
    return (t + 1) / 2, w / 2


def _paths(x0, pos, alpha):
    """One copy of x0 [B, L, ...] per scored position j, its context t < pos[j] scaled by alpha: [P * B, L, ...],
    position-major, so every position's path goes through the model in one pass."""
    ctx = torch.arange(x0.shape[1], device=x0.device)[None, :] < pos[:, None]                   # [P, L]
    scale = torch.where(ctx, torch.as_tensor(alpha, dtype=x0.dtype, device=x0.device), 1.0).to(x0.dtype)
    x = x0[None] * scale.view((len(pos), 1, x0.shape[1]) + (1,) * (x0.dim() - 2))
    return x.flatten(0, 1), ctx


def _own(out, P):
    """[P * B, P] scores of every copy -> [B, P]: copy j's score at its own position j."""
    return out.unflatten(0, (P, -1)).diagonal(dim1=0, dim2=2)


def _ig(f, x0, pos, n_nodes, k=1):
    """Integrated gradients along the context path of each scored position: [B, P, *x0.shape[1:]], zero at t >= i.
    f(x) -> the explained scores [P * B, P] of x [P * B, L, ...] (the copies of `_paths`); k > 1: k nodes go
    through the model in one pass (f then takes [k * P * B, L, ...] and returns [k * P * B, P])."""
    P = len(pos)
    total = torch.zeros((P,) + x0.shape, dtype=x0.dtype, device=x0.device)
    t_all, w_all = nodes01(n_nodes)
    for c in range(0, n_nodes, k):
        ts, ws = t_all[c:c + k], w_all[c:c + k]
        xs = [_paths(x0, pos, float(t)) for t in ts]
        ctx = xs[0][1]
        x = torch.cat([xi for xi, _ in xs]).requires_grad_(True)
        out = f(x).unflatten(0, (len(ts), -1))
        (g,) = torch.autograd.grad(sum(_own(out[j], P).sum() for j in range(len(ts))), x)
        g = g.unflatten(0, (len(ts), P, -1))
        total += torch.einsum("k,kp...->p...", torch.as_tensor(ws, dtype=g.dtype, device=g.device), g)
    ctx = ctx[:, None].view((P, 1, x0.shape[1]) + (1,) * (x0.dim() - 2))
    return x0[:, None] * (total * ctx).transpose(0, 1)


def _base(f, x0, pos):
    with torch.no_grad():
        return _own(f(_paths(x0, pos, 0.0)[0]), len(pos))


def born_scores(model, ids, pos, target, amp, fz):
    num, z = model.numerators(ids, amp, fz, at=pos)                          # read out at the scored positions only
    f = num / (z[..., None] + 1e-300)
    return f.gather(-1, target[..., None])[..., 0]                    # [B, P]


def born(model, ids, pos, target, n_nodes=None, k=1):
    """BEW [B, P, L], F(x) [B, P], F(baseline) [B, P]."""
    a0 = model.amplitudes(ids).detach()
    n_nodes = n_nodes or math.ceil(model.degree() / 2)
    with torch.no_grad():
        full = born_scores(model, ids, pos, target, a0, Frozen())
    P = len(pos)
    ids_p, tgt_p, fz = ids.repeat(P, 1), target.repeat(P, 1), Frozen()          # the copies' denominators, held
    with torch.no_grad():
        born_scores(model, ids_p, pos, tgt_p, a0.repeat(P, 1), fz)
    def f(x):                                                                # x: k node copies of the batch
        r = x.shape[0] // ids_p.shape[0]
        fz.start_replay().k = r
        return born_scores(model, ids_p.repeat(r, 1), pos, tgt_p.repeat(r, 1), x, fz)
    return _ig(f, a0, pos, n_nodes, k), full, _base(f, a0, pos)


def softmax_scores(model, x, pos, target):
    return model.from_embedded(x)[:, pos].gather(-1, target[..., None])[..., 0]


def softmax_ig(model, ids, pos, target, n_nodes=256):
    x0 = model.embed(ids).detach()
    with torch.no_grad():
        full = softmax_scores(model, x0, pos, target)
    tgt_p = target.repeat(len(pos), 1)
    f = lambda x: softmax_scores(model, x, pos, tgt_p)
    return _ig(f, x0, pos, n_nodes).sum(-1), full, _base(f, x0, pos)


def _grad(f, x0, pos):
    x = x0.clone().requires_grad_(True)
    out = f(x)
    return torch.stack([torch.autograd.grad(out[:, j].sum(), x, retain_graph=j < len(pos) - 1)[0]
                        for j in range(len(pos))], 1)


def grad_input(model, ids, pos, target):
    if model.family == "B":
        a0 = model.amplitudes(ids).detach()
        fz = Frozen()
        with torch.no_grad():
            born_scores(model, ids, pos, target, a0, fz)
        return a0[:, None, :] * _grad(lambda x: born_scores(model, ids, pos, target, x, fz.start_replay()), a0, pos)
    x0 = model.embed(ids).detach()
    return (x0[:, None] * _grad(lambda x: softmax_scores(model, x, pos, target), x0, pos)).sum(-1)


def frozen_at(model, ids):
    """Born: every denominator of the full input, recorded, for deletions that keep them (frozen=fz below).
    Deleting a token then changes the same F that BEW explains; with
    frozen=None the denominators are recomputed (the real model: what holding them costs)."""
    fz = Frozen()
    with torch.no_grad():
        model.numerators(ids, model.amplitudes(ids), fz)
    return fz


def _born_probs(model, ids, keep, frozen):
    amp = model.amplitudes(ids)
    if keep is not None:
        amp = amp * keep.to(amp.dtype)
    if frozen is None:
        return model.probs(ids, amp)
    num, z = model.numerators(ids, amp, frozen.start_replay())
    return num / (z[..., None] + 1e-300) + model.share


def explained(model, ids, pos, target, keep=None, frozen=None):
    """The explained score at the scored positions (Born P - b, denominators recomputed or, with frozen, held at the
    full input; softmax logit)."""
    if model.family == "B":
        return _born_probs(model, ids, keep, frozen)[:, pos].gather(-1, target[..., None])[..., 0] - model.share[0]
    return model(ids, keep)[:, pos].gather(-1, target[..., None])[..., 0]


def predict(model, ids, keep=None, frozen=None):
    if model.family == "B":
        return _born_probs(model, ids, keep, frozen).argmax(-1)
    return model(ids, keep).argmax(-1)


def leave_one_out(model, ids, pos, target, frozen=None):
    """Drop of the explained score when token t is deleted, [B, P, L] (Born: frozen as in explained)."""
    with torch.no_grad():
        base = explained(model, ids, pos, target, frozen=frozen)
        out = torch.zeros(ids.shape[0], len(pos), ids.shape[1], dtype=base.dtype, device=ids.device)
        for t in range(ids.shape[1]):
            keep = torch.ones_like(ids, dtype=torch.bool)
            keep[:, t] = False
            out[:, :, t] = base - explained(model, ids, pos, target, keep, frozen)
    return out


def attention(model, ids, pos):
    with torch.no_grad():
        if model.family == "B":
            return model.attention(ids)[:, pos]
        return model.from_embedded(model.embed(ids), with_attn=True)[1][:, pos]
