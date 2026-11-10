"""Adam for complex parameters stored as (2, ...) real tensors (re, im), the way models.py keeps Born's.

Plain Adam scales the real and the imaginary part of each weight by its own second moment, so its steps depend on
the weight's phase: rotate every weight by 45 degrees and P does not change, but the steps do. Here the second
moment of a complex weight is |g|^2 / 2 = (g_re^2 + g_im^2) / 2, shared by both parts (halved to keep Adam's scale),
so a phase rotation of a weight rotates its update the same way. Real parameters get plain Adam.
"""
import math

import torch

COMPLEX = ("emb", "pos", "qkvo", "wv", "u", "readout", "readout_b")         # CausalBorn's (2, ...) parameters


class ComplexAdam(torch.optim.Optimizer):
    def __init__(self, named_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        cx, re = [], []
        for name, p in named_params:
            (cx if name.split(".")[0] in COMPLEX and p.shape[0] == 2 else re).append(p)
        super().__init__([dict(params=cx, complex=True), dict(params=re, complex=False)],
                         dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, st = p.grad, self.state[p]
                if not st:
                    st["t"], st["m"] = 0, torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p[0] if group["complex"] else p)
                st["t"] += 1
                st["m"].mul_(b1).add_(g, alpha=1 - b1)
                s = (g[0] ** 2 + g[1] ** 2) / 2 if group["complex"] else g ** 2
                st["v"].mul_(b2).add_(s, alpha=1 - b2)
                mh = st["m"] / (1 - b1 ** st["t"])
                vh = (st["v"] / (1 - b2 ** st["t"])).sqrt().add_(group["eps"])
                p.sub_(group["lr"] * mh / vh)                                   # vh broadcasts over (re, im)
