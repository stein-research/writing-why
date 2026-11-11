"""Appendix A of the paper: known-answer checks of the n-pass readout on the trained 16-pass model.

P_n(w) = |1+z_w/n|^{2n} / sum_v |1+z_v/n|^{2n}, z = g psi, psi taken from compile_optics.inner (the measured u).
(1) at n = 16 it equals model.probs; (2) it tends to softmax(2 Re z), computed exactly; (3) the first correction
log P_n - log S = -(Re z^2 - <Re z^2>_S)/n + O(1/n^2) is checked: n * gap - predicted falls as 1/n;
(4) the copies picture: n independent copies of the state q_w = |1+z_w/n|^2 / sum_v |1+z_v/n|^2 all land on the
same word with probability sum_w q_w^n (the share a coincidence filter would keep), at n = 16.
Results also to results/npass_check.json.

    python npass_check.py      # first 8 validation windows, CPU, float64, about a minute
"""
import json, math, sys, torch
sys.path.insert(0, ".")
import tasks
from models import build
from compile_optics import inner
tag = "words_born-therm16p2_s0"
run = json.load(open(f"results/{tag}.json")); d = tasks.get(run["task"], n_train=10)
model = build(run["model"], d["vocab"], d["x_test"].shape[1])
model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu")); model = model.double().eval()
x = torch.tensor(d["x_test"][:8])
with torch.no_grad():
    mid, _ = inner(model, x)
    u = mid / (mid ** 2).sum(-1, keepdim=True).sqrt()
    A, B = model._c(model.pair_a), model._c(model.pair_b)
    psi = torch.einsum("blki,blkj->blij", torch.einsum("blw,kiw->blki", u.to(A.dtype), A),
                       torch.einsum("blw,kjw->blkj", u.to(A.dtype), B)).flatten(-2)
    z = torch.exp(model.log_gain) * psi
    ptrained = model.probs(x)
def Pn(n):
    li = n * torch.log((1 + z / n).abs() ** 2)
    return torch.softmax(li, -1), li
S = torch.softmax(2 * z.real, -1)
out = dict(windows=8, max_abs_z=z.abs().max().item(), median_abs_z=z.abs().median().item())
print(f"max |z| = {z.abs().max():.2f}, median |z| = {z.abs().median():.2f}")
print(f"(1) n=16 formula vs model.probs: max abs diff {(Pn(16)[0] - ptrained).abs().max():.1e}")
rz2 = (z * z).real
pred = -(rz2 - (S * rz2).sum(-1, keepdim=True))           # first-order shift of log P, recentred under S
for n in [16, 64, 256, 1024, 4096, 16384]:
    P, li = Pn(n)
    tv = 0.5 * (P - S).abs().sum(-1)
    gap = (torch.log(P) - torch.log(S))                      # per word
    top = S > 1e-3                                           # words that matter
    resid = (n * gap - pred)[top]
    print(f"(2) n={n:6d}: TV median {tv.median():.2e} max {tv.max():.2e}   n*TV median {n*tv.median():.2f}   "
          f"(3) |n*gap - predicted| on words with S>1e-3: median {resid.abs().median():.3f}, "
          f"vs |predicted| median {pred[top].abs().median():.3f}")
    out.setdefault("tv_median", {})[n] = tv.median().item()
lq = torch.log((1 + z / 16).abs() ** 2); lq = lq - torch.logsumexp(lq, -1, keepdim=True)
agree = torch.logsumexp(16 * lq, -1).flatten() / math.log(10)        # log10 P(all 16 copies agree), per position
q = torch.quantile(agree, torch.tensor([0.1, 0.5, 0.9], dtype=agree.dtype))
print(f"(4) n=16 copies agree: log10 probability median {q[1]:.1f} (10-90 %: {q[0]:.1f} to {q[2]:.1f}), "
      f"{agree.numel()} positions")
out.update(n16_diff_max=(Pn(16)[0] - ptrained).abs().max().item(),
           copies_agree_log10=dict(median=q[1].item(), p10=q[0].item(), p90=q[2].item(), positions=agree.numel()))
json.dump(out, open("results/npass_check.json", "w"), indent=1)
