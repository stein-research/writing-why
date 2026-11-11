"""The n-pass readout read as n post-selected rounds on qubits (referee round 2026-11-11, point M1).

One 13-qubit register (V = 8,192 cells) starts in the uniform superposition. Each round applies the diagonal
block encoding of the mask t_w = |1 + g psi_w / n| / max_v |1 + g psi_v / n| (an ancilla rotated by
2 arccos t_w, controlled on |w>, kept in |0>). After n rounds the register holds sum_w t_w^n |w> up to a norm,
so a measurement gives P_n(w) = t_w^{2n} / sum_v t_v^{2n}, and the n post-selections all succeed with
probability  S_n = sum_w t_w^{2n} / V.  This script computes S_n on the trained 16-pass model:

  (a) the paper's family: n rounds of the n-mask (t built with the same n), n = 1, 2, 4, 8, 16;
  (b) Reinhardt & Hauser's knob: L rounds of the fixed 16-mask, L = 1, 2, 4, 8, 16 (scores z_w = 2 log t_w,
      softmax_w(L z_w) = t_w^{2L}: their Corollary 17);
  (c) for comparison, the paper's copies picture: n copies of the state with Born probability q_w = t_w^2 / sum t^2
      all agree with probability sum_w q_w^n (the 10^-55 figure of Section 2);
  (d) the mean mask transmission (sum t^2 / V), which separates (a) from (c): S_n / agree_n = (sum t^2)^n / V;
  (e) the top-16 cells of one position, for the 4-qubit toy of npass_toy.py.

The first 64 validation windows (as the optics checks of Section 4.4), 4 at a time, CPU, float64.
Run from the repository root:  python src/npass_quantum.py   (about 4 GB of memory)
Writes results/npass_quantum.json.
"""
import json, math, os, sys
import torch
torch.set_num_threads(4)
import tasks
from models import build
from compile_optics import inner

TAG = "words_born-therm16p2_s0"
WINDOWS, CHUNK = int(sys.argv[sys.argv.index("--windows") + 1]) if "--windows" in sys.argv else 64, 4
NS = [1, 2, 4, 8, 16]
run = json.load(open(f"results/{TAG}.json")); d = tasks.get(run["task"], n_train=10)
model = build(run["model"], d["vocab"], d["x_test"].shape[1])
model.load_state_dict(torch.load(f"results/{TAG}.pt", map_location="cpu")); model = model.double().eval()
LOG10 = math.log(10)


def zs(x):
    with torch.no_grad():
        mid, _ = inner(model, x)
        u = mid / (mid ** 2).sum(-1, keepdim=True).sqrt()
        A, B = model._c(model.pair_a), model._c(model.pair_b)
        psi = torch.einsum("blki,blkj->blij", torch.einsum("blw,kiw->blki", u.to(A.dtype), A),
                           torch.einsum("blw,kjw->blkj", u.to(A.dtype), B)).flatten(-2)
        return (torch.exp(model.log_gain) * psi).flatten(0, 1), model.probs(x).flatten(0, 1)


def mask(z, n):
    """t_w = |1 + z_w/n| / max_v |1 + z_v/n|, in [0, 1], the diagonal each round applies."""
    t = (1 + z / n).abs()
    return t / t.amax(-1, keepdim=True)


def qs(v):
    q = torch.quantile(v, torch.tensor([0.1, 0.5, 0.9], dtype=v.dtype))
    return dict(median=q[1].item(), p10=q[0].item(), p90=q[2].item(), min=v.min().item(), max=v.max().item())


acc = {n: dict(log_S=[], log_agree=[], mean_t2=[], d1=0.0, did=0.0, dP=0.0) for n in NS}
accL = {L: dict(log_S=[], tv=[]) for L in NS}
toy = {}
for c0 in range(0, WINDOWS, CHUNK):
    z, ptrained = zs(torch.tensor(d["x_test"][c0:c0 + CHUNK]))
    V = z.shape[-1]
    assert V == 8192 and model.thermal == 16
    for n in NS:
        t = mask(z, n)
        log_t2 = 2 * torch.log(t.clamp(min=1e-300))
        log_S = torch.logsumexp(n * log_t2, -1) - math.log(V)
        P = torch.softmax(n * log_t2, -1)
        P_one = torch.softmax(torch.log((t ** n).clamp(min=1e-300) ** 2), -1)
        log_q = log_t2 - torch.logsumexp(log_t2, -1, keepdim=True)
        log_agree = torch.logsumexp(n * log_q, -1)
        a = acc[n]
        a["log_S"].append(log_S); a["log_agree"].append(log_agree); a["mean_t2"].append((t * t).mean(-1))
        a["d1"] = max(a["d1"], (P - P_one).abs().max().item())
        a["did"] = max(a["did"], ((log_S - log_agree) - (n * torch.logsumexp(log_t2, -1) - math.log(V))).abs().max().item())
        if n == 16:
            a["dP"] = max(a["dP"], (P - ptrained).abs().max().item())
    t16 = mask(z, 16)
    log_t2 = 2 * torch.log(t16.clamp(min=1e-300))
    P16 = torch.softmax(16 * log_t2, -1)
    for L in NS:
        accL[L]["log_S"].append(torch.logsumexp(L * log_t2, -1) - math.log(V))
        accL[L]["tv"].append(0.5 * (torch.softmax(L * log_t2, -1) - P16).abs().sum(-1))
    if c0 == 0:                                   # the toy: top-16 cells of two positions of window 0
        for pos in [32, 96]:
            top = torch.topk(t16[pos], 16).indices
            cells = {}
            for n in [1, 2, 4]:
                tt = (1 + z[pos, top] / n).abs(); tt = tt / tt.max()
                cells[n] = tt.tolist()
            toy[f"w0_p{pos}"] = dict(window=0, position=pos, cell_ids=top.tolist(),
                                     words=[d["words"][i] if i < len(d["words"]) else "?" for i in top.tolist()],
                                     z_re=z[pos, top].real.tolist(), z_im=z[pos, top].imag.tolist(), t=cells,
                                     model_mass_of_these_16_at_n16=P16[pos, top].sum().item())
    print(f"windows {c0 + CHUNK}/{WINDOWS}", flush=True)
    del z, ptrained, t, t16, P, P_one, P16

npos = sum(len(v) for v in acc[16]["log_S"])
out = dict(tag=TAG, windows=WINDOWS, positions=npos, V=8192, qubits=13,
           paper_light_camera_log10_64_windows=json.load(open(f"results/{TAG}_optics.json")).get("light_camera_log10"),
           rounds_of_n_mask={}, rounds_of_16_mask={}, toy_cells=toy)
for n in NS:
    a = acc[n]
    log_S, log_agree, mt2 = torch.cat(a["log_S"]), torch.cat(a["log_agree"]), torch.cat(a["mean_t2"])
    med = 10 ** (log_S.median().item() / LOG10)
    rec = dict(n=n, success_log10=qs(log_S / LOG10), success_median=med, copies_agree_log10=qs(log_agree / LOG10),
               mean_t2=qs(mt2), repeats_plain_median=1 / med, repeats_amplified_median=math.pi / 4 / math.sqrt(med),
               one_round_tn_max_diff=a["d1"], identity_d_max_err=a["did"])
    if n == 16:
        rec["P16_vs_model_probs_max_diff"] = a["dP"]
    out["rounds_of_n_mask"][n] = rec
    print(f"n={n:2d}: S_n median {med:.2e} (log10 10-90 % {rec['success_log10']['p10']:.2f}..{rec['success_log10']['p90']:.2f},"
          f" min {rec['success_log10']['min']:.2f}); copies log10 {rec['copies_agree_log10']['median']:.1f}; mean t^2 {rec['mean_t2']['median']:.3f}")
for L in NS:
    log_S, tv = torch.cat(accL[L]["log_S"]), torch.cat(accL[L]["tv"])
    out["rounds_of_16_mask"][L] = dict(L=L, success_log10=qs(log_S / LOG10), success_median=10 ** (log_S.median().item() / LOG10),
                                       tv_to_P16=qs(tv))
    print(f"L={L:2d}: S median {10 ** (log_S.median().item() / LOG10):.2e}  TV to P16 {tv.median():.3f}")
out["diagonal_cost_V8192"] = dict(controls=13, cnots_per_round=8192, ry_per_round=8192,
                                  note="a uniformly controlled Ry with 13 controls: 2^13 CNOT + 2^13 Ry per round and position")
json.dump(out, open("results/npass_quantum.json", "w"), indent=1)
print("written results/npass_quantum.json")
