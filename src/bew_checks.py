"""Checks of BEW asked by a referee (2026-11-10), at the 300 copy positions of word_evidence.py:
(1) node doubling: each share with D/2 and with D Gauss-Legendre nodes (exact rule => equal to roundoff), at the
    first --double positions; (2) F = P - 1/V and F_q at every position, saved (the text quotes their medians).

    python bew_checks.py words_born-therm16p2_s0 [--double 10]     # writes results/<tag>_bew_checks.json
"""
import json, sys
import numpy as np, torch
import evidence as E
tag = sys.argv[1]
nd = int(sys.argv[sys.argv.index("--double") + 1]) if "--double" in sys.argv else 10
we = json.load(open(f"results/{tag}_word_evidence.json"))
rows = we["rows"]
sys.argv = [sys.argv[0], tag]                       # word_evidence's loader reads its own args; load directly instead
import tasks
from models import build
run = json.load(open(f"results/{tag}.json")); d = tasks.get(run["task"], n_train=10)
model = build(run["model"], d["vocab"], d["x_test"].shape[1])
model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
dev = "cuda"; model = model.double().to(dev).eval()
xt = d["x_test"]
nodes = model.degree() // 2
out = []
for n, rw in enumerate(rows):
    r, i = rw["window"], rw["i"]
    ids = torch.tensor(xt[r:r + 1, :i + 1], device=dev); pos = torch.tensor([i], device=dev)
    tgt = torch.tensor([[rw["born_pred"]]], device=dev)
    ev, full, fq = E.born(model, ids, pos, tgt, n_nodes=nodes)
    rec = dict(window=r, i=i, F=float(full[0, 0]), F_q=float(fq[0, 0]), gap=float(abs(ev[0, 0].sum() - (full - fq)[0, 0])),
               check_vs_word_evidence=float(np.abs(ev[0, 0, :i + 1].double().cpu().numpy() - np.array(rw["born_evidence"])).max()))
    if n < nd:
        ev2, _, _ = E.born(model, ids, pos, tgt, n_nodes=2 * nodes)
        a, b = ev[0, 0, :i].double().cpu().numpy(), ev2[0, 0, :i].double().cpu().numpy()
        rec.update(double_max_abs=float(np.abs(a - b).max()), double_max_rel=float(np.abs(a - b).max() / np.abs(a).max()))
    out.append(rec)
    if n < nd or (n + 1) % 50 == 0:
        print(n, {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
F = np.array([o["F"] for o in out]); Fq = np.array([o["F_q"] for o in out])
summ = dict(nodes=nodes, positions=len(out), Fq_median=float(np.median(Fq)), Fq_max=float(np.abs(Fq).max()),
            Fq_over_F_median=float(np.median(np.abs(Fq / F))), Fq_over_F_max=float(np.abs(Fq / F).max()),
            gap_max=max(o["gap"] for o in out),
            double_max_abs=max(o.get("double_max_abs", 0) for o in out), double_max_rel=max(o.get("double_max_rel", 0) for o in out))
print(summ)
json.dump(dict(tag=tag, summary=summ, rows=out), open(f"results/{tag}_bew_checks.json", "w"), indent=1)
