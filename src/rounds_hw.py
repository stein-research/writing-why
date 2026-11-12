"""The n-pass readout as n post-selected rounds, run on qubits (referee round 5, M4: the qubit readout was only a
correspondence, never run).

A 4-qubit register carries 16 words: the 16 most probable cells of one position of the trained 16-pass model
(results/npass_quantum.json, toy_cells w0_p32, from npass_quantum.py), with the masks t_w of the paper's family (the
n-mask t_w = |1 + g psi_w / n| / max_v |1 + g psi_v / n| on these 16 cells). The register starts uniform (H on each
qubit). Round r rotates its own ancilla by Ry(2 arccos t_w), uniformly controlled on the register (a
UCRY: 16 angles), so the ancilla's |0> amplitude is t_w. After n rounds, keeping the shots whose n ancillas all read 0
leaves the register in sum_w t_w^n |w> (normalised), so
    the kept words follow   P_n(w) = t_w^{2n} / sum_v t_v^{2n}     (Eq. npass on these 16 words), and
    the shots are kept with S_n = sum_w t_w^{2n} / 16              (the share of the readout light at the camera).
The ancillas are separate qubits measured at the end (deferred measurement), no mid-circuit measurement.
n = 0 (H and measure) is the readout control: uniform words, every shot kept.

Measures per n: acceptance against S_n; total-variation distance of the kept words to P_n, to the uniform
distribution and to P_{n'} of the other n; the top word's kept share against its ideal share.

    python src/rounds_hw.py                       ideal and FakeKingston-noise simulation (no job); results/rounds_sim.json
    python src/rounds_hw.py --device ibm_kingston [--submit]
                                              through the ledger (ibm_jobs/): without --submit a dry run (transpiled,
                                              recorded, not sent); with --submit one SamplerV2 job; results/rounds_hw.json
    python src/rounds_hw.py --power 20            how often the criteria (judge) pass: ideal and FakeKingston, 20 draws
                                              each; results/rounds_plan.json
Options: --shots 20000 (default), --ns 0,1,2,4; --twirl (Pauli twirling of the two-qubit gates) and --no-dd (no
dynamical decoupling): post-hoc diagnostics of the first run, written to results/rounds_hw_twirl.json / _nodd.json.
"""
import json
import pathlib
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UCRYGate
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider

HERE = pathlib.Path(__file__).resolve().parent
args = sys.argv[1:]
opt = lambda k, d: args[args.index(k) + 1] if k in args else d
SHOTS = int(opt("--shots", "20000"))
NS = [int(x) for x in opt("--ns", "0,1,2,4").split(",")]
toy = json.loads((HERE.parent / "results/npass_quantum.json").read_text())["toy_cells"]["w0_p32"]
T = {int(k): np.asarray(v, float) for k, v in toy["t"].items()}
W = len(toy["words"])
NQ = int(np.log2(W))
assert 2 ** NQ == W == 16


def mask(n):
    return np.ones(W) if n == 0 else T[n]


def ideal(n):
    p = mask(n) ** (2 * n)
    return p / p.sum(), float(p.mean())


def circuit(n):
    """Register qubits 0..3 (word w = sum_b bit_b 2^b), ancillas 4..3+n; clbits 0..3 the word, 4.. the ancillas."""
    qc = QuantumCircuit(NQ + n, NQ + n, name=f"rounds_n{n}")
    qc.h(range(NQ))
    for r in range(n):
        qc.append(UCRYGate(list(2 * np.arccos(np.clip(mask(n), 0, 1)))), [NQ + r] + list(range(NQ)))
    qc.measure(range(NQ + n), range(NQ + n))
    return qc


def analyse(n, counts):
    kept = np.zeros(W)
    total = 0
    for key, c in counts.items():
        key = key.replace(" ", "")
        total += c
        if n == 0 or set(key[:-NQ]) <= {"0"}:
            kept[int(key[-NQ:], 2)] += c
    q = kept / kept.sum()
    p, s = ideal(n)
    tv = lambda a, b: float(0.5 * np.abs(a - b).sum())
    return dict(n=n, shots=total, kept=int(kept.sum()), acceptance=float(kept.sum() / total), S_n=s,
                tv_to_ideal=tv(q, p), tv_to_uniform=tv(q, np.full(W, 1 / W)),
                tv_to_other={m: tv(q, ideal(m)[0]) for m in NS if m != n},
                top_share=float(q[0]), top_share_ideal=float(p[0]), kept_words=q.round(5).tolist())


def judge(rows):
    """The criteria (fixed before any job): for every n >= 1, (a) the kept words are closer in total variation to
    their own P_n than to P_m of every other n in NS (uniform = P_0 included), (b) the acceptance lies within 0.10 of
    S_n; and across n, (c) the acceptance falls and (d) the top word's kept share rises as n grows."""
    by = {r["n"]: r for r in rows}
    ns = [n for n in NS if n >= 1]
    a = {n: all(by[n]["tv_to_ideal"] < v for v in by[n]["tv_to_other"].values()) for n in ns}
    b = {n: abs(by[n]["acceptance"] - by[n]["S_n"]) <= 0.10 for n in ns}
    acc = [by[n]["acceptance"] for n in NS]
    top = [by[n]["top_share"] for n in NS]
    c = all(x > y for x, y in zip(acc, acc[1:]))
    d = all(x < y for x, y in zip(top, top[1:]))
    return dict(closest_to_own=a, acceptance_within_0_10=b, acceptance_falls=c, top_share_rises=d,
                passed=all(a.values()) and all(b.values()) and c and d)


circs = [circuit(n) for n in NS]
if "--power" in args:
    # the plan: how often a run passes, on the FakeKingston noise model and on the ideal simulator, K draws each
    K, back = int(opt("--power", "20")), fake_provider.FakeKingston()
    res_all = {}
    for label, sim, target in (("ideal", AerSimulator(), AerSimulator()),
                               ("FakeKingston", AerSimulator.from_backend(back), back)):
        t = transpile(circs, target, optimization_level=3, seed_transpiler=0)
        verdicts = []
        for k in range(K):
            res = sim.run(t, shots=SHOTS, seed_simulator=100 + k).result()
            verdicts.append(judge([analyse(n, res.get_counts(i)) for i, n in enumerate(NS)]))
        res_all[label] = dict(draws=K, passed=sum(v["passed"] for v in verdicts),
                              closest_to_own={n: sum(v["closest_to_own"][n] for v in verdicts) for n in NS if n},
                              acceptance_within={n: sum(v["acceptance_within_0_10"][n] for v in verdicts) for n in NS if n},
                              acceptance_falls=sum(v["acceptance_falls"] for v in verdicts),
                              top_share_rises=sum(v["top_share_rises"] for v in verdicts))
        print(label, json.dumps(res_all[label]), flush=True)
    (HERE.parent / "results/rounds_plan.json").write_text(json.dumps(dict(shots=SHOTS, ns=NS, **res_all), indent=1))
elif "--device" not in args:
    out = {}
    back = fake_provider.FakeKingston()
    for label, sim, target in (("ideal", AerSimulator(), None), ("FakeKingston", AerSimulator.from_backend(back), back)):
        t = transpile(circs, target or AerSimulator(), optimization_level=3, seed_transpiler=0)
        res = sim.run(t, shots=SHOTS, seed_simulator=1).result()
        rows = [analyse(n, res.get_counts(k)) for k, n in enumerate(NS)]
        if target is not None:
            for r, tc in zip(rows, t):
                ops = tc.count_ops()
                r["transpiled"] = dict(depth=tc.depth(), cz=int(ops.get("cz", 0)), qubits=tc.num_qubits)
        out[label] = rows
        out[label + "_verdict"] = judge(rows)
        print(label, "verdict", json.dumps(out[label + "_verdict"]))
        for r in rows:
            print(f"{label:12s} n={r['n']}  accept {r['acceptance']:.4f} (S_n {r['S_n']:.4f})  TV ideal "
                  f"{r['tv_to_ideal']:.4f} uniform {r['tv_to_uniform']:.4f}  top {r['top_share']:.3f} "
                  f"({r['top_share_ideal']:.3f})  {r.get('transpiled', '')}", flush=True)
    (HERE.parent / "results/rounds_sim.json").write_text(json.dumps(dict(shots=SHOTS, ns=NS, words=toy["words"], **out), indent=1))
else:
    sys.path.insert(0, str(HERE))
    import ibm_jobs                               # src/ibm_jobs.py; the ledger is ibm_jobs/ at the repository root
    counts, folder = ibm_jobs.sampler_counts(
        circs, opt("--device", "ibm_kingston"), SHOTS, opt("--keys", "ibmq_keys.txt"),
        int(opt("--account", "0")), "the n-pass readout as n post-selected rounds, 16 words on 4 qubits, n = "
        + ", ".join(map(str, NS)) + (" (post-hoc diagnostic: gate twirling)" if "--twirl" in args else "")
        + (" (post-hoc diagnostic: no dynamical decoupling)" if "--no-dd" in args else ""), inputs=["results/npass_quantum.json"],
        circuit_notes=[f"n = {n}: H on 4 qubits, {n} uniformly controlled Ry rounds on separate ancillas, measure all"
                       for n in NS], dry_run="--submit" not in args, dd="--no-dd" not in args, twirl="--twirl" in args,
                       repeat=1 if "--twirl" in args else 2 if "--no-dd" in args else 0)
    print("ledger:", folder)
    if counts is not None:
        rows = [analyse(n, c) for n, c in zip(NS, counts)]
        print("verdict", json.dumps(judge(rows)))
        for r in rows:
            print(f"device n={r['n']}  accept {r['acceptance']:.4f} (S_n {r['S_n']:.4f})  TV ideal "
                  f"{r['tv_to_ideal']:.4f} uniform {r['tv_to_uniform']:.4f}  top {r['top_share']:.3f} "
                  f"({r['top_share_ideal']:.3f})", flush=True)
        tagname = "rounds_hw" + ("_twirl" if "--twirl" in args else "") + ("_nodd" if "--no-dd" in args else "")
        (HERE.parent / f"results/{tagname}.json").write_text(json.dumps(dict(shots=SHOTS, ns=NS, words=toy["words"],
                                                                  ledger=str(folder), device=rows,
                                                                  verdict=judge(rows)), indent=1))
