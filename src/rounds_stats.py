"""Statistics of the rounds runs on ibm_kingston beyond the fixed criteria (referee round 6), from the ledger's raw
counts and transpiled circuits; no job, no login.

Per run (the run under the criteria, ledger 9b53c5922d387e4d; the twirled diagnostic, 1d55487d942cd945) and n:
  chi2_p              chi-square test of the kept words against P_n (a test the criteria did not ask for);
  register_uniform_p  chi-square test that the register, unconditioned on the ancillas, is uniform (it is, ideally:
                      the rounds only act on the ancillas), a model-free sign of an error on the register;
  register_bit0_one   the share of all shots whose lowest register bit reads 1 (ideally 0.5);
  odd_share           the odd words' share among the kept shots, and its ideal value under P_n;
  bootstrap_a         the share of 2,000 multinomial resamples of the kept counts that pass criterion (a).
And the transpiled circuits as sent, run on the ideal simulator and on the FakeKingston noise model (20,000 shots):
acceptance, TV to P_n, TV to the nearest other P_m, top share, odd share.

    python src/rounds_stats.py      Writes results/rounds_stats.json (a minute).
"""
import json

import numpy as np
from qiskit import qpy
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider
from scipy.stats import chisquare

toy = json.load(open("results/npass_quantum.json"))["toy_cells"]["w0_p32"]
T = {int(k): np.array(v) for k, v in toy["t"].items()}
NS = [0, 1, 2, 4]
RUNS = {"run": "9b53c5922d387e4d", "twirled": "1d55487d942cd945"}


def P(n):
    if n == 0:
        return np.full(16, 1 / 16)
    p = T[n] ** (2 * n)
    return p / p.sum()


tv = lambda a, b: float(0.5 * np.abs(a - b).sum())


def split(n, counts):
    kept, reg = np.zeros(16), np.zeros(16)
    for k, v in counts.items():
        k = k.replace(" ", "")
        w = int(k[-4:], 2)
        reg[w] += v
        if n == 0 or set(k[:-4]) <= {"0"}:
            kept[w] += v
    return kept, reg


out = {"device": {}, "sent_circuits_simulated": {}}
rng = np.random.default_rng(1)
for label, key in RUNS.items():
    C = json.load(open(f"ibm_jobs/{key}/counts.json"))
    rows = []
    for n, c in zip(NS, C):
        kept, reg = split(n, c)
        N = int(kept.sum())
        q = kept / N
        boot = sum(all(tv(qb, P(n)) < tv(qb, P(m)) for m in NS if m != n)
                   for qb in (rng.multinomial(N, q) / N for _ in range(2000))) / 2000
        rows.append(dict(n=n, kept=N, chi2_p=float(chisquare(kept, P(n) * N).pvalue),
                         register_uniform_p=float(chisquare(reg, np.full(16, reg.sum() / 16)).pvalue),
                         register_bit0_one=float(reg[1::2].sum() / reg.sum()),
                         odd_share=float(q[1::2].sum()), odd_share_ideal=float(P(n)[1::2].sum()),
                         bootstrap_a=float(boot)))
        print(label, rows[-1], flush=True)
    out["device"][label] = rows
with open(f"ibm_jobs/{RUNS['run']}/transpiled.qpy", "rb") as f:
    circs = qpy.load(f)
for label, sim in (("ideal", AerSimulator()), ("FakeKingston", AerSimulator.from_backend(fake_provider.FakeKingston()))):
    res = sim.run(circs, shots=20000, seed_simulator=7).result()
    rows = []
    for i, n in enumerate(NS):
        kept, _ = split(n, res.get_counts(i))
        q = kept / kept.sum()
        rows.append(dict(n=n, acceptance=float(kept.sum() / 20000), tv_to_own=tv(q, P(n)),
                         tv_to_nearest_other=min(tv(q, P(m)) for m in NS if m != n),
                         top_share=float(q[0]), odd_share=float(q[1::2].sum())))
        print(label, rows[-1], flush=True)
    out["sent_circuits_simulated"][label] = rows
json.dump(out, open("results/rounds_stats.json", "w"), indent=1)
