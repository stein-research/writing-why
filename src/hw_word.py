"""One written word on a quantum computer: the attention overlaps of the 16-pass model read by Hadamard tests on qubits,
the rest of the pass exact (on the computer), as Reading Why's hw_attention.py does for a sentence pair.

python hw_word.py [--prompt "Lily went to the park . At the"] [--shots 2000] [--cal 32]
python hw_word.py ... --offline [--backend FakeKingston]      # every circuit on the noise model of a device snapshot
python hw_word.py ... --device ibm_kingston --account 1 [--submit] [--keys ibmq_keys.txt]   # your own IBM key file, outside the repository
#   through the ledger (ibm_jobs/); without --submit a dry run: transpiled and recorded, not sent, the pass goes on
#   with ideal-simulator counts (--noisy-dry: the noise model's)
python hw_word.py ... --plan 30     # 30 Monte Carlo draws of the device as Reading Why's runs measured it (no circuit
#   simulated): visibility 0.77-0.82 per block, readout offset +0.004 to +0.012, a row's own factor 1 + 0.035 N(0,1),
#   a circuit's own error 0.0165 N(0,1), binomial shots (Reading Why's hw_plan.py, fitted to its ibm_kingston jobs)
python hw_word.py ... --delete 4 --no-evidence     # the deletion test: word 4 sent with amplitude 0, the answer only

The model explains the word after the prompt (position i = L - 1). Each head is 16 complex modes, 4 qubits; with
the ancilla a Hadamard test of (|0>|q> + |1>|k>) / sqrt 2 on 5 qubits reads P(0) - P(1) = Re<q|k> (Im with -i on the
|1> branch); the norms |q| |k| are put back classically. Block 1: every row i' <= i (block 2 needs every earlier
position), each overlap <q_i'|k_j>, j <= i', and the reference key <q_i'|k_ref,i'>; block 2: row i only (the last
block's other rows do not reach the prediction at i). Each block is one job, block 2's circuits built from block 1's
measured weights. Calibration (as Reading Why's device runs): `--cal` circuits per block as deep as the overlaps,
Hadamard tests of random state pairs with known Re<a|b>; one slope (the visibility) and one readout offset per block
are fitted, the offset taken off every reading and the measured row sums divided by the slope squared (p = 2).

Compared with the exact pass: the next word and its probability (the device's attention weights, everything else
exact), and the word evidence with the device's row sums: BEW's numerators exact, the held attention row sums the
measured ones (models.Frozen scaled per row, HwFrozen below). Results to results/hw_word_<backend>_<shots>[_dry].json.
"""
import json
import math
import zlib
import pathlib
import sys

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence as E
import tinywords
from models import Frozen, build

args = sys.argv[1:]
opt = lambda k, v: args[args.index(k) + 1] if k in args else v
tag = opt("--tag", "words_born-therm16p2_s0")
prompt, shots, ncal = opt("--prompt", "Lily went to the park . At the"), int(opt("--shots", 2000)), int(opt("--cal", 32))
bname = opt("--backend", "FakeKingston")

run = json.load(open(f"results/{tag}.json"))
words, index = tinywords.vocab()
word = lambda t: words[int(t) - tinywords.C0] if int(t) >= tinywords.C0 else "<unk>"
model = build(run["model"], 8192, 128)
model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
model = model.double().eval()
toks = tinywords.tokens(prompt)
ids = torch.tensor([[index.get(t, tinywords.UNK) for t in toks]])
dpos = int(opt("--delete", -1))
if dpos >= 0:                                  # deleted in place: amplitude 0 (models.amplitudes), positions kept
    ids[0, dpos] = 0
L, H, last = ids.shape[1], model.heads, model.n_blocks - 1
backend = getattr(fake_provider, bname)()
noisy_sim, ideal_sim = AerSimulator.from_backend(backend), AerSimulator()


class Device:
    """The circuits on an IBM device, one SamplerV2 job per block, through Seeing Why's hardware ledger
    (src/ibm_jobs.py: a job is never paid for twice, a lost wait is fetched by its id, the quota is
    checked before sending, every job recorded), kept in this project's ibm_jobs/."""

    def __init__(self, backend_name, keys, account, submit, purpose):
        import ibm_jobs                               # src/ibm_jobs.py; the ledger is ibm_jobs/ at the repository root
        # the quota check's cost per circuit: this ledger has no done job yet (default 2 s); Reading Why's ibm_kingston
        # jobs of the same 5-qubit Hadamard tests at 2,000 shots used 0.537-0.543 s a circuit (21 jobs), so 0.6
        ibm_jobs.seconds_per_circuit = lambda name, default=2.0: 0.6
        # account: one keys-file line for both blocks, or one per block ("0,6": block 1 on line 0, block 2 on line 6)
        self.ij, self.backend_name, self.keys = ibm_jobs, backend_name, keys
        self.accounts = [int(x) for x in str(account).split(",")]
        self.submit, self.purpose, self.folders, self.measured = submit, purpose, [], 0
        self.reused, self.sources_by_block = [], []

    def run(self, circs, notes, block):
        # a circuit already measured in another job of this ledger (the same states: e.g. a word's overlap with itself
        # in block 1, which no other word has reached yet) is not sent again; its counts are taken from that job
        old = [self.ij.lookup(c, self.backend_name, shots) for c in circs]
        new = [n for n, o in enumerate(old) if o is None]
        seen = self.ij.measured(self.backend_name, shots)        # where each reused circuit was measured
        self.sources = [None if o is None else str(seen[self.ij.circuit_key(c, self.backend_name, shots)][0])
                        for c, o in zip(circs, old)]
        self.sources_by_block.append(list(self.sources))
        if not new:                                   # every circuit already measured (a replay of a run): no job
            self.measured += 1
            self.folders.append("(every circuit's counts from the ledger)")
            return old
        self.reused.append(dict(block=block + 1, circuits=[n for n, o in enumerate(old) if o is not None],
                                notes=[notes[n] for n, o in enumerate(old) if o is not None]))
        counts, folder = self.ij.sampler_counts([circs[n] for n in new], self.backend_name, shots, self.keys,
                                                self.accounts[min(block, len(self.accounts) - 1)],
                                                f"{self.purpose}, block {block + 1}"
                                                + (f" ({len(circs) - len(new)} circuits' counts from earlier jobs)"
                                                   if len(new) < len(circs) else ""), inputs=[f"results/{tag}.pt"],
                                                circuit_notes=[notes[n] for n in new], dry_run=not self.submit)
        self.folders.append(str(folder))
        if counts is not None:
            self.measured += 1
            merged = list(old)
            for n, c in zip(new, counts):
                merged[n] = c
            return merged
        sim = noisy_sim if "--noisy-dry" in args else ideal_sim
        t = transpile(circs, backend if sim is noisy_sim else sim, optimization_level=3, seed_transpiler=0)
        res = sim.run(t, shots=shots, seed_simulator=block).result()
        return [res.get_counts(n) for n in range(len(circs))]


def ledger_calibration(folder):
    """(slope, offset) fitted to the random calibration circuits of one ledger job, from its notes and counts."""
    rec = json.loads((pathlib.Path(folder) / "record.json").read_text())
    counts = json.loads((pathlib.Path(folder) / "counts.json").read_text())
    pts = [(float(n.split("ideal ")[1]), (c.get("0", 0) - c.get("1", 0)) / shots)
           for n, c in zip([r["note"] for r in rec["circuits"]], counts) if "random calibration" in n]
    x, y = np.array(pts).T
    return tuple(np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0])


class Replay:
    """Shot-noise resampling of a device run: the readings of each block as the device gave them (recorded by a first
    replay from the ledger, nothing sent), each redrawn as binomial counts of 2,000 shots around its measured value.
    Block 2 keeps its measured circuits' order, so its readings are resampled where they were measured."""

    def __init__(self, recorded, seed, resample=True):
        self.recorded, self.rng, self.resample = recorded, np.random.default_rng([11, seed]), resample

    def readings(self, block):
        p = self.recorded[block]
        if not self.resample:
            return p.copy()
        return 2 * self.rng.binomial(shots, np.clip((1 + p) / 2, 0, 1)) / shots - 1


class Plan:
    """A draw of the device as Reading Why's ibm_kingston runs measured it (see the module docstring)."""

    def __init__(self, seed):
        self.rng = np.random.default_rng([7, seed])

    def readings(self, ideal, rowkey, block):
        rng = self.rng
        eta, off = rng.uniform(0.77, 0.82), rng.uniform(0.004, 0.012)
        fac = {k: 1 + 0.035 * rng.standard_normal() for k in set(rowkey)}
        m = np.array([eta * fac[r] * x + off + 0.0165 * rng.standard_normal() for x, r in zip(ideal, rowkey)])
        p0 = np.clip((1 + m) / 2, 0, 1)
        return 2 * rng.binomial(shots, p0) / shots - 1


def circuit(q, k, imag):
    psi = np.concatenate([q, (-1j if imag else 1) * k]) / np.sqrt(2)          # ancilla = top qubit
    qc = QuantumCircuit(5, 1)
    qc.append(StatePreparation(psi), range(5))
    qc.h(4)
    qc.measure(4, 0)
    return qc


def calibration_pairs(block, e):
    # each run its own calibration states (the ledger refuses a circuit already sent): the first run (park) was drawn
    # with [2611, block], its circuits and ideal overlaps are in the ledger; every other prompt or deletion adds its own
    key = [2611, block] if (prompt == "Lily went to the park . At the" and dpos < 0) else \
        [2611, block, zlib.crc32(prompt.encode()), dpos + 1]
    rng = np.random.default_rng(key)
    out = []
    for _ in range(ncal):
        a, b = rng.standard_normal((2, 2, e))
        a = (a[0] + 1j * a[1]) / np.linalg.norm(a[0] + 1j * a[1])
        r = b[0] + 1j * b[1]
        r = r - np.vdot(a, r) * a
        r /= np.linalg.norm(r)
        th = np.arccos(rng.uniform(-1, 1))                                  # Re<a|b> spread over [-1, 1]
        bb = np.cos(th) * a + np.sin(th) * r
        out.append((a, bb))
    return out


stats = dict(circuits=[], cz=[])
recorded = []                                   # each block's readings P0 - P1, as measured (for --bootstrap)
calib_rows = []                                 # each block's fitted (slope, offset)
rows_ratio, calib = [], []


def make_hook(sim):
    def hook(blk, q, k, kr):
        qn, kn, rn = q[0].numpy(), k[0].numpy(), kr[0].numpy()               # [L, H, e]
        ov = torch.einsum("ihe,jhe->hij", q[0].conj(), k[0])[None]           # exact, [1, H, L, L]
        orr = (q[0].conj() * kr[0]).sum(-1).T[None]                          # exact, [1, H, L]
        rows = range(L) if blk < last else [L - 1]
        circs, where, notes, ideal, rowkey = [], [], [], [], []
        for h in range(H):
            for i in rows:
                for j in list(range(i + 1)) + ["ref"]:
                    kv = rn[i, h] if j == "ref" else kn[j, h]
                    nq, nk = np.linalg.norm(qn[i, h]), np.linalg.norm(kv)
                    if nq == 0 or nk == 0:                                   # a deleted word: nothing to read
                        continue
                    for part in (0, 1):
                        circs.append(circuit(qn[i, h] / nq, kv / nk, part == 1))
                        where.append((h, i, j, part, nq * nk))
                        notes.append(f"block {blk + 1} head {h} q{i} k{j} {('Re', 'Im')[part]}")
                        o = np.vdot(qn[i, h] / nq, kv / nk)
                        ideal.append(o.imag if part else o.real)
                        rowkey.append((h, i))
        for n, (a, b) in enumerate(calibration_pairs(blk, qn.shape[-1])):
            circs.append(circuit(a, b, False))
            where.append((0, n, n, 3, float(np.vdot(a, b).real)))
            notes.append(f"block {blk + 1} random calibration {n} ideal {np.vdot(a, b).real:+.6f}")
            ideal.append(float(np.vdot(a, b).real))
            rowkey.append(("cal", n))
        if isinstance(sim, Plan):
            p = sim.readings(ideal, rowkey, blk)
        elif isinstance(sim, Replay):
            p = sim.readings(blk)
        else:
            circs = transpile(circs, basis_gates=["u", "cx"], optimization_level=0)   # the ledger's form
        if isinstance(sim, (Plan, Replay)):
            pass
        elif isinstance(sim, Device):
            counts = sim.run(circs, notes, blk)
        else:
            t = transpile(circs, backend if sim is noisy_sim else sim, optimization_level=3, seed_transpiler=0)
            stats["circuits"].append(len(t))
            stats["cz"] += [c.count_ops().get("cz", 0) for c in t]
            res = sim.run(t, shots=shots, seed_simulator=blk).result()
            counts = [res.get_counts(n) for n in range(len(circs))]
        if not isinstance(sim, (Plan, Replay)):
            p = np.array([(c.get("0", 0) - c.get("1", 0)) / shots for c in counts])
        cal = np.array([(w[4], p[n]) for n, w in enumerate(where) if w[3] == 3])
        slope, off = np.linalg.lstsq(np.vstack([cal[:, 0], np.ones(len(cal))]).T, cal[:, 1], rcond=None)[0]
        calib_rows.append((float(slope), float(off)))
        calib.append(dict(block=blk + 1, slope=float(slope), offset=float(off), n=len(cal),
                          resid_sd=float(np.std(cal[:, 1] - slope * cal[:, 0] - off))))
        # readings reused from an earlier job were taken at that job's visibility; they are used as measured (block 2's
        # circuits were built from them so), and each source job's own calibration is recorded beside this one's
        srcs = getattr(sim, "sources", None) if isinstance(sim, Device) else None
        if srcs:
            fits = {f: ledger_calibration(f) for f in set(srcs) if f}
            calib[-1]["reused_from"] = {pathlib.Path(f).name: dict(slope=float(v[0]), offset=float(v[1]),
                                                                   circuits=srcs.count(f)) for f, v in fits.items()}
        if isinstance(sim, Device):
            recorded.append(p.copy())                 # as measured, reused readings on this job's scale
        ov_m, orr_m = ov.clone(), orr.clone()
        re = {}
        for n, (h, i, j, part, scale) in enumerate(where):
            if part < 3:
                re[(h, i, j, part)] = (p[n] - off) * scale
        for (h, i, j, part), v in re.items():
            if part == 0:
                val = complex(v, re[(h, i, j, 1)])
                if j == "ref":
                    orr_m[0, h, i] = val
                else:
                    ov_m[0, h, i, j] = val
        # the measured row sums over the exact ones (each reading carries the visibility once: divide by slope^2)
        causal = torch.tril(torch.ones(L, L, dtype=torch.float64))
        den_ex = ((ov.abs() ** 2) * causal).sum(-1) + orr.abs() ** 2
        den_hw = ((ov_m.abs() ** 2) * causal).sum(-1) + orr_m.abs() ** 2
        ratio = torch.ones(H, L, dtype=torch.float64)
        for i in rows:
            if (den_ex[0, :, i] > 0).all():
                ratio[:, i] = den_hw[0, :, i] / slope ** 2 / den_ex[0, :, i]
        rows_ratio.append(ratio)
        return ov_m, orr_m
    return hook


class HwFrozen(Frozen):
    """Frozen whose recorded attention row sums ([B, H, L] per block) are scaled to the measured ones."""

    def __init__(self, ratios):
        super().__init__()
        self.ratios, self.blk = ratios, 0

    def __call__(self, d):
        # per block models.hidden holds two [B, H, L] values: first each row's largest overlap c (it cancels within
        # the row and stays exact), then the row sum; only the row sum is scaled
        if not self.replay and d.dim() == 3 and d.shape[1] == H and d.shape[-1] == L and L > 1:
            self.seen = getattr(self, "seen", 0) + 1
            if self.seen % 2 == 0:
                d = d * self.ratios[self.blk][None].to(d.dtype)
                self.blk += 1
        return super().__call__(d)


def hw_born(target, ratios):
    """evidence.born at the last position with the held row sums of HwFrozen: BEW [L], F, F(baseline)."""
    pos = torch.tensor([L - 1])
    a0 = model.amplitudes(ids).detach()
    with torch.no_grad():
        full = E.born_scores(model, ids, pos, target, a0, HwFrozen(ratios))
    fz = HwFrozen(ratios)
    with torch.no_grad():
        E.born_scores(model, ids, pos, target, a0, fz)

    def f(x):
        r = x.shape[0] // ids.shape[0]
        fz.start_replay().k = r
        return E.born_scores(model, ids.repeat(r, 1), pos, target.repeat(r, 1), x, fz)
    return E._ig(f, a0, pos, math.ceil(model.degree() / 2), k=16), full, E._base(f, a0, pos)


with torch.no_grad():
    p_ex = model.probs(ids)[0, -1]
target = p_ex.argmax().view(1, 1)
top = lambda p: [(word(t), float(p[t])) for t in torch.topk(p, 3).indices]
slug = "_".join(t for t in toks if t.isalpha())[:40] + (f"_del{dpos}" if dpos >= 0 else "")
want_ev = "--no-evidence" not in args
if want_ev:
    ev_ex, f_ex, b_ex = E.born(model, ids, torch.tensor([L - 1]), target, k=16)
    a = ev_ex[0, 0, :L - 1].numpy()


def one_pass(sim):
    """The pass with the overlaps read by `sim`: (P at the explained position, evidence with its row sums or None)."""
    rows_ratio.clear(); calib.clear(); recorded.clear(); calib_rows.clear()
    model.overlap_hook = make_hook(sim)
    with torch.no_grad():
        p = model.probs(ids)[0, -1]
    model.overlap_hook = None
    ev = None
    if want_ev:
        e, f, bb = hw_born(target, rows_ratio)
        ev = (e[0, 0, :L - 1].numpy(), float(f), float(bb))
    return p, ev


if "--plan" in args:                         # Monte Carlo of the device: what would a run show?
    n = int(opt("--plan", 30))
    draws = []
    for k in range(n):
        p, ev = one_pass(Plan(k))
        d = dict(same_word=bool(p.argmax() == p_ex.argmax()), p_word=float(p[target]), top=top(p)[0][0],
                 calibration=[dict(c) for c in calib])
        if ev is not None:
            d.update(evidence_corr=float(np.corrcoef(a, ev[0])[0, 1]), evidence=ev[0].tolist())
        draws.append(d)
        print(f"draw {k}: {d['top']} P {d['p_word']:.3f}" + (f"  evidence corr {d['evidence_corr']:.4f}" if ev is not None else ""), flush=True)
    cors = [d["evidence_corr"] for d in draws if "evidence_corr" in d]
    out = dict(tag=tag, prompt=prompt, words=toks, deleted=dpos, shots=shots, calibration_circuits=ncal, draws=n,
               exact=dict(top=top(p_ex)), same_word_rate=float(np.mean([d["same_word"] for d in draws])),
               p_word_median=float(np.median([d["p_word"] for d in draws])),
               p_word_range=[float(min(d["p_word"] for d in draws)), float(max(d["p_word"] for d in draws))],
               evidence_corr_median=float(np.median(cors)) if cors else None,
               evidence_corr_min=float(min(cors)) if cors else None,
               evidence_corr_p10=float(np.quantile(cors, 0.1)) if cors else None, rows=draws)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    with open(f"results/hw_word_plan_{slug}_{shots}.json", "w") as f:
        json.dump(out, f, indent=1)
    sys.exit()

if "--nulls" in args:                       # referee M2/M6: what the evidence correlation can and cannot tell
    # Replays the device run from the ledger (nothing sent) and recomputes BEW with the device's row sums altered:
    # (a) without the visibility correction (each block's ratios times its slope squared), (b) the measured ratios
    # permuted among the rows of each block (20 draws); also the spread of the measured/exact row-sum ratios.
    dev = Device(opt("--device", "ibm_kingston"), opt("--keys", "ibmq_keys.txt"), opt("--account", "1"),
                 False, "replay")
    p_run, ev_run = one_pass(dev)
    ratios = [r.clone() for r in rows_ratio]
    slopes = [c[0] for c in calib_rows]
    corr = lambda e: float(np.corrcoef(a, e[0][0, 0, :L - 1].numpy())[0, 1])
    no_vis = [r * sl ** 2 for r, sl in zip(ratios, slopes)]
    r_novis = corr(hw_born(target, no_vis))
    rng = np.random.default_rng(5)
    r_perm = []
    for k in range(int(opt("--draws", 20))):
        perm = []
        for blk, r in enumerate(ratios):
            rows = list(range(L)) if blk < last else [L - 1]
            v = r.clone()
            flat = torch.stack([r[h, i] for h in range(H) for i in rows])
            flat = flat[torch.tensor(rng.permutation(len(flat)))]
            for n_, (h, i) in enumerate([(h, i) for h in range(H) for i in rows]):
                v[h, i] = flat[n_]
            perm.append(v)
        r_perm.append(corr(hw_born(target, perm)))
    vals = [ratios[0].flatten(), ratios[-1][:, L - 1]]            # block 1: every row; block 2: the explained row
    spread = {f"block {b + 1}": dict(median=float(v.median()), min=float(v.min()), max=float(v.max()))
              for b, v in enumerate(vals)}
    out = dict(tag=tag, prompt=prompt, device_r=float(np.corrcoef(a, ev_run[0])[0, 1]), no_visibility_correction_r=r_novis,
               permuted_rows_r=dict(median=float(np.median(r_perm)), min=float(min(r_perm)), max=float(max(r_perm)),
                                    draws=len(r_perm)), row_ratio_spread=spread, slopes=slopes)
    print(json.dumps(out, indent=1))
    with open(f"results/hw_word_nulls_{slug}.json", "w") as f:
        json.dump(out, f, indent=1)
    sys.exit()

if "--recalibrate" in args:                  # referee M1: readings reused from other jobs put on this job's scale
    # A replay from the ledger (nothing sent) gives each block's readings as the run used them. Block 1's readings that
    # came from other jobs were taken at those jobs' visibility: map each with its source job's own calibration onto
    # this job's, p' = (p - offset_src) / slope_src * slope_now + offset_now, and run the pass again. Block 2's
    # circuits were built from the as-run block 1, so block 2 keeps its measured readings in their order (its states
    # differ slightly from the recalibrated pass's: a sensitivity check, not a new measurement).
    dev = Device(opt("--device", "ibm_kingston"), opt("--keys", "ibmq_keys.txt"), opt("--account", "1"),
                 False, "replay")
    p_run, _ = one_pass(dev)
    if dev.measured != model.n_blocks:
        sys.exit("--recalibrate: this run is not in the ledger")
    rec = [r.copy() for r in recorded]
    mapped = []
    for blk, (pb, srcs) in enumerate(zip(rec, dev.sources_by_block)):
        cal_now = calib_rows[blk]
        pb = pb.copy()
        for n, f in enumerate(srcs):
            if f:
                s0, o0 = ledger_calibration(f)
                pb[n] = (pb[n] - o0) / s0 * cal_now[0] + cal_now[1]
        mapped.append(pb if blk == 0 else rec[blk])
    p_re, _ = one_pass(Replay(mapped, 0, resample=False))
    out = dict(tag=tag, prompt=prompt, deleted=dpos, as_run=top(p_run), recalibrated=top(p_re),
               p_word_as_run=float(p_run[target]), p_word_recalibrated=float(p_re[target]),
               kite_or_deleted_word=None)
    if dpos >= 0:
        wid = int(tinywords.vocab()[1].get(toks[dpos], 0))
        rank = lambda p: int((p > p[wid]).sum()) + 1
        out["kite_or_deleted_word"] = dict(word=toks[dpos], exact=float(p_ex[wid]), exact_rank=rank(p_ex),
                                           as_run=float(p_run[wid]), as_run_rank=rank(p_run),
                                           recalibrated=float(p_re[wid]), recalibrated_rank=rank(p_re))
    print(json.dumps(out, indent=1))
    with open(f"results/hw_word_recalibrated_{slug}.json", "w") as f:
        json.dump(out, f, indent=1)
    sys.exit()

if "--bootstrap" in args:                    # shot-noise intervals of a finished device run (nothing sent)
    dev = Device(opt("--device", "ibm_kingston"), opt("--keys", "ibmq_keys.txt"), opt("--account", "1"),
                 False, "replay")
    p0, ev0 = one_pass(dev)
    if dev.measured != model.n_blocks:
        sys.exit("--bootstrap: this run is not in the ledger (run it with --device --submit first)")
    rec = [r.copy() for r in recorded]
    n = int(opt("--bootstrap", 100))
    shown = list(dict.fromkeys([int(t) for t in torch.topk(p_ex, 3).indices] + [int(t) for t in torch.topk(p0, 3).indices]))
    draws, probs = [], []
    for k in range(n):
        p, ev = one_pass(Replay(rec, k))
        probs.append([float(p[t]) for t in shown])
        d = dict(p_word=float(p[target]), top=top(p)[0][0])
        if ev is not None:
            d.update(evidence=ev[0].tolist(), evidence_corr=float(np.corrcoef(a, ev[0])[0, 1]))
        draws.append(d)
        print(f"redraw {k}: {d['top']} {d['p_word']:.3f}" + (f" r {d['evidence_corr']:.4f}" if ev is not None else ""), flush=True)
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    out = dict(tag=tag, prompt=prompt, deleted=dpos, redraws=n, device_P=float(p0[target]),
               p_word_95=q([d["p_word"] for d in draws]),
               same_word_share=float(np.mean([d["top"] == top(p0)[0][0] for d in draws])),
               top_counts={w: sum(d["top"] == w for d in draws) for w in sorted({d["top"] for d in draws})},
               words={word(t): dict(exact=float(p_ex[t]), device=float(p0[t]), device_95=q([r[n_] for r in probs]))
                      for n_, t in enumerate(shown)})
    if ev0 is not None:
        E_ = np.array([d["evidence"] for d in draws])
        out.update(device_evidence=ev0[0].tolist(), evidence_95=[q(E_[:, t]) for t in range(E_.shape[1])],
                   evidence_corr=float(np.corrcoef(a, ev0[0])[0, 1]),
                   evidence_corr_95=q([d["evidence_corr"] for d in draws]),
                   signs_kept_share=float(np.mean([(np.sign(e) == np.sign(ev0[0])).all() for e in E_])))
    print(json.dumps({k: v for k, v in out.items() if k not in ("device_evidence", "evidence_95")}, indent=1))
    with open(f"results/hw_word_bootstrap_{slug}{'_answer' if ev0 is None else ''}.json", "w") as f:
        json.dump(out, f, indent=1)
    sys.exit()

if "--device" in args:
    sim = Device(opt("--device", "ibm_kingston"), opt("--keys", "ibmq_keys.txt"), opt("--account", "1"),
                 "--submit" in args, f"Writing Why: attention of the 16-pass model for the word after \"{prompt}\""
                                     + (f" with word {dpos} deleted" if dpos >= 0 else "") + " by Hadamard tests")
    where_run = sim.backend_name
else:
    sim = noisy_sim if "--offline" in args else ideal_sim
    where_run = bname if "--offline" in args else "ideal"
p_hw, ev = one_pass(sim)
measured = isinstance(sim, Device) and sim.measured == len(rows_ratio)
out = dict(tag=tag, prompt=prompt, words=toks, deleted=dpos, where=where_run, shots=shots,
           calibration_circuits=ncal, measured_on_device=measured, ledger=getattr(sim, "folders", []),
           reused_counts=getattr(sim, "reused", []),
           exact=dict(top=top(p_ex)), run=dict(top=top(p_hw), p_of_exact_word=float(p_hw[target]),
                                               row_ratio=[r.tolist() for r in rows_ratio], calibration=calib),
           same_word=bool(p_hw.argmax() == p_ex.argmax()),
           circuits_per_block=stats["circuits"], cz_median=float(np.median(stats["cz"])) if stats["cz"] else None)
if ev is not None:
    b = ev[0]
    out["exact"].update(evidence=a.tolist(), F=float(f_ex), F_base=float(b_ex))
    out["run"].update(evidence=b.tolist(), F=ev[1], F_base=ev[2], evidence_sum=float(b.sum()))
    out["evidence_corr"] = float(np.corrcoef(a, b)[0, 1])
dry = isinstance(sim, Device) and not measured
print(f"\"{prompt}\"{f' (word {dpos} deleted)' if dpos >= 0 else ''} -> exact {out['exact']['top']}")
print(f"  {where_run}{' (dry run)' if dry else ''}: {out['run']['top']}  same word {out['same_word']}"
      + (f"  evidence corr {out['evidence_corr']:.3f}" if ev is not None else ""))
if ev is not None:
    print("  evidence exact / run: " + ", ".join(f"{w} {x:+.4f}/{y:+.4f}" for w, x, y in zip(toks, a, b)))
print(f"  calibration {calib}  circuits {stats['circuits']}  CZ median {out['cz_median']}")
with open(f"results/hw_word_{slug}_{where_run}_{shots}{'_dry' if dry else ''}.json", "w") as f:
    json.dump(out, f, indent=1)
