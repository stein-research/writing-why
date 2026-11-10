"""Build every fixed matrix of the compiled optics as Clements meshes of beam splitters, with phase errors.

python mesh_check.py <tag> [--n 32] [--sigma 0 0.005 0.01 0.02 0.05] [--draws 2] [--only qkvo wv u]

As Reading Why's mesh_check.py: each fixed matrix M is s U diag(S) V^H (SVD), U and V^H each a rectangular
Clements mesh of beam splitters (theta, phi) plus output phases, found by William R. Clements' `interferometer`
package (MIT; `square_decomposition`), S attenuators, s the classical gain compile_optics already puts on measured
numbers. The meshes are rebuilt from their phases with Gaussian errors (sigma, radians, on every theta, phi and
output phase) and the model with the rebuilt matrices runs through compile_optics.compiled (noise-free otherwise).

The matrices: Q, K, V, O of each block (32 x 32), each bilinear form's W_j and V_j of each MLP (one pair of meshes
per form: 170 x 32 in block 0, 64 x 32 in the last block; m = 4 forms), the middle MLP's U (32 x 170). The
readout's A and B are not meshes here: the controller computes psi from the measured u and writes the SLM.
Decompositions are cached in results/<tag>_mesh.npz.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from interferometer.main import Interferometer, Beamsplitter, square_decomposition

import tasks
from compile_optics import compiled
from models import build

args = sys.argv[1:]
tag = args[0]
opt = lambda k, dflt: args[args.index(k) + 1] if k in args else dflt
n_win, draws = int(opt("--n", 32)), int(opt("--draws", 2))
def values(flag):
    """The numbers after `flag`, up to the next option."""
    out = []
    for a in args[args.index(flag) + 1:]:
        if a.startswith("--"):
            break
        out.append(float(a))
    return out


sigmas = values("--sigma") if "--sigma" in args else [0, 0.005, 0.01, 0.02, 0.05]
# --only qkvo wv u: phase errors on those matrices alone (the others built exactly)
only = [a for a in args[args.index("--only") + 1:] if not a.startswith("--")] if "--only" in args else None

run_cfg = json.load(open(f"results/{tag}.json"))
d = tasks.get(run_cfg["task"], n_train=10)
model = build(run_cfg["model"], d["vocab"], d["x_test"].shape[1])
model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
model = model.double().eval()
params = dict(model.named_parameters())
mix = model.mlp_mix

# (parameter, index into its leading axes, row slice): one entry per mesh pair
slots = []
for b in range(model.n_blocks):
    slots += [(f"qkvo.{b}", (i,), None) for i in range(4)]
    f = params[f"wv.{b}"].shape[2] // mix
    slots += [(f"wv.{b}", (i,), slice(j * f, (j + 1) * f)) for i in range(2) for j in range(mix)]
    if b < len(model.u):
        slots.append((f"u.{b}", (), None))


def get(name, ix, rows):
    p = params[name]
    M = torch.complex(p[0][ix] if ix else p[0], p[1][ix] if ix else p[1]).detach().numpy()
    return M[rows] if rows is not None else M


def put(name, ix, rows, Mb):
    p = params[name]
    with torch.no_grad():
        for part, val in ((0, Mb.real), (1, Mb.imag)):
            tgt = p[part][ix] if ix else p[part]
            if rows is not None:
                tgt[rows] = torch.tensor(val)
            else:
                tgt.copy_(torch.tensor(val))


def mesh_arrays(I, N):
    bs = np.array([(b.mode1, b.mode2, b.theta, b.phi) for b in I.BS_list], dtype=float).reshape(-1, 4)
    ph = np.zeros(N)
    ph[:len(I.output_phases)] = I.output_phases
    return bs, ph


def rebuild(bs, ph, N, sigma=0.0, rng=None):
    """The mesh's unitary from its phases (the package's convention), with Gaussian phase errors."""
    th, phi, out = bs[:, 2].copy(), bs[:, 3].copy(), ph.copy()
    if sigma:
        th += rng.normal(0, sigma, th.shape)
        phi += rng.normal(0, sigma, phi.shape)
        out += rng.normal(0, sigma, out.shape)
    U = np.eye(N, dtype=complex)
    for (m1, m2), t, p in zip(bs[:, :2].astype(int) - 1, th, phi):
        r1, r2 = U[m1].copy(), U[m2].copy()
        U[m1] = np.exp(1j * p) * np.cos(t) * r1 - np.sin(t) * r2
        U[m2] = np.exp(1j * p) * np.sin(t) * r1 + np.cos(t) * r2
    return np.exp(1j * out)[:, None] * U


# the fast rebuild against the package's own, on a 32-mode mesh with errors
rng = np.random.default_rng(0)
Ur = np.linalg.qr(rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32)))[0]
bs, ph = mesh_arrays(square_decomposition(Ur), 32)
bs_err = bs.copy()
bs_err[:, 2:] += rng.normal(0, 0.05, bs_err[:, 2:].shape)
I = Interferometer()
for m1, m2, t, p in bs_err:
    I.add_BS(Beamsplitter(int(m1), int(m2), t, p))
I.output_phases = list(ph)
check = dict(rebuild_vs_package=float(np.abs(rebuild(bs_err, ph, 32) - I.calculate_transformation()).max()),
             decompose_32=float(np.abs(rebuild(bs, ph, 32) - Ur).max()))
print(f"rebuild = package's calculate_transformation to {check['rebuild_vs_package']:.1e}; "
      f"32-mode decomposition rebuilt to {check['decompose_32']:.1e}", flush=True)

cache = f"results/{tag}_mesh.npz"
store = dict(np.load(cache)) if os.path.exists(cache) else {}
meshes = []
for name, ix, rows in slots:
    M = get(name, ix, rows)
    s = np.linalg.norm(M, 2)
    U, S, Vh = np.linalg.svd(M / s)
    k = len(S)
    key = f"{name}_{'_'.join(map(str, ix))}_{rows.start if rows is not None else 'all'}"
    for side, W in (("U", U), ("V", Vh)):
        if f"{key}_{side}_bs" not in store:
            t0 = time.time()
            store[f"{key}_{side}_bs"], store[f"{key}_{side}_ph"] = mesh_arrays(square_decomposition(W), len(W))
            np.savez(cache, **store)
            print(f"  {key} {side}: {len(W)} modes, {time.time() - t0:.0f} s", flush=True)
    rb = rebuild(store[f"{key}_U_bs"], store[f"{key}_U_ph"], len(U))
    vb = rebuild(store[f"{key}_V_bs"], store[f"{key}_V_ph"], len(Vh))
    exact = float(np.abs(s * (rb[:, :k] * S) @ vb[:k] - M).max() / np.abs(M).max())
    meshes.append((name, ix, rows, M, s, S, k, key, len(U), len(Vh), exact))
print(f"{len(meshes)} matrices, {sum(len(store[f'{m[7]}_{sd}_bs']) for m in meshes for sd in 'UV')} beam splitters; "
      f"rebuilt without errors to {max(m[-1] for m in meshes):.1e} (relative)", flush=True)

xs = [torch.tensor(d["x_test"][i:i + 8]) for i in range(0, n_win, 8)]
ys = [torch.tensor(d["y_test"][i:i + 8]) for i in range(0, n_win, 8)]


def evaluate():
    hits = logp = tot = 0
    tops = []
    with torch.no_grad():
        for x, y in zip(xs, ys):
            p = compiled(model, x)[0]
            m = y >= 0
            hits += (p.argmax(-1)[m] == y[m]).sum().item()
            logp += torch.log(p.gather(-1, y.clamp(min=0)[..., None])[..., 0][m].clamp(min=1e-300)).sum().item()
            tot += int(m.sum())
            tops.append(p.argmax(-1)[m])
    return hits / tot, float(np.exp(-logp / tot)), torch.cat(tops)


acc0, ppl0, top0 = evaluate()
print(f"compiled, trained matrices: acc {acc0:.4f}, ppl {ppl0:.3f}", flush=True)
rows_out = []
for sigma in sigmas:
    for draw in range(draws if sigma else 1):
        rng = np.random.default_rng(1000 + draw)
        err = 0.0
        for name, ix, rows, M, s, S, k, key, nu, nv, _ in meshes:
            sg = sigma if only is None or name.split(".")[0] in only else 0.0
            Ub = rebuild(store[f"{key}_U_bs"], store[f"{key}_U_ph"], nu, sg, rng)
            Vb = rebuild(store[f"{key}_V_bs"], store[f"{key}_V_ph"], nv, sg, rng)
            Mb = s * (Ub[:, :k] * S) @ Vb[:k]
            err = max(err, float(np.abs(Mb - M).max() / np.abs(M).max()))
            put(name, ix, rows, Mb)
        acc, ppl, top = evaluate()
        rows_out.append(dict(sigma=sigma, draw=draw, matrix_err_max=err, acc=acc, ppl=ppl,
                             top_kept=float((top == top0).double().mean())))
        print(f"sigma {sigma:.3f} draw {draw}: matrices off by {err:.1e} (relative), acc {acc:.4f}, ppl {ppl:.3f}, "
              f"top word kept {rows_out[-1]['top_kept']:.4f}", flush=True)
for name, ix, rows, M, *_ in meshes:                                      # the trained matrices back
    put(name, ix, rows, M)
with open(f"results/{tag}_mesh{'_' + '_'.join(only) if only else ''}.json", "w") as f:
    json.dump(dict(tag=tag, windows=n_win, check=check, acc=acc0, ppl=ppl0, rows=rows_out,
                   beam_splitters=int(sum(len(store[f"{m[7]}_{sd}_bs"]) for m in meshes for sd in "UV"))), f, indent=1)
