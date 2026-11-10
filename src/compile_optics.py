"""Compile a trained thermal Born LM (born-x8x2:...,t<n>) to passive optics, exactly, and count the light.

The pass is Reading Why's compile (compile_hw.py there) made causal, with this model's readout:
- token fields: h_t = a_t (E w_t + P_t) e^{i omega t}, divided by the largest field so every slot's own field
  (a_t taken out) has norm <= 1; s is the running factor, h_compiled = s h_trained.
- every fixed matrix is divided by its largest singular value (a contraction: a mesh plus attenuators); what it
  lost comes back as a passive combiner ratio or as a classical gain on a measured number.
- attention: h is split four ways (Q, K, V, residual: 1/2 each in field). The weights |<q_i|k_j>|^p / den are
  measured overlaps turned into programmed attenuators; they do not change when q, k are scaled. The reference key
  is the prefix mean g sum_{t<=i} h_t / sum_{t<=i} a_t, so its value V r_i adds w_ir g / mass_i to every v_t, t <= i:
  y_i = sum_t W_it v_t with one weight matrix W. Built as fan-out of each v_t to L outputs (1/sqrt L), attenuators
  W_it as one output port of a passive network (sum_t |W_it|^2 <= 1 after the rows are scaled by their largest
  2-norm, at least 1): y keeps 1 / (sqrt L max_i ||W_i||). Residual combiner
  alpha h / 2 + beta O y, |alpha|^2 + |beta|^2 = 1.
- MLP: h split three ways (W, V, residual: 1/sqrt 3); the m bilinear forms Re[(W_j h) conj(V_j h)] are balanced
  homodyne readings, summed and rescaled as numbers. A middle block's update U mid is fresh light: a modulator
  writes mid (each slot's own field <= 1) into the mesh U; residual combiner as above.
- last block: its MLP output u (`wide` real numbers) is measured and divided by its norm (numbers). The controller
  forms psi = sum_k (A_k u) (x) (B_k u) on the n1 x n2 word grid and writes the mask T_v = (1 + g psi_v / n) / M,
  M = max_v |1 + g psi_v / n| (passive: |T| <= 1), on one SLM. A uniform beam (1 / sqrt V per word) passes the mask
  n times (a multipass cell imaging the SLM onto itself), so the field is T_v^n, and the camera's Born readout gives
  P_v = |1 + g psi_v / n|^(2n) / sum_w |.|^(2n): the trained model's P, with nothing post-selected. (n copies
  counted in n-fold coincidence would give the same P but keep about 1e-55 of the events on this model.)

    python compile_optics.py <tag> [--n 64] [--bits 6 8 10 12 16] [--reprep]
    python compile_optics.py <tag> --photons 6 8 10 12 14 16        # log10 source photons per slot

Reports max |P_compiled - P_trained| over every word and position of n test windows, accuracy and perplexity of
both, the light (log10 of the source's share) at the last MLP's homodyne and at the camera, and accuracy and
perplexity when the SLM mask (amplitude only: the camera reads |T|^(2n)) is written with `bits` bits, levels over [0, 1].
"""
import json
import math
import sys

import torch

import tasks
from models import build


def sig(m):
    """Largest singular value, at least 1: what a matrix must be divided by to be a contraction."""
    return torch.linalg.matrix_norm(m, ord=2).max().clamp(min=1)


def vmax(x, amp):
    """Largest slot field per sequence, a_t taken out: [B, 1, 1]."""
    own = torch.where(amp > 0, torch.linalg.vector_norm(x, dim=-1) / amp.clamp(min=1e-30), torch.zeros_like(amp))
    return own.amax(1)[:, None, None]


def balanced(x, y, rate, dims):
    """Re<x|y> read by a balanced pair: counts of |x + y|^2 / 2 and |x - y|^2 / 2 (summed over `dims`), `rate`
    photons per unit of field^2 (broadcast); rate None: the exact value. Large counts by the normal approximation,
    drawn as the exact difference plus noise so two huge counts do not cancel in floating point (Reading Why's
    compile_hw.balanced)."""
    exact = (x.conj() * y).real.sum(dims) if dims is not None else (x.conj() * y).real
    if rate is None:
        return exact
    def inten(z):
        z2 = z.real ** 2 + z.imag ** 2
        return z2.sum(dims) if dims is not None else z2
    rp, rm = rate * inten(x + y) / 2, rate * inten(x - y) / 2
    big = (rp + rm) > 1e4
    diff_big = 2 * rate * exact + (rp + rm).sqrt() * torch.randn_like(rp)
    zero = torch.zeros_like(rp)
    diff = torch.where(big, diff_big, torch.poisson(torch.where(big, zero, rp)) - torch.poisson(torch.where(big, zero, rm)))
    return diff / (2 * rate)


def energy(h, amp):
    """log10 of the light in each slot's own field (|h_t|^2 / a_t^2, the source's share), median over the slots
    present, per sequence [B]."""
    e = (h.real ** 2 + h.imag ** 2).sum(-1) / (amp * amp).clamp(min=1e-30) if h.is_complex() else \
        (h * h).sum(-1) / (amp * amp).clamp(min=1e-30)
    e = torch.where(amp > 0, e, torch.full_like(e, float("nan")))
    return torch.log10(e.clamp(min=1e-38)).nanmedian(1).values


def inner(model, ids, stages=None, reprep=False, photons=None):
    """(the last MLP's output in trained numbers [B, L, wide], log10 of the light at its homodyne per slot [B]).
    Differentiable: `train.py --light` penalises the second value.
    reprep: after every attention and every middle MLP, h is measured (homodyne, Re and Im of each of the d
    components per slot) and prepared afresh by a modulator at full scale; the second value is then the least light
    at any field measurement (those re-preparations and the MLP homodynes), per slot.
    photons: source photons per slot (a slot's own field of norm 1 carries that many); Poisson counts at every
    overlap's balanced pairs (Re, Im; q_i fanned out to L + 1, k_t split between the overlaps and the reference
    combiner and fanned out to L) and at every MLP homodyne channel. Not counted: the camera (P is taken exact:
    in generation one detected photon draws one word), the middle MLP modulator's fresh light and the readout's
    beam (their own sources). No noise on re-preparations (not used with photons)."""
    seen = []

    def remeasure(h, s, name):
        lit = energy(h, amp)                                               # light in the field here
        if stages is not None:
            stages.append((name, lit.median().item()))
        if not reprep:
            return h, s
        seen.append(lit)                                                   # light at this homodyne
        ht = h / s                                                         # the trained numbers, measured
        c1 = vmax(ht, amp)
        return ht / c1, 1 / c1
    c = model._c
    amp = model.amplitudes(ids)                                            # [B, L], a_t (0 = not sent)
    B, L = ids.shape
    present = amp > 0
    dh = model.d // model.heads
    e = c(model.emb)[ids] + c(model.pos)[:L]
    t = torch.arange(L, device=ids.device, dtype=model.omega.dtype)
    ang = t[:, None] * model.omega[None, :]
    h = e * torch.polar(torch.ones_like(ang), ang) * amp[..., None].to(e.dtype)
    c0 = vmax(h, amp)
    h = h / c0                                                             # every slot's own field <= 1
    s = 1 / c0                                                             # h_compiled = s h_trained, [B, 1, 1]
    if stages is not None:
        stages.append(("token fields", energy(h, amp).median().item()))
    causal = torch.tril(torch.ones(L, L, device=ids.device, dtype=amp.dtype))
    mass = amp.cumsum(1)                                                   # sum_{t<=i} a_t
    for blk, (qkvo, wv) in enumerate(zip(model.qkvo, model.wv)):
        ga, gm = model.res_gain[blk] if model.res_gain is not None else (1.0, 1.0)
        m = c(qkvo)
        cm = [mi / sig(mi) for mi in m]
        hs = h / 2                                                         # four-way split
        q, k, v = (torch.einsum("btd,ed->bte", hs, cm[i]).unflatten(-1, (model.heads, dh)) for i in range(3))
        # measured overlaps; the reference key from the prefix mean (a combiner over the prefix, its scale free:
        # only ratios of overlaps are used)
        if photons is None:
            ov = torch.einsum("bihe,bjhe->bhij", q.conj(), k).abs() * causal
        else:
            rate = photons / (amp * amp).clamp(min=1e-30).amax(1)[:, None, None, None]   # per unit field^2
            # light budget: q_i fans out to L keys and the reference (L + 1), then Re / Im (1/sqrt 2); k_t is split
            # between the overlaps and the reference combiner (1/sqrt 2), fans out to L, then Re / Im (1/sqrt 2)
            fq, fk = 1 / math.sqrt(2 * (L + 1)), 1 / math.sqrt(4 * L)
            qi, kj = q.permute(0, 2, 1, 3)[:, :, :, None] * fq, k.permute(0, 2, 1, 3)[:, :, None] * fk
            re = balanced(qi, kj, rate, -1) / (fq * fk)
            im = balanced(qi, kj * 1j, rate, -1) / (fq * fk)
            pm = (present[:, None, :, None] & present[:, None, None, :]).to(re.dtype)   # no slot, no light
            ov = (re * re + im * im).sqrt() * causal * pm
        if stages is not None:                                             # light at an overlap's balanced pair
            eq, ek = (torch.log10((x.real ** 2 + x.imag ** 2).sum(-1).mean(-1) / (amp * amp).clamp(min=1e-30)
                                  ).masked_fill(~present, float("nan")).nanmedian() for x in (q, k))
            pair = torch.log10(10 ** eq / (2 * (L + 1)) + 10 ** ek / (4 * L))   # the budget above
            stages.append((f"block {blk} q, k per head (slot) / one overlap pair", f"{eq:.1f} {ek:.1f} / {pair:.1f}"))
        r = model.mean_gain * hs.cumsum(1) / (mass[..., None] + 1e-30).to(h.dtype)
        kr = torch.einsum("btd,ed->bte", r, cm[1]).unflatten(-1, (model.heads, dh))
        if photons is None:
            orr = (q.conj() * kr).sum(-1).abs().transpose(1, 2)            # [B, H, L]
        else:
            # the light that meets q_i: k_t's other half fans out to the L positions (1/sqrt L), position i
            # combines its prefix (1/sqrt i each: passive), then Re / Im (1/sqrt 2); the measured overlap times a
            # classical gain g sqrt(i) / (i a) gives |<q_i|K r_i>|
            idx = torch.arange(1, L + 1, device=ids.device, dtype=amp.dtype)
            kp = torch.einsum("btd,ed->bte", hs.cumsum(1), cm[1]).unflatten(-1, (model.heads, dh)) / idx.sqrt()[:, None, None]
            qf, kf = q * fq, kp * fk
            re = balanced(qf, kf, rate[..., 0], -1) / (fq * fk)
            im = balanced(qf, kf * 1j, rate[..., 0], -1) / (fq * fk)
            gain_r = model.mean_gain * idx.sqrt() / (mass + 1e-30)          # g sqrt(i) / (i a)
            orr = ((re * re + im * im).sqrt() * gain_r[..., None] * present[..., None]).transpose(1, 2)
        top = torch.maximum(ov.amax(-1), orr).clamp(min=1e-30)[..., None]
        sp = (ov / top) ** model.power
        sr = (orr / top[..., 0]) ** model.power
        den = (sp.sum(-1) + sr).clamp(min=1e-30)                           # [B, H, L]
        # one weight matrix: W_it = s_it / den_i + (sr_i / den_i) g / mass_i for every t <= i
        W = (sp + (sr * model.mean_gain / (mass[:, None, :] + 1e-30))[..., None] * causal) / den[..., None]
        W = W * present[:, None, None, :] * present[:, None, :, None]      # no slot, no light
        # each output y_i is one port of a passive network fed by the fanned-out v_t (1/sqrt L each): passive
        # when sum_t |W_it|^2 <= 1, so the rows are scaled by their largest 2-norm (at least 1)
        wn = (W * W).sum(-1).add(1e-30).sqrt().amax((1, 2))[:, None, None].clamp(min=1)   # no NaN grad at 0
        keep_y = 1 / (math.sqrt(L) * wn)
        y = torch.einsum("bhij,bjhe->bihe", (W / wn[..., None]).to(v.dtype), v).flatten(-2) / math.sqrt(L)
        oy = y @ cm[3].T                                                   # = keep_y s / (2 sV sO) O y_trained
        rho = ga * sig(m[2]) * sig(m[3]) / keep_y                          # beta / alpha
        alpha = 1 / torch.sqrt(1 + rho * rho)
        h = alpha * (h / 2) + (rho * alpha).to(h.dtype) * oy
        s = s * alpha / 2
        h, s = remeasure(h, s, f"block {blk} after attention")
        w2 = c(wv)
        hs = h / math.sqrt(3)                                              # three-way split
        # each of the m bilinear forms has its own pair of meshes (as mesh_check.py builds them): the W and V
        # arms are each split m ways (1/sqrt m in field), and every form's W_j, V_j is divided by its own largest
        # singular value; the m readings are summed as numbers
        mix = model.mlp_mix
        f = w2.shape[1] // mix
        hf = hs / math.sqrt(mix)
        rate_m = None if photons is None else photons / (amp * amp).clamp(min=1e-30).amax(1)[:, None, None]
        mid, pair_light = 0, []
        for j in range(mix):
            Wj, Vj = w2[0][j * f:(j + 1) * f], w2[1][j * f:(j + 1) * f]
            sW, sV = sig(Wj), sig(Vj)
            a, b = hf @ (Wj / sW).T, hf @ (Vj / sV).T
            mid = mid + balanced(a, b, rate_m, None) * (3 * mix * sW * sV / (s * s))   # back to trained numbers
            if stages is not None:                                         # light at one balanced pair
                pair_light.append(((a.real ** 2 + a.imag ** 2 + b.real ** 2 + b.imag ** 2)
                                   / (amp * amp).clamp(min=1e-30)[..., None])[present])
        if stages is not None:                                             # median over slots and channels
            pl = torch.log10(torch.cat([p.flatten() for p in pair_light]).clamp(min=1e-38)).median()
            stages.append((f"block {blk} MLP: light at one balanced pair (slot share, median)", f"{pl:.2f}"))
        if reprep and not (model.wide and blk == model.n_blocks - 1):
            seen.append(energy(hs, amp))                                   # light into the homodyne
        if model.wide and blk == model.n_blocks - 1:
            light = energy(hs, amp)                                        # light into the homodyne, per slot
            if stages is not None:
                stages.append(("last MLP homodyne", light.median().item()))
            return mid, torch.stack(seen + [light]).amin(0) if reprep else light
        uu = c(model.u[blk])
        su = sig(uu)
        mmax = vmax(mid, amp).clamp(min=1e-30)
        f = (mid / mmax).to(h.dtype) @ (uu / su).T                         # modulator (<= 1) into the mesh U
        # f = update_trained / (gm su mmax); beta / alpha (no division by gm, which starts at 0)
        ratio = (s / math.sqrt(3)) * gm * su * mmax
        alpha = 1 / torch.sqrt(1 + ratio * ratio)
        h = alpha * hs + (ratio * alpha).to(h.dtype) * f
        s = s * alpha / math.sqrt(3)
        h, s = remeasure(h, s, f"block {blk} after MLP")
    raise ValueError("needs a wide last block")


def compiled(model, ids, bits=None, stages=None, reprep=False, photons=None):
    """(P [B, L, V] by the compiled pass, log10 light at the last homodyne [B], log10 light at the camera [B, L])."""
    mid, light_mid = inner(model, ids, stages, reprep, photons)
    c = model._c
    u = mid / (mid ** 2).sum(-1, keepdim=True).sqrt().clamp(min=1e-300)    # numbers
    A, Bm = c(model.pair_a), c(model.pair_b)
    ra = torch.einsum("blw,kiw->blki", u.to(A.dtype), A)
    rb = torch.einsum("blw,kjw->blkj", u.to(A.dtype), Bm)
    psi = torch.einsum("blki,blkj->blij", ra, rb).flatten(-2)              # [B, L, V], the controller's numbers
    n = model.thermal
    tm = 1 + torch.exp(model.log_gain) * psi / n
    tm = tm / tm.abs().amax(-1, keepdim=True)                              # passive mask, |T| <= 1
    # the camera reads |T|^(2n), so only the modulus is written: an amplitude-only SLM, `bits` levels over [0, 1]
    tm = tm.abs()
    if bits is not None:
        tm = torch.round(tm * (2 ** bits - 1)) / (2 ** bits - 1)
    # n passes: field T^n on a uniform beam 1 / sqrt V; the camera reads |T|^(2n) / V (logs for range)
    li = n * torch.log((tm * tm).clamp(min=1e-300)) - math.log(psi.shape[-1])
    light_cam = torch.logsumexp(li, -1) / math.log(10)
    p = torch.softmax(li, -1)
    return p, light_mid, light_cam


def scores(p, y):
    m = y >= 0
    pt = p.gather(-1, y.clamp(min=0)[..., None])[..., 0]
    acc = float((p.argmax(-1) == y)[m].double().mean())
    ppl = float(torch.exp(-torch.log(pt[m].clamp(min=1e-300)).mean()))
    return acc, ppl


if __name__ == "__main__":
    args = sys.argv[1:]
    tag = args[0]
    n_win = int(args[args.index("--n") + 1]) if "--n" in args else 64
    def values(flag):                                  # the numbers after `flag`, up to the next option
        out = []
        for a in args[args.index(flag) + 1:]:
            if a.startswith("--"):
                break
            out.append(int(a))
        return out
    bit_list = values("--bits") if "--bits" in args \
        else [6, 8, 10, 12, 16]
    run = json.load(open(f"results/{tag}.json"))
    d = tasks.get(run["task"], n_train=10)
    model = build(run["model"], d["vocab"], d["x_test"].shape[1])
    model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
    model = model.double().eval()
    assert model.pair and model.thermal and model.pair_mix == 1 and not model.norm
    reprep = "--reprep" in args
    if "--photons" in args:
        lgs = values("--photons")
        run = json.load(open(f"results/{tag}.json"))
        d = tasks.get(run["task"], n_train=10)
        model = build(run["model"], d["vocab"], d["x_test"].shape[1])
        model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
        model = model.double().eval()
        torch.manual_seed(0)
        out = {}
        for lg in lgs + [None]:
            hits = logp = tot = 0
            with torch.no_grad():
                for i in range(0, n_win, 8):
                    x = torch.tensor(d["x_test"][i:i + 8]); y = torch.tensor(d["y_test"][i:i + 8])
                    pq = compiled(model, x, photons=None if lg is None else 10.0 ** lg)[0]
                    m = y >= 0
                    hits += (pq.argmax(-1)[m] == y[m]).sum().item()
                    logp += torch.log(pq.gather(-1, y.clamp(min=0)[..., None])[..., 0][m].clamp(min=1e-300)).sum().item()
                    tot += int(m.sum())
            out[str(lg)] = dict(acc=hits / tot, ppl=math.exp(-logp / tot))
            print(f"  photons per slot {'none' if lg is None else f'1e{lg}'}: acc {hits / tot:.4f}, "
                  f"ppl {math.exp(-logp / tot):.3f}", flush=True)
        json.dump(dict(tag=tag, windows=n_win, photons=out), open(f"results/{tag}_photons.json", "w"), indent=1)
        sys.exit()
    res = dict(tag=tag, windows=n_win, reprep=reprep, bits={})
    diffs, ref_p, comp_p, ys, lm, lc = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, n_win, 8):
            x = torch.tensor(d["x_test"][i:i + 8]); y = torch.tensor(d["y_test"][i:i + 8])
            pr = model.probs(x)
            pc, lmid, lcam = compiled(model, x, reprep=reprep)
            m = y >= 0
            diffs.append(float((pc - pr).abs()[m].max()))
            ref_p.append(pr.gather(-1, y.clamp(min=0)[..., None])[..., 0]); comp_p.append(pc)
            ys.append(y); lm.append(lmid); lc.append(lcam[m])
            for bt in bit_list:
                pq = compiled(model, x, bits=bt, reprep=reprep)[0]
                res["bits"].setdefault(bt, []).append((pq.argmax(-1)[m] == y[m]).double().sum().item())
                res["bits"].setdefault(f"{bt}_logp", []).append(
                    torch.log(pq.gather(-1, y.clamp(min=0)[..., None])[..., 0][m].clamp(min=1e-300)).sum().item())
    y = torch.cat(ys)
    m = y >= 0
    pc = torch.cat(comp_p)
    acc_c, ppl_c = scores(pc, y)
    pr_t = torch.cat(ref_p)
    ppl_r = float(torch.exp(-torch.log(pr_t[m]).mean()))
    lm, lc = torch.cat(lm), torch.cat(lc)
    res.update(max_abs_diff=max(diffs), acc=acc_c, ppl=ppl_c, ppl_trained=ppl_r,
               light_homodyne_log10=dict(median=float(lm.median()), min=float(lm.min())),
               light_camera_log10=dict(median=float(lc.median()), p10=float(lc.quantile(.1)), p90=float(lc.quantile(.9))))
    tot = int(m.sum())
    print(f"{tag}: {n_win} windows, {tot} predictions; max |P_compiled - P_trained| {max(diffs):.1e}")
    print(f"  compiled: acc {acc_c:.4f}, ppl {ppl_c:.3f} (trained ppl {ppl_r:.3f})")
    print(f"  light at the {'least-lit field measurement' if reprep else 'last homodyne'}, per slot: 10^{res['light_homodyne_log10']['median']:.1f} (median), "
          f"10^{res['light_homodyne_log10']['min']:.1f} (worst)")
    print(f"  light at the camera after {model.thermal} passes: 10^{res['light_camera_log10']['median']:.2f} "
          f"(10% 10^{res['light_camera_log10']['p10']:.2f}, 90% 10^{res['light_camera_log10']['p90']:.2f})")
    out = {}
    for bt in bit_list:
        acc = sum(res["bits"][bt]) / tot
        ppl = math.exp(-sum(res["bits"][f"{bt}_logp"]) / tot)
        out[bt] = dict(acc=acc, ppl=ppl)
    res["bits"] = out
    print("  SLM bits: " + "  ".join(f"{bt}: {v['acc']:.4f}/{v['ppl']:.2f}" for bt, v in out.items()))
    json.dump(res, open(f"results/{tag}_optics{'_reprep' if reprep else ''}.json", "w"), indent=1)
