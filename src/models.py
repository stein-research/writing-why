"""Causal next-token models of one budget (about 100k parameters), each returning per-position scores [B, L, V].

B  causal Born Transformer (Reading Why's Born-p2-ap made causal): token t enters as the complex field
   a_t (E w_t + P_t) e^{i omega t} with a constant amplitude a_t = 1/sqrt(L_max) (0 = token not sent);
   attention w_ij = |<q_i|k_j>|^p / (sum_{j' <= i} |<q_i|k_j'>|^p + |<q_i|k_ref,i>|^p), p even (4 at first, 2 for
   the word models from 2026-11-09), over the keys j <= i and a
   reference key, the prefix mean g sum_{t <= i} h_t / sum_{t <= i} a_t; bilinear MLP Re[(W h) conj(V h)]; Born
   readout per position, phi_i = R h_i, P_v = sum_{b in S_v} |phi_ib|^2 / Z_i over N = 4V outcomes, b_v = 1/V.
S  causal softmax transformer: pre-LayerNorm, softmax attention, GELU MLP, final LayerNorm, linear head per position.

Every denominator of B goes through `Frozen` (recorded on a plain pass, replayed when explaining), so the explained
P_v - b_v of each position is a polynomial of the amplitudes that vanishes at a = 0, of degree `degree()` (2 * 18^B for
the first models; 6,400 for born-x8x2:m4,t16,x1,r16,p2: 2 (2p + 1) per block, x 4 for the pair readout, x n thermal). A constant
amplitude (not 1/sqrt(n)) keeps a position blind to how many tokens follow it: the MLP is not scale-invariant.
"""
import math
import os

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F


class Frozen:
    """Records denominators on the first pass, replays them on later passes."""

    def __init__(self):
        self.values, self.replay, self.i, self.k = [], False, 0, 1

    def __call__(self, d):
        if not self.replay:
            self.values.append(d.detach())
            return d
        v = self.values[self.i]
        self.i += 1
        if self.k > 1:                       # k batched copies of the recorded batch (several path nodes at once)
            v = v.repeat((self.k,) + (1,) * (v.dim() - 1))
        return v

    def start_replay(self):
        self.replay, self.i = True, 0
        return self


def _logp_head(h, rr, ri, target, per_token):
    """log P_target = log I_target - log Z from the last state h [B, L, d] and readout r = rr + i ri [N, d], for
    training. One real matmul gives Re and Im of phi = h @ r.T; h is first divided by its norm, a held per-position
    scale that leaves every P unchanged (so no pass over the [B, L, N] intensities for their maximum). Compiled, the
    squares, sum and gather fuse: half the time of the plain version on a word vocabulary, the same numbers."""
    if h.is_complex():
        hh = torch.cat([h.real, h.imag], -1) / (h.real ** 2 + h.imag ** 2).sum(-1, keepdim=True).sqrt().detach()
        w = torch.cat([torch.cat([rr, -ri], 1), torch.cat([ri, rr], 1)], 0)  # [2N, 2d]: rows give Re, then Im
    else:                                                                    # real amplitudes (a wide readout)
        hh = h / (h ** 2).sum(-1, keepdim=True).sqrt().detach()
        w = torch.cat([rr, ri], 0)
    re, im = (hh @ w.T).chunk(2, -1)
    inten = re * re + im * im
    z = inten.sum(-1)
    it = inten.unflatten(-1, (-1, per_token)).sum(-1).gather(-1, target.clamp(min=0)[..., None])[..., 0]
    return (torch.log(torch.maximum(it, 1e-12 * z)) - torch.log(z)) * (target >= 0)


_logp_head_c = torch.compile(_logp_head, dynamic=False)


def _free(frozen):
    return frozen if frozen is not None else (lambda d: d)


def _gue(n, lead, tau, device, dtype):
    """Hermitian K [*lead, n, n] with E|K_ij|^2 = tau^2 (GUE)."""
    a = torch.randn(*lead, n, n, device=device, dtype=dtype) + 1j * torch.randn(*lead, n, n, device=device, dtype=dtype)
    return (a + a.mH) * (tau / 2)


def mesh_noise(m, sigma):
    """A fixed matrix as built by two meshes with phase errors sigma (rad): e^{iK_L} M e^{iK_R}, K GUE with entry
    scale tau = 1.22 sigma. Calibrated on this model's own Clements meshes (32, 64, 170 modes): their relative
    Frobenius error is 1.22 sigma sqrt(N) at every size, as is ||e^{iK} - I||_F / sqrt(N) = tau sqrt(N). A
    surrogate (errors spread evenly over the matrix); mesh_check.py tests the result on the real meshes."""
    if not sigma:
        return m
    tau = 1.22 * sigma
    lead, (r, c) = m.shape[:-2], m.shape[-2:]
    kl = torch.linalg.matrix_exp(1j * _gue(r, lead, tau, m.device, m.real.dtype))
    kr = torch.linalg.matrix_exp(1j * _gue(c, lead, tau, m.device, m.real.dtype))
    return kl @ m @ kr


def _thermal_lse_eager(ar, ai, br, bi, g, n: int):
    """logsumexp over the cells of n log|1 + g psi / n|^2, psi = a b, for one mixed state, in real arithmetic and
    fused by the compiler (inductor handles complex poorly). The GPU training path of CausalBorn._thermal_logz:
    equal to the complex one to 2e-14 in float64; in float32 its rounding is the original's (median gradient
    error 2.4e-6 against float64 for both); 262 -> 137 ms a step on the p2 main model, peak 3.2 -> 1.9 GB (no
    checkpoint needed: the compiler recomputes what the backward needs)."""
    pr = torch.einsum("blki,blkj->blij", ar, br) - torch.einsum("blki,blkj->blij", ai, bi)
    pi = torch.einsum("blki,blkj->blij", ar, bi) + torch.einsum("blki,blkj->blij", ai, br)
    c = g / n
    zr, zi = 1 + c * pr, c * pi
    return torch.logsumexp((n * torch.log((zr * zr + zi * zi).clamp(min=1e-30))).flatten(-2), -1)


_FUSED = {"fn": None, "checked": set(), "off": os.environ.get("BORN_NO_COMPILE", "") not in ("", "0")}


def _thermal_lse(ar, ai, br, bi, g, n: int):
    """_thermal_lse_eager, compiled (Triton through torch.compile) where it compiles and agrees.
    Any GPU torch.compile supports (Colab's T4 sm_75 included) takes the fused path; where it cannot compile (no
    Triton, CPU, Apple), or BORN_NO_COMPILE=1, or the first call of a new shape does not match the eager code to
    1e-4, it falls back to the eager code, which gives the same numbers, only slower and with more memory."""
    if _FUSED["off"]:
        return _thermal_lse_eager(ar, ai, br, bi, g, n)
    if _FUSED["fn"] is None:
        _FUSED["fn"] = torch.compile(_thermal_lse_eager, dynamic=False)
    key = (tuple(ar.shape), tuple(br.shape), ar.dtype, ar.device.type)
    try:
        out = _FUSED["fn"](ar, ai, br, bi, g, n)
    except Exception as e:                                                   # no Triton, an unsupported GPU, ...
        import warnings
        warnings.warn(f"thermal readout: torch.compile failed ({type(e).__name__}); using the eager code")
        _FUSED["off"] = True
        return _thermal_lse_eager(ar, ai, br, bi, g, n)
    if key not in _FUSED["checked"]:                                         # once per shape: the same numbers?
        with torch.no_grad():
            ref = _thermal_lse_eager(ar.detach(), ai.detach(), br.detach(), bi.detach(), g.detach(), n)
            bad = not torch.allclose(out.detach(), ref, rtol=1e-4, atol=1e-4 * ref.abs().max().item())
        if bad:
            import warnings
            warnings.warn("thermal readout: the compiled code disagrees with the eager code; using the eager code")
            _FUSED["off"] = True
            return _thermal_lse_eager(ar, ai, br, bi, g, n)
        _FUSED["checked"].add(key)
    return out


class CausalBorn(nn.Module):
    family = "B"

    def __init__(self, vocab, max_len, d=32, heads=2, d_ff=170, blocks=2, per_token=4, ref=True, gain0=None, power=4, norm=False, tie=False, wide=0, pair=None, pair_rank=4, pair_mix=1, mlp_mix=1, thermal=0, softlimit=0, linhead=0):
        super().__init__()
        assert d % heads == 0
        self.norm = norm                      # pre-norm, held: each block reads h / rms(h), the rms a held constant
        self.power = power                    # attention weight |<q|k>|^power (even, so a polynomial)
        assert power % 2 == 0
        self.d, self.heads, self.n_blocks, self.max_len, self.ref = d, heads, blocks, max_len, ref
        s = 1 / math.sqrt(2 * d)
        self.emb = nn.Parameter(torch.randn(2, vocab, d))                     # embeddings started at scale 1
        self.pos = nn.Parameter(torch.randn(2, max_len, d))                   # additive position fields
        self.omega = nn.Parameter(torch.rand(d) * 2 * math.pi / max_len)
        self.mean_gain = nn.Parameter(torch.ones(()))
        self.qkvo = nn.ParameterList(nn.Parameter(torch.randn(2, 4, d, d) * s) for _ in range(blocks))
        # wide: the readout measures the last block's MLP output Re[(W h) * conj(V h)] itself, `wide` numbers (about
        # d^2 independent quadratic forms of h), not the d-dimensional h it is folded back into. A measurement in D
        # dimensions names at most about D outcomes with confidence (sum_v max_h P_v <= D), and a word vocabulary
        # needs thousands. The degree is unchanged: that output is already the last block's top-degree term.
        self.wide = wide
        ffs = [d_ff] * (blocks - 1) + [wide or d_ff]
        # mlp_mix = m: each MLP output is a sum of m bilinear forms, mid_k = sum_j Re[(W_j h)_k conj(V_j h)_k] (a mixed
        # measurement in the middle of the block). It stays polynomial, so BEW stays exact.
        self.mlp_mix = mlp_mix
        self.wv = nn.ParameterList(nn.Parameter(torch.randn(2, 2, f * mlp_mix, d) * s) for f in ffs)
        self.u = nn.ParameterList(nn.Parameter(torch.randn(2, d, d_ff) / math.sqrt(2 * d_ff))
                                  for _ in range(blocks - (1 if wide else 0)))
        # Optional gain on each residual branch (attention, MLP), started at gain0 (ReZero-like): a deep model starts
        # close to its embedding and opens its blocks as it learns. A constant when explaining: the degree is unchanged
        self.res_gain = nn.Parameter(torch.full((blocks, 2), float(gain0))) if gain0 is not None else None
        n_out = per_token * vocab
        # tie: the readout measures along the embeddings themselves, r_v = e_v (one outcome per token); for a word
        # vocabulary, where a separate readout would double the parameters
        self.tie = tie
        assert not tie or per_token == 1
        self.readout = None if tie else nn.Parameter(torch.randn(2, n_out, d) * s)
        if wide and not pair:                       # outcome v measures along a_v (x) b_v on the output read as a d x (wide / d) grid
            assert tie and wide % d == 0                                     # a_v = the embedding e_v
            self.readout_b = nn.Parameter(torch.randn(2, vocab, wide // d) / math.sqrt(2 * wide // d))
        # pair = (n1, n2): two registers. The last MLP's output u (`wide` numbers) prepares a = A_k u in C^n1 and
        # b = B_k u in C^n2; the joint state psi = sum_k a_k (x) b_k (Schmidt rank <= pair_rank, entangled for k > 1)
        # lives in n1 * n2 = V dimensions and is measured once over every cell; word v is cell (v // n2, v % n2).
        # A linear readout of u keeps the state inside a `wide`-dimensional subspace (sum_v max P_v <= wide); the
        # product a (x) b, quadratic in u, spans all V dimensions. Rank 1 would be a product state, P = |a_i|^2 |b_j|^2:
        # a row choice times a column choice; entanglement makes it one joint measurement. Degree: 4 x that of u.
        # pair_mix = m: a mixed state, m such states measured together, P_v = sum_j |psi_j(v)|^2 / sum_j ||psi_j||^2
        # (an unmeasured mode, or an incoherent mixture; the Born counterpart of a mixture of softmaxes; locally
        # purified states, Glasser et al. 2019). Weights are the states' own norms. Degree unchanged.
        self.pair, self.pair_mix = pair, pair_mix
        if pair:
            pair_rank = pair_rank * pair_mix                                  # m groups of rank r, flattened
            assert wide and pair[0] * pair[1] == vocab and per_token == 1
            self.pair_a = nn.Parameter(torch.randn(2, pair_rank, pair[0], wide) / math.sqrt(2 * wide))
            self.pair_b = nn.Parameter(torch.randn(2, pair_rank, pair[1], wide) / math.sqrt(2 * wide))
            # thermal = n: each mixed state's n copies, each a reference beam (amplitude 1 on every word) plus the
            # state g psi / n, counted only when all n land on the same word (an n-fold coincidence):
            # I_v = sum_j |1 + g psi_j(v) / n|^(2n), P_v = I_v / sum_w I_w. As n grows this tends to
            # exp(2 g Re psi_j(v)), softmax's Gibbs weights, but for every finite n it is a Born probability of a
            # pure n-copy state, polynomial in u (degree x n), so BEW stays exact. g = exp(log_gain), one number.
            # u is divided by its norm (held when explaining), so the reference has a fixed size against it.
            self.thermal = thermal
            # softlimit (ablation): the n -> infinity readout exp(2 g Re psi) trained directly - a
            # softmax over the same psi, no longer polynomial (no exact BEW). Needs thermal > 0 to take these paths.
            self.softlimit = bool(softlimit)
            assert not softlimit or thermal, "softlimit needs t<n> (any n) for the thermal code paths"
            if thermal:
                self.log_gain = nn.Parameter(torch.tensor(math.log(math.sqrt(vocab))))
        # linhead (the paper's control, Table 2): the same Born body, its last block's measured numbers u read by an
        # ordinary linear softmax head over the words. Not a Born readout, no exact evidence; pair_a / pair_b unused.
        self.linhead = bool(linhead)
        if linhead:
            assert wide, "linhead reads the wide last block's u"
            self.lin = nn.Linear(wide, vocab)
        self.per_token = per_token                                           # S_v = outcomes v*k .. v*k + k - 1
        self.register_buffer("share", torch.full((vocab,), 1 / vocab))       # b_v = 1 / V
        # The [V, N] group matrix it replaces (N^2 for a word vocabulary) is still in older checkpoints
        self._register_load_state_dict_pre_hook(lambda sd, prefix, *a: sd.pop(prefix + "groups", None))

    @staticmethod
    def _c(p):
        return torch.complex(p[0], p[1])

    def _prenorm(self, h, fz):
        """h / rms(h) per position (an RMS norm with the rms held when explaining, so the degree is unchanged)."""
        if not self.norm:
            return h
        return h / fz((h.real ** 2 + h.imag ** 2).mean(-1, keepdim=True).sqrt().clamp(min=1e-30)).to(h.dtype)

    def degree(self):
        deg = 1
        for _ in range(self.n_blocks):
            deg = 2 * ((2 * self.power + 1) * deg)   # attention, then the bilinear MLP
        return (4 if self.pair else 2) * max(1, getattr(self, "thermal", 0)) * deg     # Born readout (of a state quadratic in u for a pair)

    def amplitudes(self, ids):
        return (ids != 0).to(self.emb.dtype) / math.sqrt(self.max_len)

    def numerators(self, ids, amp, frozen=None, scores=None, at=None):
        """(N, Z) per position: N_v = sum_{b in S_v} |phi_b|^2 - b_v Z, [B, L, V]; Z [B, L]; P_v - b_v = N_v / Z.
        at: only those positions [P] are read out ([B, P, V]): the readout is most of the cost for a word vocabulary."""
        inten, fz = self.intensities(ids, amp, frozen, scores, at)
        z = inten.sum(-1)
        num = inten.unflatten(-1, (-1, self.per_token)).sum(-1) - self.share * z[..., None]
        return num, fz(z)

    def registers(self, u):
        """(a [B, L, r, n1], b [B, L, r, n2]) from the last MLP output u [B, L, wide] (real)."""
        A, B = self._c(self.pair_a), self._c(self.pair_b)
        uc = u.to(A.dtype)
        return torch.einsum("blw,kiw->blki", uc, A), torch.einsum("blw,kjw->blkj", uc, B)

    def _pair_psi(self, a, b):
        m = self.pair_mix
        return torch.einsum("blgki,blgkj->blgij", a.unflatten(2, (m, -1)), b.unflatten(2, (m, -1))).flatten(-2)

    def _thermal_logi(self, psi):
        """log I = n log|1 + g psi / n|^2 for psi [..., V] (a log only for range; I itself is a polynomial)."""
        if getattr(self, "softlimit", False):
            return 2 * torch.exp(self.log_gain) * psi.real
        n = self.thermal
        z = 1 + torch.exp(self.log_gain) * psi / n
        return n * torch.log((z.real ** 2 + z.imag ** 2).clamp(min=1e-30))

    def _thermal_logz(self, a, b):
        """log sum_v sum_j I_jv [B, L], one mixed state at a time (checkpointed: no [B, L, m, V] kept)."""
        m = self.pair_mix
        a, b = a.unflatten(2, (m, -1)), b.unflatten(2, (m, -1))
        if a.is_cuda and a.dtype == torch.complex64 and not getattr(self, "softlimit", False):   # fused (_thermal_lse)
            g = torch.exp(self.log_gain)
            return torch.logsumexp(torch.stack([_thermal_lse(a[:, :, k].real, a[:, :, k].imag, b[:, :, k].real,
                                                             b[:, :, k].imag, g, self.thermal) for k in range(m)]), 0)

        def one(aj, bj, lg):
            psi = torch.einsum("blki,blkj->blij", aj, bj).flatten(-2)
            return torch.logsumexp(self._thermal_logi(psi), -1)
        parts = [torch.utils.checkpoint.checkpoint(one, a[:, :, g], b[:, :, g], self.log_gain, use_reentrant=False)
                 if a.requires_grad else one(a[:, :, g], b[:, :, g], self.log_gain) for g in range(m)]
        return torch.logsumexp(torch.stack(parts), 0)

    def _pair_logp(self, u, target):
        if self.thermal:                                                     # the reference fixes the scale: kept in
            u = u / (u ** 2).sum(-1, keepdim=True).sqrt().clamp(min=1e-30)   # an absent position: u = 0, not 0/0
        else:
            u = u / (u ** 2).sum(-1, keepdim=True).sqrt().detach()         # held scale: P is unchanged
        a, b = self.registers(u)
        t = target.clamp(min=0)
        if self.thermal:
            n2, m = self.pair[1], self.pair_mix
            ai = a.gather(-1, (t // n2)[..., None, None].expand(*t.shape, a.shape[2], 1))[..., 0]
            bj = b.gather(-1, (t % n2)[..., None, None].expand(*t.shape, b.shape[2], 1))[..., 0]
            psi_t = (ai * bj).unflatten(-1, (m, -1)).sum(-1)                   # [B, L, m]
            lit = torch.logsumexp(self._thermal_logi(psi_t), -1)
            return (lit - self._thermal_logz(a, b)) * (target >= 0)
        n2 = self.pair[1]
        ai = a.gather(-1, (t // n2)[..., None, None].expand(*t.shape, a.shape[2], 1))[..., 0]   # [B, L, r]
        bj = b.gather(-1, (t % n2)[..., None, None].expand(*t.shape, b.shape[2], 1))[..., 0]
        m = self.pair_mix
        amp = (ai * bj).unflatten(-1, (m, -1)).sum(-1)                         # [B, L, m]
        it = (amp.real ** 2 + amp.imag ** 2).sum(-1)
        a, b = a.unflatten(2, (m, -1)), b.unflatten(2, (m, -1))
        ga = torch.einsum("blgki,blgni->blgkn", a.conj(), a)                 # <a_k|a_n> within a state
        gb = torch.einsum("blgkj,blgnj->blgkn", b.conj(), b)
        z = (ga * gb).sum((-1, -2, -3)).real                                 # sum_j ||sum_k a_jk (x) b_jk||^2
        return (torch.log(torch.maximum(it, 1e-12 * z)) - torch.log(z)) * (target >= 0)

    def log_prob(self, ids, target):
        """log P_target [B, L] for training: log I_target - log Z, with no [B, L, V] probabilities built (the same
        numbers as log probs(ids) gathered, faster for a word vocabulary). target -100 gives 0 there."""
        h, _ = self.hidden(ids, self.amplitudes(ids))
        if getattr(self, "linhead", False):
            lp = torch.log_softmax(self.lin(h.real if h.is_complex() else h), -1)
            return lp.gather(-1, target.clamp(min=0)[..., None])[..., 0] * (target >= 0)
        if self.pair:
            return self._pair_logp(h, target)
        r = self.readout_rows()
        head = _logp_head_c if h.is_cuda else _logp_head
        return head(h, r.real, r.imag, target, self.per_token)

    def readout_rows(self):
        # phi_v = <e_v|h> when tied. conj() would be a lazy view whose .imag torch.compile read with the wrong sign
        if self.wide:
            a = torch.complex(self.emb[0], -self.emb[1])
            b = self._c(self.readout_b)
            return (a[:, :, None] * b[:, None, :]).flatten(1)                   # [V, wide]
        return torch.complex(self.emb[0], -self.emb[1]) if self.tie else self._c(self.readout)

    def intensities(self, ids, amp, frozen=None, scores=None, at=None):
        """(|phi_b|^2 [B, L, N] over a held scale, the frozen-value wrapper); [B, P, N] at positions `at`."""
        h, fz = self.hidden(ids, amp, frozen, scores)
        if at is not None:
            h = h[:, at]
        if self.pair and self.thermal:
            u = h / fz((h ** 2).sum(-1, keepdim=True).sqrt().clamp(min=1e-30))   # held when explaining; no 0/0
            a, b = self.registers(u)
            m = self.pair_mix
            a, b = a.unflatten(2, (m, -1)), b.unflatten(2, (m, -1))
            logi = [self._thermal_logi(torch.einsum("blki,blkj->blij", a[:, :, g], b[:, :, g]).flatten(-2))
                    for g in range(m)]
            top = fz(torch.stack([x.detach().amax(-1) for x in logi]).amax(0))[..., None]   # held, as c
            inten = 0
            for x in logi:
                inten = inten + torch.exp(x - top)
            return inten, fz
        if self.pair:
            a, b = self.registers(h)
            m = self.pair_mix                                                # one mixed state at a time: no [B, L, m, V]
            a, b = a.unflatten(2, (m, -1)), b.unflatten(2, (m, -1))
            inten = 0
            for g in range(m):
                psi = torch.einsum("blki,blkj->blij", a[:, :, g], b[:, :, g]).flatten(-2)
                inten = inten + psi.real ** 2 + psi.imag ** 2
            return inten / fz(inten.detach().amax(-1, keepdim=True).clamp(min=1e-60)), fz
        r = self.readout_rows()
        # phi = h @ r.T in real arithmetic: a complex matmul and its [B, L, N] complex tensors cost a word
        # vocabulary's step 15 %
        rr, ri = r.real.T, r.imag.T
        if h.is_complex():
            hr, hi = h.real, h.imag
            inten = (hr @ rr - hi @ ri) ** 2 + (hr @ ri + hi @ rr) ** 2
        else:                                                                # real amplitudes (a wide readout)
            inten = (h @ rr) ** 2 + (h @ ri) ** 2
        return inten / fz(inten.detach().amax(-1, keepdim=True).clamp(min=1e-60)), fz   # held, as c (|phi| max)

    def hidden(self, ids, amp, frozen=None, scores=None):
        """(the last state h [B, L, d], the frozen-value wrapper).
        `scores`, a list, receives each block's attention weights w_ij [B, H, L, L] (reference key left out)."""
        fz = _free(frozen)
        L, dh = ids.shape[1], self.d // self.heads
        e = self._c(self.emb)[ids] + self._c(self.pos)[:L]
        t = torch.arange(L, device=ids.device, dtype=self.omega.dtype)
        ang = t[:, None] * self.omega[None, :]
        h = e * torch.polar(torch.ones_like(ang), ang) * amp.unsqueeze(-1).to(e.dtype)
        causal = torch.tril(torch.ones(L, L, device=ids.device, dtype=amp.dtype))
        mass_prefix = fz(amp.cumsum(1)) if self.ref else None                # [B, L]
        for blk, (qkvo, wv) in enumerate(zip(self.qkvo, self.wv)):
            ga, gm = self.res_gain[blk] if self.res_gain is not None else (1.0, 1.0)
            m = self._c(qkvo)
            ps = getattr(self, "phase_sigma", 0.0) if self.training else 0.0
            m = mesh_noise(m, ps)                                            # [4, d, d]: one mesh pair each
            z = self._prenorm(h, fz)
            q, k, v = (torch.einsum("btd,ed->bte", z, m[i]).unflatten(-1, (self.heads, dh)) for i in range(3))
            ov = torch.einsum("bihe,bjhe->bhij", q.conj(), k).abs() * causal     # |<q_i|k_j>|, j <= i
            if self.ref:
                r = self.mean_gain * z.cumsum(1) / (mass_prefix[..., None] + 1e-30).to(h.dtype)
                kr, vr = (torch.einsum("btd,ed->bte", r, m[i]).unflatten(-1, (self.heads, dh)) for i in (1, 2))
                orr = (q.conj() * kr).sum(-1).abs().transpose(1, 2)             # [B, H, L]
            hook = getattr(self, "overlap_hook", None)
            if hook is not None:                       # overlaps read elsewhere (hw_word.py: Hadamard tests on qubits)
                ov_m, orr_m = hook(blk, q, k, kr if self.ref else None)          # complex [B, H, L, L], [B, H, L]
                ov = ov_m.abs().to(ov.dtype) * causal
                if self.ref:
                    orr = orr_m.abs().to(orr.dtype)
            # The loss does not see the scale of the state, which drifts (Z reached 1e24 and |<q|k>|^4 overflowed
            # float32); each row is divided by its largest overlap before any power, a held constant, so w is
            # unchanged exactly
            c = ov.amax(-1)
            if self.ref:
                c = torch.maximum(c, orr)
            c = fz(c.detach().clamp(min=1e-30))[..., None]                     # [B, H, L, 1]
            s = (ov / c) ** self.power                                         # |<q_i|k_j>|^p / c_i^p
            y = torch.einsum("bhij,bjhe->bihe", s.to(v.dtype), v)
            den = s.sum(-1)                                                    # [B, H, L]
            if self.ref:
                sr = (orr / c[..., 0]) ** self.power
                y = y + sr.transpose(1, 2)[..., None].to(vr.dtype) * vr
                den = den + sr
            den = fz(den)
            if scores is not None:
                scores.append((s / (den[..., None] + 1e-30)).detach())
            y = (y / (den.transpose(1, 2)[..., None] + 1e-30).to(y.dtype)).flatten(-2)
            h = h + ga * (y @ m[3].T)
            w = self._c(wv)
            if ps:                                                           # one mesh pair per bilinear form
                w = mesh_noise(w.unflatten(1, (self.mlp_mix, -1)), ps).flatten(1, 2)
            z = self._prenorm(h, fz)
            mid = (z @ w[0].T * (z @ w[1].T).conj()).real                        # Re[(W h) * conj(V h)]
            if self.mlp_mix > 1:
                mid = mid.unflatten(-1, (self.mlp_mix, -1)).sum(-2)              # m forms per output
            if self.wide and blk == self.n_blocks - 1:
                return mid, fz                                                 # measured as it is (real amplitudes)
            h = h + gm * (mid.to(h.dtype) @ mesh_noise(self._c(self.u[blk]), ps).T)
        return h, fz

    def probs(self, ids, amp=None):
        if getattr(self, "linhead", False):
            h, _ = self.hidden(ids, self.amplitudes(ids) if amp is None else amp)
            return torch.softmax(self.lin(h.real if h.is_complex() else h), -1)
        num, z = self.numerators(ids, self.amplitudes(ids) if amp is None else amp)
        return num / (z[..., None] + 1e-30) + self.share

    def attention(self, ids):
        """Last block's weights over the keys j <= i, mean of heads, [B, L, L]."""
        scores = []
        self.numerators(ids, self.amplitudes(ids), scores=scores)
        return scores[-1].mean(1)


class SoftmaxBlock(nn.Module):
    def __init__(self, d, heads, d_ff):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, h, allowed):
        """allowed [B, L, L]: query i may read key j."""
        z = self.ln1(h)
        q, k, v = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in self.qkv(z).chunk(3, -1))
        s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        s = s.masked_fill(~allowed[:, None], -1e9)
        w = torch.softmax(s, -1)
        h = h + self.o((w @ v).transpose(1, 2).flatten(-2))
        return h + self.mlp(self.ln2(h)), w


class CausalSoftmax(nn.Module):
    family = "S"

    def __init__(self, vocab, max_len, d=64, heads=2, d_ff=256, blocks=2, tie=False):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(SoftmaxBlock(d, heads, d_ff) for _ in range(blocks))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)
        if tie:                                             # the head is the embedding table (GPT-2's choice)
            self.head.weight = self.emb.weight

    def embed(self, ids):
        return self.emb(ids) + self.pos.weight[: ids.shape[1]]

    def from_embedded(self, x, keep=None, with_attn=False):
        """Logits [B, L, V]; keep [B, L] (False = token deleted: its field is zero and no query reads it)."""
        L = x.shape[1]
        allowed = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device)).expand(x.shape[0], L, L)
        if keep is not None:
            x = x * keep[..., None].to(x.dtype)
            allowed = allowed & keep[:, None, :]
            allowed = allowed | torch.eye(L, dtype=torch.bool, device=x.device)    # a row never empty
        h, attn = x, None
        for b in self.blocks:
            h, attn = b(h, allowed)
        out = self.head(self.ln(h))
        return (out, attn.mean(1)) if with_attn else out

    def forward(self, ids, keep=None):
        return self.from_embedded(self.embed(ids), keep)



def build(name, vocab, max_len):
    """The paper's models. born-x8x2:<options>: the Born language model (m4,x1,r16,p2 in the paper; t<n> the n-pass
    readout, s1 its n -> infinity softmax limit); softmax-w96: the softmax transformer of the same size."""
    if name.startswith("born-x8x2:"):
        kw = dict(d=32, blocks=2, power=4, pair_rank=2, pair_mix=8)
        for opt in name.split(":", 1)[1].split(","):
            kw[{"d": "d", "b": "blocks", "p": "power", "n": "norm", "h": "heads", "f": "d_ff", "m": "mlp_mix",
                "t": "thermal", "x": "pair_mix", "r": "pair_rank", "s": "softlimit", "l": "linhead"}[opt[0]]] = int(opt[1:])
        return CausalBorn(vocab, max_len, per_token=1, tie=True, wide=64, gain0=0.0, pair=(64, 128), **kw)
    if name == "softmax-w96":                             # softmax-w at about the Born 8x2 model's size (~1M)
        return CausalSoftmax(vocab, max_len, d=96, d_ff=384, tie=True)
    raise ValueError(f"unknown model {name!r}")


def count(model):
    return sum(p.numel() for p in model.parameters())
