"""
usadel_free_energy.py
=====================
Matsubara Usadel solver + quasiclassical free energy for a Py / Nb / Ho
trilayer, built to compute the superconducting free-energy cost of
equal-spin-triplet generation by Ho's conical magnetisation, and hence the
effective magnetic anisotropy field that the superconductor exerts back on Ho.

Physics & literature anchors
----------------------------
* Usadel equation in the diffusive limit, full 4x4 Nambu(x)spin structure,
  Riccati parametrisation with INDEPENDENT gamma and gamma-tilde
  (Schopohl-Maki; Eschrig; Jacobsen-Ouassou-Linder PRB 92, 024510;
  Linder-Yokoyama-Sudbo PRB 79, 054523 for the conical-Ho setup).
* Free energy from the quasiclassical (Luttinger-Ward / Eilenberger-type)
  functional in the diffusive limit,
      F_s/N0 = |Delta|^2/lambda
               - (pi T/2) Sum_w tr{ (w_n + i h.sigma) tau3 (g - g_n)
                                    + Delta_hat g - (hbar D/4) (grad g)^2 },
  Virtanen, Vargunin & Silaev, PRB 101, 094507 (2020) [arXiv:1909.00992],
  Eqs. (20)-(21). Its saddle point reproduces the Usadel + gap equations,
  so the self-consistently solved profile is a stationary point of F.
* SC-induced magnetic anisotropy as an angle-dependent free-energy
  contribution from triplet generation: Johnsen-Banerjee-Linder PRB 99,
  134516 (2019); Gonzalez-Ruano et al. PRB 102, 020405(R) (2020).

Conventions (all verified numerically in tests.py)
--------------------------------------------------
ghat(gamma, gtil) = [[2N-1,        2 N gamma   ],
                     [-2 Nt gtil, -(2 Nt - 1) ]],
    N = (1 - gamma gtil)^-1,  Nt = (1 - gtil gamma)^-1     (ghat^2 = 1).

Usadel (Matsubara):   hbarD d/dx( ghat d/dx ghat ) = [Lam, ghat],
    Lam = w_n tau3 + i Vh + Dhat,
    Vh  = diag( h.sigma, +(h.sigma)^* ),
    Dhat= [[0, Delta i sy], [-Delta i sy, 0]]  (Delta real).

Bulk BCS:  gamma = gtil = Delta/(w_n + Omega) * i sy,
           ghat = (w_n tau3 + Dhat)/Omega,  Omega = sqrt(w_n^2 + Delta^2).
Collinear h||z bulk: pair amplitudes Delta/sqrt((w_n +/- i h)^2 + Delta^2)
(verified; the diag(h.sigma, -(h.sigma)^*) structure would wrongly commute
with the singlet and produce NO pair breaking).

Riccati form (derived numerically from the (12),(21) blocks of the full
Usadel operator; exact to machine precision, see scratch_fit.py / tests.py;
identical in structure to Linder-Yokoyama-Sudbo PRB 79, 054523 Eq. (10)
after continuation E -> i w_n):

  hbarD [g'' + 2 g' Nt gtil g']   = 2 w_n g + i(hs g - g hs*) - D isy - D g isy g
  hbarD [gt'' + 2 gt' N gamma gt']= 2 w_n gt + i(gt hs - hs* gt) - D isy - D gt isy gt

with hs = h.sigma. NOTE: for a noncollinear h(x) there is NO pointwise
relation gtil = f(gamma); both must be evolved (verified numerically).

Units
-----
Energies in Delta00 (zero-T bulk gap of Nb). Lengths in xi0 = sqrt(hbar D_Nb /
Delta00). Temperatures as t = T/Tc, with k_B T = t/1.7639 in gap units.
Free energy returned in units of N0 * Delta00^2 * xi0 (per unit interface
area), where N0 is the (two-spin) normal DOS at E_F of Nb. The code
calibrates/verifies against the BCS condensation energy -N0 Delta^2/2 in
tests.py, and run_anisotropy.py converts to SI.

Different diffusion constants per layer are handled exactly by the
coordinate transformation u with dx = (D_j/D_Nb) du, under the assumption of
a uniform N0 (stated approximation): gamma and d(gamma)/du are then
continuous across transparent interfaces and the whole stack is one smooth
boundary-value problem. Interfaces are perfectly transparent (rigid-contact
limit); Kupriyanov-Lukichev barriers are a documented extension.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_bvp

# --- NumPy compatibility: trapezoid was added in NumPy 2.0 (was trapz) ---
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# ---------------------------------------------------------------------------
# constants / matrices
# ---------------------------------------------------------------------------
BCS_RATIO = 1.7639  # Delta00 / (kB Tc)

s0 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
isy = 1j * sy
SIG = np.stack([sx, sy, sz])          # (3,2,2)
SIGC = SIG.conj()                     # h.sigma^*
TAU3 = np.kron(np.diag([1, -1]).astype(complex), s0)
PGAP = np.block([[np.zeros((2, 2)), isy], [-isy, np.zeros((2, 2))]])  # dDhat/dDelta


# ---------------------------------------------------------------------------
# structure definition
# ---------------------------------------------------------------------------
@dataclass
class Layer:
    name: str
    d: float                 # thickness [xi0]
    D: float = 1.0           # diffusion constant / D_Nb
    sc: bool = False         # participates in pairing (Nb)
    h_fun: object = None     # callable h(x_local)->(3,n) exchange field [Delta00]


@dataclass
class Stack:
    layers: list
    t: float                       # T / Tc
    n_matsubara: int = 20
    n_per_layer: int = 40          # initial mesh points per layer (u space)
    min_pts_per_period: int = 12   # resolve spiral
    tol: float = 1e-6
    max_nodes: int = 12000
    w_int: float = 0.04            # interface smoothing width [u units]
                                   # (smooth coefficient profiles keep the
                                   # collocation BVP well-posed; F differences
                                   # are insensitive, checked in tests)

    def __post_init__(self):
        self.Theta = self.t / BCS_RATIO                     # k_B T [Delta00]
        self.wn = np.pi * self.Theta * (2 * np.arange(self.n_matsubara) + 1)
        # lambda^-1 = ln t + 2 pi Theta sum 1/wn   (cutoff-consistent BCS)
        self.inv_lambda = np.log(self.t) + 2 * np.pi * self.Theta * np.sum(1 / self.wn)
        # u-grid (dx = D_j du  =>  L_u = L_x / D_j)
        self.x_edges = np.concatenate([[0.0], np.cumsum([L.d for L in self.layers])])
        self.u_edges = np.concatenate([[0.0], np.cumsum([L.d / L.D for L in self.layers])])
        self._build_mesh()

    def _build_mesh(self):
        us = []
        for j, L in enumerate(self.layers):
            n = self.n_per_layer
            if L.h_fun is not None and getattr(L, "_Q", 0.0):
                periods = L.d * abs(L._Q) / (2 * np.pi)
                n = max(n, int(np.ceil(self.min_pts_per_period * periods)) + 4)
            grid = np.linspace(self.u_edges[j], self.u_edges[j + 1], n, endpoint=False)
            us.append(grid)
        us.append([self.u_edges[-1]])
        self.u_mesh = np.concatenate(us)

    # ---- coordinate helpers ----
    def layer_index(self, u):
        idx = np.searchsorted(self.u_edges, u, side="right") - 1
        return np.clip(idx, 0, len(self.layers) - 1)

    def x_of_u(self, u):
        j = self.layer_index(u)
        Dj = np.array([L.D for L in self.layers])[j]
        return self.x_edges[j] + (u - self.u_edges[j]) * Dj

    def D_of_u(self, u):
        """Smoothed diffusion-constant profile (ratio to D_Nb)."""
        u = np.atleast_1d(u)
        num = np.zeros(u.size)
        den = np.zeros(u.size)
        for k, L in enumerate(self.layers):
            w = self.window(u, k)
            num += w * L.D
            den += w
        return num / np.maximum(den, 1e-12)

    def window(self, u, k):
        """Smooth indicator of layer k in u-space (tanh walls of width w_int).
        Outer surfaces are hard (no material beyond)."""
        u = np.atleast_1d(u)
        a, b = self.u_edges[k], self.u_edges[k + 1]
        one = np.ones(u.size)
        wl = one if k == 0 else 0.5 * (1 + np.tanh((u - a) / self.w_int))
        wr = one if k == len(self.layers) - 1 else 0.5 * (1 - np.tanh((u - b) / self.w_int))
        return wl * wr

    def h_of_u(self, u):
        """exchange field, shape (3, n), smoothly windowed per layer"""
        u = np.atleast_1d(u)
        h = np.zeros((3, u.size))
        for k, L in enumerate(self.layers):
            if L.h_fun is not None:
                xloc = np.clip((u - self.u_edges[k]) * L.D, 0.0,
                               L.d - 1e-12)
                h += self.window(u, k)[None, :] * L.h_fun(xloc)
        return h

    def sc_window(self, u):
        """Smooth SC indicator (for the pairing interaction)."""
        u = np.atleast_1d(u)
        w = np.zeros(u.size)
        for k, L in enumerate(self.layers):
            if L.sc:
                w += self.window(u, k)
        return np.clip(w, 0.0, 1.0)

    def sc_mask(self, u):
        j = self.layer_index(np.atleast_1d(u))
        return np.array([L.sc for L in self.layers])[j]


# ---------------------------------------------------------------------------
# vectorised Riccati algebra   (arrays shaped (n, 2, 2))
# ---------------------------------------------------------------------------
def _unpack(y):
    """(32, n) real -> gamma, gtil, dgamma, dgtil each (n,2,2) complex."""
    n = y.shape[1]
    z = (y[0:16] + 1j * y[16:32]).T.reshape(n, 4, 2, 2)
    return z[:, 0], z[:, 1], z[:, 2], z[:, 3]


def _pack(g, gt, dg, dgt):
    n = g.shape[0]
    z = np.stack([g, gt, dg, dgt], axis=1).reshape(n, 16).T
    return np.concatenate([z.real, z.imag], axis=0)


def _inv22(a):
    det = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
    # regularise transient singularities during Newton iterations
    small = np.abs(det) < 1e-10
    if np.any(small):
        det = np.where(small, det + 1e-10, det)
    out = np.empty_like(a)
    out[..., 0, 0] = a[..., 1, 1]
    out[..., 1, 1] = a[..., 0, 0]
    out[..., 0, 1] = -a[..., 0, 1]
    out[..., 1, 0] = -a[..., 1, 0]
    return out / det[..., None, None]


def _NNt(g, gt):
    I = np.broadcast_to(s0, g.shape)
    return _inv22(I - g @ gt), _inv22(I - gt @ g)


def ghat_blocks(g, gt):
    """Return the four 2x2 blocks of ghat, each (n,2,2)."""
    N, Nt = _NNt(g, gt)
    I = np.broadcast_to(s0, g.shape)
    return 2 * N - I, 2 * (N @ g), -2 * (Nt @ gt), -(2 * Nt - I)


def dghat_blocks(g, gt, dg, dgt):
    """Blocks of d(ghat)/du."""
    N, Nt = _NNt(g, gt)
    A1 = dg @ gt + g @ dgt
    A1t = dgt @ g + gt @ dg
    dN = N @ A1 @ N
    dNt = Nt @ A1t @ Nt
    return 2 * dN, 2 * (dN @ g + N @ dg), -2 * (dNt @ gt + Nt @ dgt), -2 * dNt


def assemble4(b11, b12, b21, b22):
    n = b11.shape[0]
    M = np.zeros((n, 4, 4), dtype=complex)
    M[:, :2, :2] = b11
    M[:, :2, 2:] = b12
    M[:, 2:, :2] = b21
    M[:, 2:, 2:] = b22
    return M


# ---------------------------------------------------------------------------
# ODE right-hand side (in u coordinate)
# ---------------------------------------------------------------------------
def make_rhs(stack: Stack, wn: float, Delta_interp):
    def rhs(u, y):
        g, gt, dg, dgt = _unpack(y)
        Dj = stack.D_of_u(u)[:, None, None]
        h = stack.h_of_u(u)                                   # (3,n)
        De = Delta_interp(u)[:, None, None]
        hs = np.einsum("kn,kij->nij", h, SIG)
        hsc = np.einsum("kn,kij->nij", h, SIGC)
        N, Nt = _NNt(g, gt)
        d2g = Dj * (2 * wn * g + 1j * (hs @ g - g @ hsc)
                    - De * isy - De * (g @ isy @ g)) \
            - 2 * (dg @ Nt @ gt @ dg)
        d2gt = Dj * (2 * wn * gt + 1j * (gt @ hs - hsc @ gt)
                     - De * isy - De * (gt @ isy @ gt)) \
            - 2 * (dgt @ N @ g @ dgt)
        return _pack(dg, dgt, d2g, d2gt)
    return rhs


def _bc(ya, yb):
    """vacuum: d(gamma)/du = d(gtil)/du = 0 at both outer surfaces."""
    return np.concatenate([ya[8:16], ya[24:32], yb[8:16], yb[24:32]])


def initial_guess(stack: Stack, wn: float, Delta_of_u):
    """Bulk-like gamma in SC, exponentially tapered into the F layers."""
    u = stack.u_mesh
    n = u.size
    De = Delta_of_u(u)
    scmask = stack.sc_mask(u)
    # distance (in u) to nearest SC region
    if scmask.any():
        usc = u[scmask]
        dist = np.min(np.abs(u[:, None] - usc[None, :]), axis=1)
        Dbar = max(De[scmask].mean(), 1e-6)
    else:
        dist = np.zeros(n)
        Dbar = 1e-6
    Om = np.hypot(wn, Dbar)
    c = Dbar / (wn + Om)
    kappa = np.sqrt(2 * wn)
    amp = c * np.exp(-kappa * dist)
    g = amp[:, None, None] * isy
    gt = g.copy()
    dg = np.zeros_like(g)
    return _pack(g, gt, dg, np.zeros_like(g))


def solve_frequency(stack: Stack, wn: float, Delta_of_u, y0=None, u0=None):
    """Solve the trilayer BVP at one Matsubara frequency."""
    rhs = make_rhs(stack, wn, Delta_of_u)
    if y0 is None:
        u = stack.u_mesh
        y = initial_guess(stack, wn, Delta_of_u)
    else:
        u, y = u0, y0
    sol = solve_bvp(rhs, _bc, u, y, tol=stack.tol,
                    max_nodes=stack.max_nodes, verbose=0)
    if not sol.success:
        # retry from scratch guess with looser tol
        sol = solve_bvp(rhs, _bc, stack.u_mesh,
                        initial_guess(stack, wn, Delta_of_u),
                        tol=stack.tol * 10, max_nodes=stack.max_nodes, verbose=0)
    return sol


# ---------------------------------------------------------------------------
# observables
# ---------------------------------------------------------------------------
def pair_amplitudes(g, gt):
    """Singlet f0 and triplet vector (fx,fy,fz) from the upper-right block
    f = 2 N gamma, decomposed as f = (f0 + f_t . sigma) i sy.
    Also the tilde amplitudes from the lower-left block."""
    _, f, fbar, _ = ghat_blocks(g, gt)
    M = f @ (-isy)           # = f0 + f.sigma
    f0 = 0.5 * np.trace(M, axis1=-2, axis2=-1)
    ft = 0.5 * np.stack([np.trace(M @ s, axis1=-2, axis2=-1) for s in SIG], axis=-1)
    return f0, ft


def singlet_for_gap(g, gt):
    """f_s = tr[P ghat]/4 ; equals Delta/Omega in bulk. Drives the gap eqn."""
    G = assemble4(*ghat_blocks(g, gt))
    return np.einsum("ij,nji->n", PGAP, G) / 4.0


def free_energy_density(stack: Stack, wn, sols, Delta_of_u, u_grid):
    """Free-energy areal density integrand on u_grid, all frequencies summed.

    In u-coordinates (dx = D_j du):
      F = int du { D_j * [ Delta^2 (ln t)/(#SC handled globally) ... ] }  --
    implemented explicitly below:

      F = int dx Delta^2 ln t
        + sum_n pi Theta int dx { 2 Delta^2 / w_n }
        - pi Theta sum_n Re int dx tr[ (w_n tau3 + i Vh)(ghat - tau3) + Dhat ghat ]
        - pi Theta sum_n Re int du (-1/4) tr[ (d_u ghat)^2 ]  ... sign folded in

    with int dx = int du D_j and (hbarD_j/4)(d_x ghat)^2 dx = (1/4)(d_u ghat)^2 du.
    Returns (u_grid, f_density_in_u) where F = trapz(f_density, u_grid).
    """
    Theta = stack.Theta
    Dj = stack.D_of_u(u_grid)
    De = Delta_of_u(u_grid)
    h = stack.h_of_u(u_grid)
    hs = np.einsum("kn,kij->nij", h, SIG)
    hsc = np.einsum("kn,kij->nij", h, SIGC)
    n = u_grid.size

    Vh = assemble4(hs, np.zeros_like(hs), np.zeros_like(hs), hsc)
    Dhat = De[:, None, None] * np.broadcast_to(PGAP, (n, 4, 4))

    fdens = np.zeros(n)
    # condensation counterterm (x-measure -> multiply by Dj)
    fdens += Dj * De**2 * np.log(stack.t)
    Fn = np.zeros(len(wn))
    for k, w in enumerate(wn):
        contrib = Dj * np.pi * Theta * 2 * De**2 / w

        y = sols[k].sol(u_grid)
        g, gt, dg, dgt = _unpack(y)
        G = assemble4(*ghat_blocks(g, gt))
        dG = assemble4(*dghat_blocks(g, gt, dg, dgt))

        L1 = w * TAU3 + 1j * Vh
        tr1 = np.einsum("nij,nji->n", L1, G - TAU3)
        tr2 = np.einsum("nij,nji->n", Dhat, G)
        tr3 = np.einsum("nij,nji->n", dG, dG)
        # - pi T Re { Dj*(tr1+tr2) - (1/4) tr3 }   [tr3 already in du measure]
        contrib = contrib - np.pi * Theta * np.real(Dj * (tr1 + tr2) - 0.25 * tr3)
        fdens += contrib
        Fn[k] = np.trapezoid(contrib, u_grid)
    return fdens, Fn


def extrapolate_tail(F_base, Fn):
    """F(N) = F_base + cumsum(Fn). Interface contributions leave a power-law
    tail Fn ~ a n^{-p} (p ~ 1.5-2). Fit p on the last third and add the
    analytic remainder sum_{n>N} a n^{-p} ~ Fn[N] * N/(p-1)."""
    FN = F_base + np.cumsum(Fn)
    N = len(Fn)
    if N < 8 or np.any(np.abs(Fn[int(0.6 * N):]) < 1e-14):
        return FN[-1]
    n = np.arange(1, N + 1)
    m = n >= max(4, int(0.6 * N))
    sgn = np.sign(Fn[-1])
    if not np.all(np.sign(Fn[m]) == sgn):     # non-monotone tail: don't fit
        return FN[-1]
    p = -np.polyfit(np.log(n[m]), np.log(np.abs(Fn[m])), 1)[0]
    p = np.clip(p, 1.2, 3.0)
    return FN[-1] + Fn[-1] * N / (p - 1.0)


# ---------------------------------------------------------------------------
# self-consistent solve
# ---------------------------------------------------------------------------
@dataclass
class Result:
    stack: Stack
    u: np.ndarray
    x: np.ndarray
    Delta: np.ndarray
    F: float
    F_extrap: float           # Matsubara-tail-extrapolated free energy
    Fn: np.ndarray            # per-frequency contributions to F
    f_density: np.ndarray
    f0: np.ndarray            # (n_w, n_u) singlet amplitude
    ft: np.ndarray            # (n_w, n_u, 3) triplet vector
    converged: bool
    n_outer: int
    sols: list = field(repr=False, default=None)


def bcs_gap(t):
    """Self-consistent bulk BCS gap at reduced temperature t (units Delta00)."""
    if t >= 1:
        return 0.0
    Theta = t / BCS_RATIO
    wn = np.pi * Theta * (2 * np.arange(400) + 1)
    inv_lam = np.log(t) + 2 * np.pi * Theta * np.sum(1 / wn)
    De = 1.0
    for _ in range(200):
        rhs = 2 * np.pi * Theta * np.sum(De / np.hypot(wn, De)) / inv_lam
        if abs(rhs - De) < 1e-12:
            De = rhs
            break
        De = rhs
    return De


def solve_selfconsistent(stack: Stack, Delta_init=None, warm=None,
                         max_outer=40, tol_delta=2e-4, mix=0.5, verbose=False):
    """Solve all Matsubara frequencies + gap self-consistency."""
    u_out = stack.u_mesh.copy()
    scwin = stack.sc_window(u_out)
    scmask = stack.sc_mask(u_out)
    Dbulk = bcs_gap(stack.t)

    if Delta_init is None:
        Delta = scwin * Dbulk
    else:
        Delta = Delta_init(u_out) if callable(Delta_init) else np.array(Delta_init)

    def make_interp(Dvals):
        def f(u):
            return np.interp(np.atleast_1d(u), u_out, Dvals)
        return f

    sols = [None] * stack.n_matsubara if warm is None else list(warm)
    converged = False
    D_prev, r_prev = None, None
    for it in range(max_outer):
        Dint = make_interp(Delta)
        fs = np.zeros(u_out.size)
        for k, w in enumerate(stack.wn):
            if sols[k] is None:
                sol = solve_frequency(stack, w, Dint)
            else:
                sol = solve_frequency(stack, w, Dint,
                                      y0=sols[k].sol(sols[k].x), u0=sols[k].x)
            sols[k] = sol
            g, gt, _, _ = _unpack(sol.sol(u_out))
            fs += np.real(singlet_for_gap(g, gt))
        GD = np.clip(scwin * 2 * np.pi * stack.Theta * fs / stack.inv_lambda,
                     0.0, 1.2)
        r = GD - Delta                       # fixed-point residual
        err = np.max(np.abs(r))
        if verbose:
            print(f"    outer {it+1:2d}: |dDelta| = {err:.2e}  "
                  f"Delta_mid/D00 = {Delta[scmask].mean():.4f}")
        if err < tol_delta and it >= 1:
            converged = True
            break
        # Anderson(1) acceleration of Delta -> G(Delta)
        if r_prev is not None:
            dr = r - r_prev
            denom = np.dot(dr, dr)
            theta = np.dot(r, dr) / denom if denom > 1e-30 else 0.0
            theta = np.clip(theta, -2.0, 0.9)
            Dnext = (1 - theta) * GD + theta * (D_prev + r_prev)
        else:
            Dnext = Delta + mix * r
        D_prev, r_prev = Delta.copy(), r.copy()
        Delta = np.clip(Dnext, 0.0, 1.2)
        if Delta[scmask].max() < 5e-3:
            Delta[:] = 0.0
            converged = True
            if verbose:
                print("    gap collapsed -> normal state")
            break

    # ---- final pass: re-solve ALL frequencies at the final Delta so that the
    # free energy is evaluated on a fully consistent saddle point (otherwise
    # stale warm-started solutions leave a spurious ~1/w_n tail in F_n).
    collapsed = Delta.max() <= 0.0
    Dint = make_interp(Delta)
    for k, w in enumerate(stack.wn):
        if sols[k] is None:
            sols[k] = solve_frequency(stack, w, Dint)
        else:
            sols[k] = solve_frequency(stack, w, Dint,
                                      y0=sols[k].sol(sols[k].x), u0=sols[k].x)

    # dense evaluation grid for the free energy (union of solver meshes)
    u_f = np.unique(np.concatenate([s.x for s in sols] + [u_out]))
    fdens, Fn = free_energy_density(stack, stack.wn, sols, Dint, u_f)
    F = np.trapezoid(fdens, u_f)
    F_base = F - Fn.sum()               # ln(t) counterterm part
    F_extrap = extrapolate_tail(F_base, Fn)
    if collapsed:                        # gamma = 0, Delta = 0: F = 0 exactly
        F, F_extrap, Fn = 0.0, 0.0, np.zeros_like(Fn)
        fdens = np.zeros_like(fdens)

    f0 = np.zeros((stack.n_matsubara, u_out.size), dtype=complex)
    ft = np.zeros((stack.n_matsubara, u_out.size, 3), dtype=complex)
    for k in range(stack.n_matsubara):
        g, gt, _, _ = _unpack(sols[k].sol(u_out))
        f0[k], ft[k] = pair_amplitudes(g, gt)

    return Result(stack=stack, u=u_out, x=stack.x_of_u(u_out), Delta=Delta,
                  F=F, F_extrap=F_extrap, Fn=Fn, f_density=fdens, f0=f0, ft=ft,
                  converged=converged, n_outer=it + 1, sols=sols)


# ---------------------------------------------------------------------------
# exchange-field factories
# ---------------------------------------------------------------------------
def uniform_field(h, direction):
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    def f(xloc):
        xloc = np.atleast_1d(xloc)
        return h * np.outer(d, np.ones(xloc.size))
    return f


def conical_field(h, alpha, Q, phi0=0.0):
    """Ho conical texture, cone axis = x (film normal):
       h(x) = h [ cos(a) x^ + sin(a) ( cos(Qx+phi0) y^ + sin(Qx+phi0) z^ ) ].
    alpha: cone opening angle from the axis (Ho: 4pi/9 = 80 deg)."""
    def f(xloc):
        xloc = np.atleast_1d(xloc)
        ph = Q * xloc + phi0
        return h * np.vstack([np.full(xloc.size, np.cos(alpha)),
                              np.sin(alpha) * np.cos(ph),
                              np.sin(alpha) * np.sin(ph)])
    return f


def make_pynbho_stack(t=0.5, d_py=0.5, d_nb=2.0, d_ho=1.0,
                      h_py=10.0, h_ho=5.0, alpha=np.deg2rad(80.0),
                      lam_spiral=0.63, phi0=0.0,
                      D_py=1.0, D_nb=1.0, D_ho=1.0,
                      py_dir=(0, 0, 1), include_py=True,
                      n_matsubara=20, n_per_layer=30, tol=1e-6):
    """Build the Py/Nb/Ho stack. Lengths in xi0, energies in Delta00."""
    Q = 2 * np.pi / lam_spiral
    layers = []
    if include_py and d_py > 0:
        layers.append(Layer("Py", d_py, D=D_py, h_fun=uniform_field(h_py, py_dir)))
    layers.append(Layer("Nb", d_nb, D=D_nb, sc=True))
    ho = Layer("Ho", d_ho, D=D_ho, h_fun=conical_field(h_ho, alpha, Q, phi0))
    ho._Q = Q
    layers.append(ho)
    return Stack(layers=layers, t=t, n_matsubara=n_matsubara,
                 n_per_layer=n_per_layer, tol=tol)
