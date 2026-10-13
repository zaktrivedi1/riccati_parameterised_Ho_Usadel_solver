"""
run_anisotropy.py
=================
Superconductor-induced magnetic anisotropy on Ho in a Py/Nb/Ho stack.

Physics
-------
Ho's conical magnetisation (cone axis = film normal x, opening angle alpha,
turning wavevector Q) converts singlet pairs in Nb into (equal-spin) triplet
pairs. This pair "drain" costs superconducting condensation energy, so the
superconducting free energy F_S depends on the cone angle: F_S = F_S(alpha).
The superconductor therefore exerts a torque -dF_S/dalpha on the Ho cone,
i.e. an effective anisotropy field, opposed by Ho's intrinsic
magnetocrystalline anisotropy, modelled as

    E_K(alpha) = K_eff * d_Ho * sin^2(alpha - alpha0)   [per unit area]

with alpha0 the intrinsic cone angle (80 deg in Ho). Minimising

    F_tot(alpha) = F_S(alpha) + E_K(alpha)

gives the equilibrium tilt shift  d_alpha = alpha* - alpha0  and the
SC-induced anisotropy quantities:

    torque per area      tau_SC       = -dF_S/dalpha |_{alpha0}
    equivalent axial field mu0*H_an   =  tau_SC / (M_s d_Ho sin(alpha0))
        (a uniform field B along the cone axis exerts the torque
         +B M_s d_Ho sin(alpha) on the cone angle, since the axial moment
         per area is M_s d_Ho cos(alpha))
    effective anisotropy constant K_SC = (1/2) d^2F_S/dalpha^2 / d_Ho

Everything is computed from the self-consistent Usadel solution and the
quasiclassical free-energy functional (Virtanen-Vargunin-Silaev PRB 101,
094507); see usadel_free_energy.py and README.md.

Units and SI conversion
-----------------------
Code units: F in N0*Delta00^2*xi0 per area, with the calibration
F_bulk(T->0) = -1/2 (tests.py T2), i.e. the condensation energy. Hence

    N0*Delta00^2 = Hc(0)^2 / mu0      (thermodynamic critical field)

and the SI conversion is  F_SI [J/m^2] = F_code * (Hc0^2/mu0) * xi0.

Run:  python run_anisotropy.py [i0 i1]   (optional slice of the alpha list,
so long sweeps can be executed in several invocations; results accumulate
in results.csv / profiles.npz)
"""
import sys, os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from usadel_free_energy import (make_pynbho_stack, solve_selfconsistent,
                                bcs_gap)

# ----------------------------------------------------------------- settings
OUT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(OUT, "results.csv")
NPZ = os.path.join(OUT, "profiles.npz")

PHYS = dict(
    t=0.30,             # T / Tc
    d_py=0.6,           # Py thickness [xi0]
    d_nb=3.0,           # Nb thickness [xi0]
    d_ho=1.5,           # Ho thickness [xi0]
    h_py=6.0,           # Py exchange field [Delta00]  (weak-FM surrogate)
    h_ho=4.0,           # Ho exchange field [Delta00]  (weak-FM surrogate)
    lam_spiral=0.63,    # Ho spiral wavelength [xi0] (~6.3 nm / xi0=10 nm)
    py_dir=(0, 0, 1),   # Py magnetisation, in plane
)
# ---------------------------------------------------------------------------
# TOLERANCE / SPEED KNOBS ---------------------------------------------------
# For a fast preliminary run, loosen these (roughly 2-3x speedup):
#     NUM = dict(n_matsubara=16, n_per_layer=24, tol=1e-4)
#     TOL_DELTA = 1e-3
# The report-quality defaults below are converged (dF_extrap/dalpha
# changes <0.1% between n_matsubara=24 and 36).
# ---------------------------------------------------------------------------
NUM = dict(n_matsubara=24, n_per_layer=30, tol=1e-5)
TOL_DELTA = 1e-4       # self-consistent-gap outer-loop tolerance

# Angles: 10 spanning [0,90], with extra density near alpha_0 = 80 deg where
# the derivative and curvature that set the physics are extracted.
ALPHAS_DEG = np.array([0, 15, 30, 45, 60, 70, 75, 80, 85, 90])

SI = dict(
    xi0=10e-9,          # m
    Hc0=0.20,           # T, Nb thermodynamic critical field
    mu0=4e-7*np.pi,
    Ms=3.0e6,           # A/m, Ho saturation magnetisation (low T)
    K_eff=5.0e5,        # J/m^3, effective cone-angle anisotropy of Ho
    alpha0_deg=80.0,    # intrinsic Ho cone angle
)

# ----------------------------------------------------------------- sweep
def run_sweep(i0=0, i1=None):
    i1 = len(ALPHAS_DEG) if i1 is None else i1
    rows = []
    Delta_prev = None
    for adeg in ALPHAS_DEG[i0:i1]:
        al = np.deg2rad(float(adeg))
        stk = make_pynbho_stack(alpha=al, **PHYS, **NUM)
        t0 = time.time()
        res = solve_selfconsistent(
            stk, mix=0.6, tol_delta=TOL_DELTA,
            Delta_init=(lambda u, D=Delta_prev, uu=None: D) if False else
                       (None if Delta_prev is None else
                        (lambda u: np.interp(np.atleast_1d(u),
                                             Delta_prev[0], Delta_prev[1]))))
        Delta_prev = (res.u, res.Delta)
        # triplet diagnostics at the lowest Matsubara frequency in Ho
        x = res.x
        in_ho = x >= (PHYS["d_py"] + PHYS["d_nb"])
        hvec = stk.h_of_u(res.u)                       # (3,n)
        hn = np.linalg.norm(hvec, axis=0)
        hh = np.where(hn > 1e-9, hn, 1.0)
        hdir = hvec / hh
        ft0 = res.ft[0]                                # (n,3) lowest w_n
        f_par = np.einsum("nk,kn->n", ft0, hdir)
        f_perp = np.linalg.norm(ft0 - f_par[:, None]*hdir.T, axis=1)
        row = dict(alpha_deg=float(adeg),
                   F=res.F, F_extrap=res.F_extrap,
                   Delta_mid=float(res.Delta[stk.sc_mask(res.u)].mean()),
                   f_perp_ho=float(np.abs(f_perp[in_ho]).max()) if in_ho.any() else 0.0,
                   f_sing_ho=float(np.abs(res.f0[0][in_ho]).max()) if in_ho.any() else 0.0,
                   converged=bool(res.converged), n_outer=res.n_outer,
                   seconds=round(time.time()-t0, 1))
        rows.append(row)
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        # store profiles for the plots
        np.savez(NPZ.replace(".npz", f"_a{int(adeg):03d}.npz"),
                 x=res.x, Delta=res.Delta, f0=res.f0[0],
                 ft=res.ft[0], fdens=res.f_density)
    header = not os.path.exists(CSV)
    with open(CSV, "a") as fh:
        if header:
            fh.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r.values()) + "\n")

# ----------------------------------------------------------------- report plots
# Publication-quality plots following the report's MH/MT style guide.
# Two figures produced, each saved as PNG (450 dpi) and SVG:
#     free_energy_vs_angle.{png,svg}      -- F_S(alpha) over full range
#     free_energy_minimisation.{png,svg}  -- F_tot decomposition near alpha_0
# The bulk 2x2 diagnostic figure `anisotropy_results.png` is kept as-is.

# Report colour anchors (from style guide)
_COL_NB = "#1f8fe0"
_COL_HO = "#9b3fd6"
_COL_PY = "#0fa878"

def _apply_report_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 16,
        "axes.linewidth": 1.1,
        "axes.labelsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

def _grid(ax):
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

def _savefig(fig, stem):
    fig.savefig(os.path.join(OUT, stem + ".png"), dpi=450, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem + ".svg"), bbox_inches="tight")

def plot_free_energy_vs_angle(a, F, Fx, aa, Fs, a0_deg):
    """F_S(alpha) plot.

    Design choice: only the Matsubara-tail-EXTRAPOLATED points are shown,
    together with the spline used for the derivative analysis. The raw-cutoff
    F(alpha) has ~7% residual cutoff drift in its slope and is not the
    quantity used to extract the anisotropy, so plotting it too would be
    misleading in a report figure. To include it as well, uncomment the
    `ax.plot(..., F, ...)` line below.
    """
    _apply_report_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    _grid(ax)
    ax.plot(np.rad2deg(aa), Fs(aa) * 1e3, "-", color=_COL_HO, lw=1.6,
            zorder=2, label="spline fit")
    ax.plot(a, Fx * 1e3, "o", color=_COL_HO, ms=8,
            markeredgecolor="black", markeredgewidth=0.6, zorder=3,
            label=r"$F_S(\alpha)$")
    # ax.plot(a, F * 1e3, "s", color="#888888", ms=6,
    #         markeredgecolor="black", markeredgewidth=0.6,
    #         label="raw cutoff (not used)")
    ax.axvline(a0_deg, color="k", ls=":", lw=1.0, zorder=1)
    ax.set_xlabel(r"Ho cone angle $\alpha$ (deg)")
    ax.set_ylabel(r"$F_S$  [$10^{-3}\,N_0\Delta_{00}^2\xi_0$]")
    ax.set_xlim(0, 90)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    _savefig(fig, "free_energy_vs_angle")
    plt.close(fig)
    print("wrote free_energy_vs_angle.{png,svg}")

def plot_minimisation_about_alpha0(aa, Fs, e_unit, Kd, a0, a_star):
    """F_tot(alpha) decomposition near alpha_0.

    Three curves: superconducting cost, intrinsic magnetocrystalline
    anisotropy, and their sum, with the equilibrium tilt marked.
    """
    _apply_report_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    _grid(ax)
    a_deg = np.rad2deg(aa)
    dFs = (Fs(aa) - Fs(a0)) * e_unit * 1e6              # uJ/m^2
    EK = Kd * np.sin(aa - a0) ** 2 * 1e6                # uJ/m^2
    Ftot = dFs + EK

    # Use the report's three-anchor palette directly (three curves =
    # three distinct meanings, not a gradient).
    ax.plot(a_deg, dFs, "-", color=_COL_NB, lw=1.8,
            label=r"$F_S-F_S(\alpha_0)$")
    ax.plot(a_deg, EK,  "-", color=_COL_PY, lw=1.8,
            label=r"$K_{\rm eff}d_{\rm Ho}\sin^2(\alpha-\alpha_0)$")
    ax.plot(a_deg, Ftot, "-", color=_COL_HO, lw=2.2,
            label=r"$F_{\rm tot}$")
    ax.axvline(np.rad2deg(a_star), color="k", ls="--", lw=1.0,
               label=fr"$\alpha^*={np.rad2deg(a_star):.3f}^\circ$")
    ax.axhline(0, color="0.5", lw=0.6, zorder=0)

    a0_deg = np.rad2deg(a0)
    ax.set_xlim(a0_deg - 8, a0_deg + 8)
    span = max(abs(dFs).max(), 1e-3)
    ax.set_ylim(-2 * span, 10 * span)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"energy per area  [$\mu$J/m$^2$]")
    ax.legend(frameon=False, loc="upper left", fontsize=14)
    fig.tight_layout()
    _savefig(fig, "free_energy_minimisation")
    plt.close(fig)
    print("wrote free_energy_minimisation.{png,svg}")

# ----------------------------------------------------------------- analysis
def analyse():
    import csv
    with open(CSV) as fh:
        rows = list(csv.DictReader(fh))
    rows = sorted(rows, key=lambda r: float(r["alpha_deg"]))
    a = np.array([float(r["alpha_deg"]) for r in rows])
    keep = np.unique(a, return_index=True)[1]
    a = a[keep]
    F = np.array([float(r["F"]) for r in rows])[keep]
    Fx = np.array([float(r["F_extrap"]) for r in rows])[keep]
    Dm = np.array([float(r["Delta_mid"]) for r in rows])[keep]
    fperp = np.array([float(r["f_perp_ho"]) for r in rows])[keep]
    arad = np.deg2rad(a)

    # smooth model F_S(alpha): cubic spline through the computed points.
    # We use the Matsubara-tail-EXTRAPOLATED free energy: validation shows
    # d(F_extrap)/dalpha is cutoff-converged to <0.1% between n_w=24 and 36,
    # while the raw-F slope still drifts ~7%. Even reflection about alpha=0
    # enforces F(-a) = F(a).
    from scipy.interpolate import CubicSpline
    # even reflection about alpha=0 AND alpha=90deg (mirror symmetries of the
    # texture) so the spline satisfies dF/dalpha = 0 at both ends
    a_ext = np.concatenate([-arad[::-1][:-1], arad,
                            np.pi - arad[::-1][1:]])
    F_ext = np.concatenate([Fx[::-1][:-1], Fx, Fx[::-1][1:]])
    spl = CubicSpline(a_ext, F_ext)
    Fs = lambda al: spl(al)
    dFs = lambda al: spl(al, 1)
    d2Fs = lambda al: spl(al, 2)

    # SI conversion
    e_unit = SI["Hc0"]**2/SI["mu0"] * SI["xi0"]        # J/m^2 per code unit
    d_ho_SI = PHYS["d_ho"]*SI["xi0"]
    a0 = np.deg2rad(SI["alpha0_deg"])
    Kd = SI["K_eff"]*d_ho_SI                            # J/m^2

    Ftot = lambda al: Fs(al)*e_unit + Kd*np.sin(al-a0)**2
    from scipy.optimize import minimize_scalar
    m = minimize_scalar(Ftot, bounds=(np.deg2rad(30), np.deg2rad(90)),
                        method="bounded")
    a_star = m.x
    tau = -dFs(a0)*e_unit                               # J/m^2 per rad
    Ban = tau/(SI["Ms"]*d_ho_SI*np.sin(a0))             # Tesla (axial field)
    Ksc = 0.5*d2Fs(a0)*e_unit/d_ho_SI                   # J/m^3
    dalpha = a_star - a0

    summary = dict(
        F_at_alpha0_code=float(Fs(a0)),
        energy_unit_J_per_m2=e_unit,
        dF_dalpha_at_alpha0_code=float(dFs(a0)),
        torque_J_per_m2=float(tau),
        B_anisotropy_T=float(Ban),
        B_anisotropy_Oe=float(Ban*1e4),
        K_SC_J_per_m3=float(Ksc),
        alpha_eq_deg=float(np.rad2deg(a_star)),
        tilt_shift_deg=float(np.rad2deg(dalpha)),
        tilt_shift_urad=float(dalpha*1e6),
        params=dict(PHYS=PHYS, NUM=NUM, SI=SI),
    )
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "params"},
                     indent=2))

    # ---------------------------------------- report-quality plots (styled)
    aa = np.linspace(0, np.pi/2, 400)
    plot_free_energy_vs_angle(a, F, Fx, aa, Fs, SI["alpha0_deg"])
    plot_minimisation_about_alpha0(aa, Fs, e_unit, Kd, a0, a_star)
    # Restore rc defaults so the diagnostic 2x2 figure below is not affected
    plt.rcdefaults()

    # ------------------------------------------------------------- figures
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2))
    ax = axs[0, 0]
    ax.plot(a, F*1e3, "o", color="C0", label=r"$F_S(\alpha)$ (computed)")
    aa = np.linspace(0, np.pi/2, 200)
    ax.plot(a, Fx*1e3, "s", ms=4, color="C3", alpha=0.8,
            label="tail-extrapolated (used for analysis)")
    ax.plot(np.rad2deg(aa), Fs(aa)*1e3, "-", color="C3", alpha=0.7,
            label="spline fit")
    ax.axvline(SI["alpha0_deg"], color="k", ls=":", lw=1)
    ax.set_xlabel(r"Ho cone angle $\alpha$ (deg, from film normal)")
    ax.set_ylabel(r"$F_S$  [$10^{-3}\,N_0\Delta_{00}^2\xi_0$]")
    ax.set_title("Superconducting free energy vs cone angle")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    EK = Kd*np.sin(aa-a0)**2
    ax.plot(np.rad2deg(aa), (Fs(aa)-Fs(a0))*e_unit*1e6, label=r"$F_S-F_S(\alpha_0)$")
    ax.plot(np.rad2deg(aa), EK*1e6, label=r"$K_{\rm eff}d_{\rm Ho}\sin^2(\alpha-\alpha_0)$")
    ax.plot(np.rad2deg(aa), ((Fs(aa)-Fs(a0))*e_unit+EK)*1e6, "k",
            label=r"$F_{\rm tot}$")
    ax.axvline(np.rad2deg(a_star), color="r", ls="--", lw=1,
               label=fr"$\alpha^*={np.rad2deg(a_star):.3f}^\circ$")
    ax.set_xlim(SI["alpha0_deg"]-8, SI["alpha0_deg"]+8)
    span = max(abs((Fs(aa)-Fs(a0))*e_unit*1e6).max(), 1e-3)
    ax.set_ylim(-2*span, 10*span)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"energy per area [$\mu$J/m$^2$]")
    ax.set_title("Free-energy minimisation about $\\alpha_0$")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.plot(a, Dm, "o-", color="C2")
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$\langle\Delta\rangle_{\rm Nb}/\Delta_{00}$")
    ax.set_title("Self-consistent gap vs cone angle")
    ax2 = ax.twinx()
    ax2.plot(a, fperp, "s--", color="C3")
    ax2.set_ylabel(r"max $|f_{t\perp h}|$ in Ho (lowest $\omega_n$)",
                   color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    ax = axs[1, 1]
    for adeg, cc in [(a[0], "C0"), (SI["alpha0_deg"], "C3")]:
        f = NPZ.replace(".npz", f"_a{int(adeg):03d}.npz")
        if os.path.exists(f):
            d = np.load(f)
            ax.plot(d["x"], d["Delta"], color=cc, label=fr"$\Delta(x)$, $\alpha={int(adeg)}^\circ$")
            ax.plot(d["x"], np.abs(d["f0"]), color=cc, ls="--", alpha=0.7,
                    label=fr"$|f_0|$, $\alpha={int(adeg)}^\circ$")
            ax.plot(d["x"], np.linalg.norm(d["ft"], axis=1), color=cc, ls=":",
                    label=fr"$|\mathbf{{f}}_t|$, $\alpha={int(adeg)}^\circ$")
    for xe in np.cumsum([PHYS["d_py"], PHYS["d_nb"]]):
        ax.axvline(xe, color="k", lw=0.5)
    ax.set_xlabel(r"$x/\xi_0$  (Py | Nb | Ho)")
    ax.set_ylabel(r"$\Delta/\Delta_{00}$,  pair amplitudes")
    ax.set_title("Profiles (lowest Matsubara frequency)")
    ax.legend(fontsize=7)

    fig.suptitle("SC-induced anisotropy on Ho in Py/Nb/Ho "
                 f"(t={PHYS['t']}, $h_{{Ho}}$={PHYS['h_ho']}$\\Delta_{{00}}$, "
                 f"$h_{{Py}}$={PHYS['h_py']}$\\Delta_{{00}}$)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "anisotropy_results.png"), dpi=160)
    print("wrote anisotropy_results.png, summary.json")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        run_sweep(int(sys.argv[1]), int(sys.argv[2]))
    elif len(sys.argv) == 2 and sys.argv[1] == "analyse":
        analyse()
    else:
        run_sweep()
        analyse()