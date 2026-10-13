"""
plot_components.py
==================
Spatial distribution of the superconducting pair correlations across the
Py / Nb / Ho stack, decomposed into:

  singlet        f0(x)
  m = 0 triplet  f_par(x)  = f_t . h^(x)          (short-range in a magnet)
  equal-spin     f_perp(x) = |f_t - (f_t.h^) h^|   (long-range in a magnet)

The quantisation axis is the LOCAL exchange-field direction h^(x), which is
what determines short- vs long-rangedness (a triplet component parallel to
the local field is the m=0 pair and dephases over xi_h = sqrt(hbarD/2h);
perpendicular components are equal-spin pairs and decay only over the
thermal length xi_w = sqrt(hbarD/2 w_n)). Inside Nb there is no exchange
field, so the split is not defined there and we show the total |f_t|.

Amplitudes are shown at the lowest Matsubara frequency w_0 = pi k_B T
(standard practice for odd-frequency components, whose sum over all w_n
vanishes by antisymmetry). Uses the profiles saved by run_anisotropy.py;
re-run that first if the .npz files are missing.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_anisotropy import PHYS, NUM, SI

XI0_NM = SI["xi0"] * 1e9   # xi0 expressed in nm (for axis conversion)
from usadel_free_energy import make_pynbho_stack

OUT = os.path.dirname(os.path.abspath(__file__))

# =========================
# Plot design settings (matches the rest of the report's figures)
# =========================

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 13,
        "axes.linewidth": 1.1,
        "axes.labelsize": 15,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.fontsize": 11,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    }
)

# Layer shading: light, but still colourful, tints of the report's
# established Nb blue / Ho lavender / Py teal palette - pale enough not to
# compete with the data curves, but not washed-out grey.
shade_py = "#cdf0e2"   # light teal
shade_nb = "#cde8fb"   # light blue
shade_ho = "#e8d6f7"   # light lavender

# Accent colours for the plotted pair-amplitude curves themselves (physical
# quantities, not tied to a material, so they get their own palette).
col_singlet = "#20304a"   # dark charcoal-navy
col_m0 = "#c8791a"        # muted amber/ochre
col_eq_spin = "#a3305c"   # muted crimson

def load(alpha_deg):
    d = np.load(os.path.join(OUT, f"profiles_a{int(alpha_deg):03d}.npz"))
    stk = make_pynbho_stack(alpha=np.deg2rad(alpha_deg), **PHYS, **NUM)
    x = d["x"]
    # with all D equal (default), u == x
    h = stk.h_of_u(x)                       # (3, n)
    hn = np.linalg.norm(h, axis=0)
    mag = hn > 0.5 * min(PHYS["h_py"], PHYS["h_ho"])   # inside a magnet
    hdir = np.where(hn > 1e-12, h / np.maximum(hn, 1e-12), 0.0)
    ft = d["ft"]                            # (n, 3) complex, lowest w_n
    f_par = np.einsum("nk,kn->n", ft, hdir)
    f_perp_vec = ft - f_par[:, None] * hdir.T
    return dict(x=x * XI0_NM, Delta=d["Delta"], f0=d["f0"], ft=ft,
                f_par=np.abs(f_par), f_perp=np.linalg.norm(f_perp_vec, axis=1),
                f_tot=np.linalg.norm(ft, axis=1), mag=mag)

def shade(ax):
    e = np.cumsum([0, PHYS["d_py"], PHYS["d_nb"], PHYS["d_ho"]]) * XI0_NM
    for (a, b, c) in [(e[0], e[1], shade_py),
                      (e[1], e[2], shade_nb),
                      (e[2], e[3], shade_ho)]:
        ax.axvspan(a, b, color=c, lw=0, zorder=0)
    for boundary in e[1:-1]:
        ax.axvline(boundary, color="0.6", lw=0.8, ls=(0, (4, 2)), zorder=1)
    for xx, lab in [(e[0], "Py"), (e[1], "Nb"), (e[2], "Ho")]:
        ax.text(xx + 0.04 * XI0_NM, 0.965, lab, transform=ax.get_xaxis_transform(),
                fontsize=10, va="top", color="0.3")

def draw(ax, D, adeg, log=False):
    x, mag = D["x"], D["mag"]
    # threshold anywhere inside Nb separates Py (left magnet) from Ho (right)
    x_mid_nb = (PHYS["d_py"] + 0.5 * PHYS["d_nb"]) * XI0_NM
    ax.plot(x, np.abs(D["f0"]), color=col_singlet, lw=2, label=r"Singlet")
    # split curves only where the axis is defined (inside magnets)
    for msk, ls in [(mag & (x < x_mid_nb), "-"), (mag & (x > x_mid_nb), "-")]:
        if msk.any():
            ax.plot(x[msk], D["f_par"][msk], color=col_m0, ls=ls, lw=2)
            ax.plot(x[msk], D["f_perp"][msk], color=col_eq_spin, ls=ls, lw=2)
    ax.plot(x[~mag], D["f_tot"][~mag], color="0.35", ls=":", lw=1.8)
    ax.plot([], [], color=col_m0, lw=2, label=r"$m=0$ triplet (short-range)")
    ax.plot([], [], color=col_eq_spin, lw=2, label=r"Equal-spin triplet (long-range)")
    ax.plot([], [], color="0.35", ls=":", lw=1.8, label=r"Total triplet (Nb)")
    ax.plot(x, D["Delta"], color="0.2", lw=1, alpha=0.5)
    ax.plot([], [], color="0.2", lw=1, alpha=0.5, label=r"$\Delta(x)/\Delta_{00}$")
    shade(ax)
    ax.text(0.96, 0.94, fr"$\alpha = {adeg}^\circ$", transform=ax.transAxes,
            ha="right", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.5", linewidth=0.8, alpha=0.92))
    if log:
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 0.7)
    else:
        ax.set_ylim(0, None)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel(r"$x$ (nm)")

fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
for col, adeg in enumerate([80, 0]):
    D = load(adeg)
    draw(axs[col], D, adeg, log=False)
axs[0].set_ylabel(r"Pair amplitude ($\omega_0$)")

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=11, ncol=1, frameon=True,
          facecolor="white", edgecolor="0.7", framealpha=0.95,
          loc="center left", bbox_to_anchor=(1.0, 0.5),
          borderpad=0.8, labelspacing=0.9, handlelength=1.8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "pair_components.png"), dpi=450, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "pair_components.svg"), bbox_inches="tight")
print("wrote pair_components.png")