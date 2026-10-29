"""Teaching figure: how loop strength reshapes the landscape *and* the switching.

One row per value of the loop strength ``a``, with everything else held fixed:

    r_on = r_off = 1   ->  r = r_on + r_off = 2, so the pitchfork sits at a = 2
    w*_int, sigma_noise fixed  ->  identical effective noise w*sigma in every row
    identical RNG seed         ->  identical sequence of random kicks in every row

So the *only* thing that differs down the column is the shape of the landscape.
Any difference in the traces is caused by that shape and nothing else.

  left   energy landscape V(L): the wells, and the barrier between them. All rows
         share the same axes, so the barrier visibly grows down the column.
  right  one simulated trajectory L(t), shaded by the MAP percept sign(L) --
         i.e. what the observer reports seeing at each moment.

The reason for putting them side by side: the barrier height dV on the left is
what sets the dwell time on the right. Escape over a barrier is a Kramers
process, rate ~ exp(-2*dV / (w*sigma)^2), so dV enters *inside an exponential* --
a modest change in a produces an enormous change in how often the percept flips.

Reading down the rows:

  a < r    Monostable: one well, no barrier. There is no second state to switch
           to. sign(L) still flickers, but that is an undecided system grazing
           zero, not a percept alternating. a = 0 is the ideal Bayesian observer,
           and it does NOT produce bistable perception -- the paper's central
           negative result (Leptourgos et al. 2020, Results).
  a > r    Two wells. Plateaus appear and perception becomes discrete.
  a >> r   Wells deep enough that escapes are rare; perception nearly freezes.

Two distinct properties of a well are annotated, because they are easy to
conflate and they answer different questions:

    dV  = V(barrier) - V(well)   barrier height  -> how long until it switches
    tau = 1 / |f'(L*)|           floor curvature -> how fast it settles after a
                                                    small kick

Not a paper figure; written to make the landscape/behaviour link visible.
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import savefig
from dci import (
    ReducedParams,
    simulate,
    committed_state,
    switches_min_dwell,
    potential,
    drift_deriv,
    find_fixed_points,
)

DT = 0.01
T_TOTAL = 300.0                    # time units shown in each trace
N_STEPS = int(T_TOTAL / DT)

R_ON = R_OFF = 1.0                 # r = 2  ->  pitchfork threshold at a = 2
R_SUM = R_ON + R_OFF

W_INT = 0.8                        # sensory gain w*_int
SIGMA = 1.5                        # sensory noise; effective noise = W_INT*SIGMA

A_VALUES = (0.0, 1.5, 2.5, 3.5, 4.5)

SWITCH_TAU = 0.1                   # min-dwell debounce, as in fig2.py
SEED = 7                           # same noise realization in every row

L_GRID = np.linspace(-4.0, 4.0, 801)
V_YLIM = (-0.25, 5.0)
L_YLIM = (-4.5, 4.5)

C_SFA = "tab:red"
C_SFB = "tab:blue"


def _geometry(p: ReducedParams):
    """Landscape geometry: wells, barrier top, barrier height dV, settling time tau."""
    fps = find_fixed_points(p)
    stable = [L for L, s in fps if s]
    unstable = [L for L, s in fps if not s]

    # Exactly at the bifurcation (a = r) we have f'(0) = a - r = 0: the origin is
    # *marginally* stable. find_fixed_points' strict f' < 0 test therefore files
    # it as unstable and `stable` comes back empty. Linearisation genuinely gives
    # no time constant there -- tau -> infinity (critical slowing down); what
    # actually restores L is the cubic term -(r/6)L^3, which the tangent misses.
    marginal = (not stable) and len(unstable) == 1

    if marginal:
        tau = np.inf
    elif stable:
        tau = 1.0 / abs(drift_deriv(stable[0], p))
    else:
        tau = None

    dV = None
    if len(stable) >= 2 and unstable:
        pts = np.array(stable + [unstable[0]])
        V = potential(pts, p, shift=False)
        dV = float(V[-1] - np.min(V[:-1]))

    return dict(stable=stable, unstable=unstable, dV=dV, tau=tau,
                marginal=marginal)


def main():
    fig, axs = plt.subplots(
        len(A_VALUES), 2,
        figsize=(13.5, 2.7 * len(A_VALUES)),
        gridspec_kw=dict(width_ratios=[1.0, 2.1]),
    )

    t = np.arange(N_STEPS) * DT

    for row, a in enumerate(A_VALUES):
        p = ReducedParams(r_on=R_ON, r_off=R_OFF, a=a, w=W_INT,
                          mu_noise=0.0, sigma_noise=SIGMA)
        g = _geometry(p)
        bistable = g["dV"] is not None

        # ---------------- left: energy landscape ----------------
        ax = axs[row, 0]
        V = potential(L_GRID, p, shift=True)
        ax.plot(L_GRID, V, color="0.15", lw=2)
        ax.axvline(0.0, color="0.8", lw=0.8, zorder=0)

        for Ls in g["stable"]:
            ax.plot(Ls, np.interp(Ls, L_GRID, V), "o", color="tab:green",
                    ms=8, zorder=3)
        for Lu in g["unstable"]:
            ax.plot(Lu, np.interp(Lu, L_GRID, V), "o", mfc="white",
                    mec="tab:orange", mew=2, ms=8, zorder=3)

        if bistable:
            Lw = g["stable"][-1]
            y_well = np.interp(Lw, L_GRID, V)
            y_top = np.interp(g["unstable"][0], L_GRID, V)
            ax.annotate("", xy=(Lw, y_top), xytext=(Lw, y_well),
                        arrowprops=dict(arrowstyle="<->", color="tab:purple", lw=1.6))
            # anchor the label to the barrier top, not the arrow midpoint: for a
            # shallow barrier the midpoint sits on top of the well marker.
            ax.text(Lw + 0.18, y_top, f"$\\Delta V$ = {g['dV']:.2f}",
                    color="tab:purple", fontsize=9, va="bottom", ha="left")
            label = (f"wells at $L^*$ = $\\pm${abs(Lw):.2f}\n"
                     f"$\\tau$ = {g['tau']:.2f}")
        else:
            label = f"monostable\n$\\tau$ = {g['tau']:.2f}"

        ax.text(0.03, 0.96, label, transform=ax.transAxes, fontsize=9,
                va="top", ha="left",
                bbox=dict(fc="white", ec="0.8", alpha=0.9, boxstyle="round,pad=0.3"))
        ax.set_xlim(L_GRID[0], L_GRID[-1])
        ax.set_ylim(*V_YLIM)
        ax.set_ylabel("Potential $V(L)$")
        if row == len(A_VALUES) - 1:
            ax.set_xlabel("Confidence $L$")
        if row == 0:
            ax.set_title("Energy landscape", fontsize=11)

        # row label: the one parameter that changes
        rel = "<" if a < R_SUM else ("=" if a == R_SUM else ">")
        ax.text(-0.30, 0.5, f"$a$ = {a:g}\n($a$ {rel} $r$ = {R_SUM:g})",
                transform=ax.transAxes, fontsize=12, va="center", ha="center",
                rotation=0)

        # ---------------- right: trajectory + percept ----------------
        ax = axs[row, 1]
        L = simulate(p, N_STEPS, dt=DT, L0=0.0, n_traj=1,
                     rng=np.random.default_rng(SEED))
        s = committed_state(L, threshold=0.0)

        ax.fill_between(t, L_YLIM[0], L_YLIM[1], where=s > 0,
                        color=C_SFA, alpha=0.11, lw=0, step="mid")
        ax.fill_between(t, L_YLIM[0], L_YLIM[1], where=s < 0,
                        color=C_SFB, alpha=0.11, lw=0, step="mid")
        ax.plot(t, L, color="0.15", lw=0.5)
        ax.axhline(0.0, color="0.4", ls="--", lw=0.9)
        for Ls in g["stable"]:
            if abs(Ls) > 1e-9:
                ax.axhline(Ls, color="tab:green", ls=":", lw=1.0)

        n_down, n_up = switches_min_dwell(L, DT, tau=SWITCH_TAU)
        n_sw = n_down + n_up
        # T/n is only a usable dwell estimate with enough completed phases; with
        # 0-1 switches the window is censored and only a lower bound is honest.
        if n_sw >= 3:
            dwell_txt = f"mean dwell $\\approx$ {T_TOTAL / n_sw:.1f}"
        else:
            dwell_txt = f"dwell $>$ {T_TOTAL:.0f} (window too short to measure)"
        kind = "" if bistable else "  [zero-crossings, not percepts]"
        ax.text(0.012, 0.96,
                f"switches = {n_sw}   {dwell_txt}{kind}",
                transform=ax.transAxes, fontsize=9, va="top", ha="left",
                bbox=dict(fc="white", ec="0.8", alpha=0.9, boxstyle="round,pad=0.3"))

        ax.set_xlim(0, T_TOTAL)
        ax.set_ylim(*L_YLIM)
        ax.set_ylabel("$L(t)$")
        if row == len(A_VALUES) - 1:
            ax.set_xlabel("Time")
        if row == 0:
            ax.set_title("Simulated trace, shaded by reported percept "
                         "(red = SFA, blue = SFB)", fontsize=11)

    fig.suptitle(
        "Loop strength $a$ reshapes the landscape, and the landscape sets the switching\n"
        f"(everything else fixed: $r_{{on}}$ = $r_{{off}}$ = 1, "
        f"$w^*_{{int}}$ = {W_INT:g}, $\\sigma_{{noise}}$ = {SIGMA:g}, "
        "identical noise realization in every row)",
        fontsize=13, y=0.995)
    fig.tight_layout(rect=[0.02, 0, 1, 0.965])
    savefig(fig, "fig_landscape_switching.png")


if __name__ == "__main__":
    main()
