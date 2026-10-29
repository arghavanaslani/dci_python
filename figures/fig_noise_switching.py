"""Teaching figure: the *other* knob on switching -- sensory noise.

Companion to fig_landscape_switching.py, which varied the loop strength a and
held the noise fixed. Here it is the reverse: the loop strength (and therefore
the entire landscape) is IDENTICAL in every row, and only sigma_noise changes.

    a = 3.5, r_on = r_off = 1   ->  the same two wells at L* = +/-1.93 and the
                                    same barrier dV = 1.48 in every single row
    identical RNG seed          ->  the same underlying sequence of random draws

The left column therefore shows the same curve five times. That is the point:
behaviour on the right changes just as drastically as it did when the landscape
was reshaped, even though nothing about the landscape moved.

What *is* drawn differently on the left is the shaded band, which marks the
noise energy

    D = sigma_eff^2 / 2,        sigma_eff = w*_int * sigma_noise

the scale of the random energy available to climb the hill. Kramers' escape rate
goes as exp(-dV / D), so the controlling quantity is the *ratio* dV/D -- the
barrier measured in units of noise. Watch the band rise up a fixed hill:

    dV/D >> 1   band far below the barrier top; escapes are rare; percept frozen
    dV/D ~  1   band reaches the barrier top; escapes are frequent
    dV/D <  1   noise swamps the landscape; wells stop mattering; percept mush

Taken together with fig_landscape_switching.py, this is where identifiability
dies: dV is set by a and D is set by w*_int*sigma_noise, but only their ratio
reaches behaviour. A deep well shaken hard and a shallow well shaken gently give
the same switch rate, so counting switches under continuous viewing cannot
separate the two. Breaking that tie is the entire reason the parameter-recovery
experiment (S4 Text, recovery.py) uses intermittent presentation: during a blank
the sensory term is switched off, so what happens across the gap depends on the
landscape alone.

Not a paper figure; written to make the noise/landscape trade-off visible.
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
T_TOTAL = 300.0
N_STEPS = int(T_TOTAL / DT)

R_ON = R_OFF = 1.0                 # r = 2  ->  pitchfork threshold at a = 2
R_SUM = R_ON + R_OFF

A_FIXED = 3.5                      # well past threshold; landscape never changes
W_INT = 0.8                        # sensory gain w*_int

SIGMA_VALUES = (0.8, 1.2, 1.6, 2.2, 3.0)

SWITCH_TAU = 0.1
SEED = 7

L_GRID = np.linspace(-4.0, 4.0, 801)
V_YLIM = (-0.2, 3.6)
L_YLIM = (-4.5, 4.5)

C_SFA = "tab:red"
C_SFB = "tab:blue"


def _geometry(p: ReducedParams):
    """Wells, barrier top, barrier height dV, settling time tau."""
    fps = find_fixed_points(p)
    stable = [L for L, s in fps if s]
    unstable = [L for L, s in fps if not s]
    tau = 1.0 / abs(drift_deriv(stable[0], p)) if stable else None
    dV = None
    if len(stable) >= 2 and unstable:
        V = potential(np.array(stable + [unstable[0]]), p, shift=False)
        dV = float(V[-1] - np.min(V[:-1]))
    return dict(stable=stable, unstable=unstable, dV=dV, tau=tau)


def main():
    fig, axs = plt.subplots(
        len(SIGMA_VALUES), 2,
        figsize=(13.5, 2.7 * len(SIGMA_VALUES)),
        gridspec_kw=dict(width_ratios=[1.0, 2.1]),
    )

    t = np.arange(N_STEPS) * DT

    for row, sigma in enumerate(SIGMA_VALUES):
        p = ReducedParams(r_on=R_ON, r_off=R_OFF, a=A_FIXED, w=W_INT,
                          mu_noise=0.0, sigma_noise=sigma)
        g = _geometry(p)
        dV = g["dV"]

        sigma_eff = W_INT * sigma        # the only combination that reaches the SDE
        D = 0.5 * sigma_eff ** 2         # noise energy scale
        ratio = dV / D

        # ---------------- left: landscape (identical) + noise band ----------------
        ax = axs[row, 0]
        V = potential(L_GRID, p, shift=True)
        ax.axhspan(0.0, D, color="tab:orange", alpha=0.16, lw=0, zorder=0)
        ax.axhline(D, color="tab:orange", lw=1.4, ls="-")
        ax.axhline(dV, color="tab:purple", lw=1.2, ls="--")
        ax.plot(L_GRID, V, color="0.15", lw=2, zorder=2)
        ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)

        for Ls in g["stable"]:
            ax.plot(Ls, np.interp(Ls, L_GRID, V), "o", color="tab:green",
                    ms=8, zorder=3)
        for Lu in g["unstable"]:
            ax.plot(Lu, np.interp(Lu, L_GRID, V), "o", mfc="white",
                    mec="tab:orange", mew=2, ms=8, zorder=3)

        ax.text(-3.85, D, f" noise $D$ = {D:.2f}", color="tab:orange",
                fontsize=8.5, va="bottom", ha="left")
        ax.text(3.85, dV, f"barrier $\\Delta V$ = {dV:.2f} ", color="tab:purple",
                fontsize=8.5, va="bottom", ha="right")
        ax.text(0.03, 0.96, f"$\\Delta V / D$ = {ratio:.2f}",
                transform=ax.transAxes, fontsize=10, va="top", ha="left",
                bbox=dict(fc="white", ec="0.8", alpha=0.9, boxstyle="round,pad=0.3"))

        ax.set_xlim(L_GRID[0], L_GRID[-1])
        ax.set_ylim(*V_YLIM)
        ax.set_ylabel("Potential $V(L)$")
        if row == len(SIGMA_VALUES) - 1:
            ax.set_xlabel("Confidence $L$")
        if row == 0:
            ax.set_title("Same landscape every row; only the noise band rises",
                         fontsize=11)

        ax.text(-0.30, 0.5,
                f"$\\sigma_{{noise}}$ = {sigma:g}\n($\\sigma_{{eff}}$ = {sigma_eff:.2f})",
                transform=ax.transAxes, fontsize=12, va="center", ha="center")

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
            ax.axhline(Ls, color="tab:green", ls=":", lw=1.0)

        n_down, n_up = switches_min_dwell(L, DT, tau=SWITCH_TAU)
        n_sw = n_down + n_up
        if n_sw >= 3:
            dwell_txt = f"mean dwell $\\approx$ {T_TOTAL / n_sw:.1f}"
        else:
            dwell_txt = f"dwell $>$ {T_TOTAL:.0f} (window too short to measure)"
        ax.text(0.012, 0.96, f"switches = {n_sw}   {dwell_txt}",
                transform=ax.transAxes, fontsize=9, va="top", ha="left",
                bbox=dict(fc="white", ec="0.8", alpha=0.9, boxstyle="round,pad=0.3"))

        ax.set_xlim(0, T_TOTAL)
        ax.set_ylim(*L_YLIM)
        ax.set_ylabel("$L(t)$")
        if row == len(SIGMA_VALUES) - 1:
            ax.set_xlabel("Time")
        if row == 0:
            ax.set_title("Simulated trace, shaded by reported percept "
                         "(red = SFA, blue = SFB)", fontsize=11)

    fig.suptitle(
        "Noise sets the switching too: the landscape never moves, only the shaking\n"
        f"(fixed: $a$ = {A_FIXED:g} > $r$ = {R_SUM:g}, wells at $L^*$ = $\\pm$1.93, "
        f"$\\Delta V$ = 1.48, $w^*_{{int}}$ = {W_INT:g}, identical noise realization)",
        fontsize=13, y=0.995)
    fig.tight_layout(rect=[0.02, 0, 1, 0.965])
    savefig(fig, "fig_noise_switching.png")


if __name__ == "__main__":
    main()
