"""Teaching figure: five samples taken across the iso-rate contours.

fig_regime_map.py showed that a and sigma_eff reach behaviour only through the
Kramers exponent dV/D, so its white contours are curves of constant predicted
rate. This takes one cell from each of five of them -- dV/D = 8, 4, 2, 1, 0.1 --
and shows what a trace on each actually looks like.

Cells are hand-picked: a is chosen (round values), dV is then computed exactly
from find_fixed_points/potential, and sigma_eff = sqrt(2*dV/c) places the cell on
the target contour. a falls and sigma_eff rises down the rows, so the samples
walk across the fan rather than along one ray. The bottom row is forced close to
the bifurcation -- dV/D = 0.1 needs an almost flat barrier.

Per row: the landscape with its noise band, one T = 5000 trace, and the
dominance-duration histogram. Switches use the same amplitude hysteresis as the
regime map (commit only past +/-THRESHOLD), so the measured dwell is directly
comparable to that figure's colour scale.

Two quantities called "dwell" appear and are NOT the same:

    measured  = observed time / switches counted   -- depends on THRESHOLD
    Kramers   = 1 / (prefactor * exp(-dV/D))       -- depends only on V and D

They agree while THRESHOLD sits well inside the wells and escapes are rare. The
bottom row breaks all three conditions at once (dV/D << 1 is outside Kramers'
range of validity, the prefactor collapses as a -> r, and THRESHOLD = 1 sits at
84% of the way out to L* = 1.20), which is why it is not a percept process.

Not a paper figure; written to inspect the switching statistics along the fan.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from _common import savefig
from dci import (
    ReducedParams,
    simulate,
    switch_indices,
    phase_durations,
    potential,
    drift_deriv,
    find_fixed_points,
)
import fig_regime_map as rmap

DT = 0.01
T_TOTAL = 300.0
T_BURN = 30.0
N_STEPS = int((T_TOTAL + T_BURN) / DT)
N_BURN = int(T_BURN / DT)

R_ON = R_OFF = 1.0
R_SUM = R_ON + R_OFF

THRESHOLD = 1.0                    # same commitment rule as the regime map
SEED = 11

# (target dV/D, loop strength a); sigma_eff is solved for below
SAMPLES = ((2.0, 4.0), (1.0, 3.5))

L_GRID = np.linspace(-3.5, 3.5, 701)
V_YLIM = (-0.4, 7.0)
L_YLIM = (-5.0, 5.0)

C_SFA = "tab:red"
C_SFB = "tab:blue"


def barrier(p: ReducedParams):
    """(wells, barrier top, dV) for a bistable landscape."""
    fps = find_fixed_points(p)
    stable = [L for L, s in fps if s]
    unstable = [L for L, s in fps if not s]
    V = potential(np.array(stable + unstable[:1]), p, shift=False)
    return stable, unstable, float(V[-1] - np.min(V[:-1]))


def build_rows():
    """Solve each sample's sigma_eff, then simulate and measure it."""
    rows = []
    for k, (c_target, a) in enumerate(SAMPLES):
        p0 = ReducedParams(r_on=R_ON, r_off=R_OFF, a=a, w=1.0)
        stable, unstable, dV = barrier(p0)

        sigma_eff = float(np.sqrt(2.0 * dV / c_target))
        D = 0.5 * sigma_eff ** 2
        p = ReducedParams(r_on=R_ON, r_off=R_OFF, a=a, w=1.0,
                          mu_noise=0.0, sigma_noise=sigma_eff)

        L_star = max(stable)
        curv_well = -drift_deriv(L_star, p)          # V''(L*)
        curv_top = abs(-drift_deriv(unstable[0], p))  # |V''(0)|
        prefactor = np.sqrt(curv_well * curv_top) / (2.0 * np.pi)
        tau = 1.0 / abs(drift_deriv(L_star, p))

        L = simulate(p, N_STEPS, dt=DT, L0=0.0, n_traj=1,
                     rng=np.random.default_rng(SEED + k))[N_BURN:]
        n_sw = switch_indices(L, THRESHOLD).size
        durs = phase_durations(L, DT, THRESHOLD)

        rows.append(dict(
            c=dV / D, a=a, sigma_eff=sigma_eff, dV=dV, D=D,
            stable=stable, unstable=unstable, L_star=L_star, tau=tau,
            rate_kramers=prefactor * np.exp(-dV / D),
            rate_meas=n_sw / T_TOTAL, n_sw=n_sw, durs=durs, L=L, p=p,
        ))
    return rows


def print_table(rows):
    head = (f"{'dV/D':>6} {'a':>5} {'sig_eff':>8} {'dV':>6} {'D':>6} {'L*':>6} "
            f"{'tau':>6} {'n_sw':>6} {'dwell_meas':>11} {'dwell_kram':>11} "
            f"{'dwell/tau':>10} {'CV':>6}")
    print(head)
    print("-" * len(head))
    for r in rows:
        dwell_meas = T_TOTAL / r["n_sw"] if r["n_sw"] else np.inf
        dwell_kram = 1.0 / r["rate_kramers"]
        cv = (r["durs"].std() / r["durs"].mean()) if r["durs"].size > 1 else np.nan
        print(f"{r['c']:6.2f} {r['a']:5.1f} {r['sigma_eff']:8.3f} {r['dV']:6.2f} "
              f"{r['D']:6.2f} {r['L_star']:6.2f} {r['tau']:6.2f} {r['n_sw']:6d} "
              f"{dwell_meas:11.1f} {dwell_kram:11.1f} "
              f"{dwell_meas / r['tau']:10.1f} {cv:6.2f}")


def draw_map(ax, rows):
    rate = rmap.sweep()
    dV = rmap.geometry()
    ratio_map = dV[:, None] / (0.5 * rmap.SIG_GRID[None, :] ** 2)
    X, Y = np.meshgrid(rmap.A_GRID, rmap.SIG_GRID, indexing="ij")

    pc = ax.pcolormesh(X, Y, np.ma.masked_where(rate <= 0, rate),
                       norm=LogNorm(), cmap="viridis", shading="auto")
    ax.figure.colorbar(pc, ax=ax, label="alternation rate (switches / unit time)")

    cs = ax.contour(X, Y, ratio_map, levels=[1, 2, 4, 8], colors="white",
                    linewidths=1.1, linestyles="--")
    ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=8)

    ax.axvline(R_SUM, color="tab:red", lw=2)
    ax.axvspan(rmap.A_GRID[0], R_SUM, color="tab:red", alpha=0.12, lw=0)

    for i, r in enumerate(rows, start=1):
        ax.plot(r["a"], r["sigma_eff"], "o", mfc="white", mec="k", mew=1.8, ms=11,
                zorder=5)
        ax.text(r["a"], r["sigma_eff"], str(i), fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=6)
        ax.annotate(f"$\\Delta V/D$ = {r['c']:g}",
                    xy=(r["a"], r["sigma_eff"]), xytext=(8, 10),
                    textcoords="offset points", color="white", fontsize=9)

    ax.set_xlabel("loop strength $a$        ($r$ = 2)")
    ax.set_ylabel("effective noise $\\sigma_{eff}$")
    ax.set_xlim(rmap.A_GRID[0], rmap.A_GRID[-1])
    ax.set_ylim(rmap.SIG_GRID[0], rmap.SIG_GRID[-1])
    ax.set_title("Where the five samples sit on the regime map", fontsize=11)


def main():
    rows = build_rows()
    print_table(rows)

    n = len(rows)
    fig = plt.figure(figsize=(15.5, 10.0))
    gs = fig.add_gridspec(n + 1, 3, height_ratios=[3.0] + [1.0] * n,
                          width_ratios=[1.0, 2.2, 1.0], hspace=0.42, wspace=0.28,
                          left=0.07, right=0.98, top=0.945, bottom=0.035)

    draw_map(fig.add_subplot(gs[0, :2]), rows)
    fig.add_subplot(gs[0, 2]).axis("off")

    t = np.arange(rows[0]["L"].size) * DT

    for i, r in enumerate(rows):
        # ---------------- left: landscape + noise band ----------------
        ax = fig.add_subplot(gs[i + 1, 0])
        V = potential(L_GRID, r["p"], shift=True)
        ax.axhspan(0.0, r["D"], color="tab:orange", alpha=0.16, lw=0, zorder=0)
        ax.axhline(r["D"], color="tab:orange", lw=1.4)
        ax.axhline(r["dV"], color="tab:purple", lw=1.2, ls="--")
        ax.plot(L_GRID, V, color="0.15", lw=2, zorder=2)
        ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)

        for Ls in r["stable"]:
            ax.plot(Ls, np.interp(Ls, L_GRID, V), "o", color="tab:green", ms=8,
                    zorder=3)
        for Lu in r["unstable"]:
            ax.plot(Lu, np.interp(Lu, L_GRID, V), "o", mfc="white",
                    mec="tab:orange", mew=2, ms=8, zorder=3)
        for sgn in (-1, 1):
            ax.axvline(sgn * THRESHOLD, color="0.45", ls=":", lw=1.0, zorder=1)

        lbl = dict(fontsize=8.5, va="bottom",
                   bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
        ax.text(-3.4, r["D"], f"$D$ = {r['D']:.2f}", color="tab:orange",
                ha="left", **lbl)
        ax.text(3.4, r["dV"], f"$\\Delta V$ = {r['dV']:.2f}", color="tab:purple",
                ha="right", **lbl)
        ax.text(0.03, 0.96, f"$\\Delta V/D$ = {r['c']:.2f}", transform=ax.transAxes,
                fontsize=10, va="top", ha="left",
                bbox=dict(fc="white", ec="0.8", alpha=0.9, boxstyle="round,pad=0.3"))

        ax.set_xlim(L_GRID[0], L_GRID[-1])
        ax.set_ylim(*V_YLIM)
        ax.set_ylabel("$V(L)$")
        if i == n - 1:
            ax.set_xlabel("Confidence $L$")
        if i == 0:
            ax.set_title("Landscape (dotted = commitment threshold)", fontsize=10)

        ax.text(-0.34, 0.5,
                f"{i + 1}\n$a$ = {r['a']:g}\n$\\sigma_{{eff}}$ = {r['sigma_eff']:.2f}",
                transform=ax.transAxes, fontsize=11, va="center", ha="center")

        # ---------------- middle: trace ----------------
        ax = fig.add_subplot(gs[i + 1, 1])
        ax.fill_between(t, L_YLIM[0], L_YLIM[1], where=r["L"] > 0,
                        color=C_SFA, alpha=0.10, lw=0, step="mid")
        ax.fill_between(t, L_YLIM[0], L_YLIM[1], where=r["L"] < 0,
                        color=C_SFB, alpha=0.10, lw=0, step="mid")
        ax.plot(t, r["L"], color="0.15", lw=0.25)
        # for sgn in (-1, 1):
        #     ax.axhline(sgn * THRESHOLD, color="0.45", ls=":", lw=1.0)
        for Ls in r["stable"]:
            ax.axhline(Ls, color="tab:green", ls=":", lw=1.0)

        # if r["n_sw"] >= 3:
        #     meas_txt = f"measured dwell = {T_TOTAL / r['n_sw']:.1f}"
        # else:
        #     meas_txt = "measured dwell: not resolvable"
        # ax.text(0.008, 0.96,
        #         f"switches = {r['n_sw']}    {meas_txt}"
        #         f"    Kramers dwell = {1 / r['rate_kramers']:.1f}",
        #         transform=ax.transAxes, fontsize=9, va="top", ha="left",
        #         bbox=dict(fc="white", ec="0.8", alpha=0.9, boxstyle="round,pad=0.3"))

        ax.set_xlim(0, T_TOTAL)
        ax.set_ylim(*L_YLIM)
        ax.set_ylabel("$L(t)$")
        if i == n - 1:
            ax.set_xlabel("Time")
        if i == 0:
            ax.set_title(f"One trajectory, $T$ = {T_TOTAL:g} "
                         "(red = SFA, blue = SFB)", fontsize=10)

        # ---------------- right: dominance durations ----------------
        ax = fig.add_subplot(gs[i + 1, 2])
        d = r["durs"]
        if d.size > 1:
            ax.hist(d, bins=min(30, max(5, d.size // 8)), density=True,
                    color="0.65", edgecolor="0.35", lw=0.5)
            xs = np.linspace(0, d.max(), 200)
            ax.plot(xs, np.exp(-xs / d.mean()) / d.mean(), color="tab:purple",
                    lw=1.6)
            cv = d.std() / d.mean()
            ax.text(0.96, 0.94, f"$n$ = {d.size}\nmean = {d.mean():.1f}\n"
                                f"CV = {cv:.2f}",
                    transform=ax.transAxes, fontsize=8.5, va="top", ha="right")
        else:
            ax.text(0.5, 0.5,
                    f"only {d.size} duration{'s' if d.size != 1 else ''}\n"
                    f"in {T_TOTAL:g} time units",
                    transform=ax.transAxes, fontsize=9.5, va="center", ha="center",
                    color="0.35")
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_ylabel("density")
        if i == n - 1:
            ax.set_xlabel("dominance duration")
        if i == 0:
            ax.set_title("Durations (purple = exponential fit)", fontsize=10)

    fig.suptitle(
        "Sampling across the iso-rate fan: five cells, five switching regimes\n"
        f"($r_{{on}}$ = $r_{{off}}$ = 1, $\\mu_{{noise}}$ = 0, $T$ = {T_TOTAL:g} "
        f"after a {T_BURN:g} burn-in, hysteresis threshold = {THRESHOLD:g})",
        fontsize=14, y=0.988)
    savefig(fig, "fig_isorate_samples.png")


if __name__ == "__main__":
    main()
