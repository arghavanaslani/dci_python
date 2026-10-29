"""Teaching figure: the (a, sigma_eff) regime map for switching.

fig_landscape_switching.py varied the loop strength a at fixed noise;
fig_noise_switching.py varied the noise at fixed a. This sweeps both at once and
asks: over the (a, sigma_eff) plane, where does the model actually alternate?

    a          loop strength; with r it sets the barrier dV
    sigma_eff  = w*_int * sigma_noise, the only noise combination that reaches
               the SDE (w and sigma never appear apart)

Plotted: measured alternation rate (switches per unit time), simulated.
  Overlaid:
    * the bifurcation line a = r. To its left there is only one well, so there
      is no second state to switch to; anything counted there is an undecided
      variable crossing zero, not a percept. Noise does not move this line.
    * white contours of the Kramers exponent dV/D, with D = sigma_eff^2/2.
      These are the analytic iso-rate curves. Since dV ~ (3/2)(a-r)^2/r near
      threshold and rate ~ exp(-dV/D), constant rate means sigma_eff ~ |a - r|:
      the contours are straight rays fanning out of the point (a = r, sigma = 0).
      That fan is the identifiability problem drawn -- a measured alternation
      rate pins you to a curve in this plane, never to a point.

Note on switch counting: amplitude hysteresis as in analysis.py (commit only
past +/-THRESHOLD), evaluated online so no trajectory is stored -- the full grid
history would be gigabytes. The count does depend on THRESHOLD; that dependence
is itself diagnostic (see the `switches_min_dwell` docstring in analysis.py: a
genuinely bistable trace has a plateau where the count stops caring about the
debounce, a monostable one does not).

The integrator below duplicates model.simulate's update rule deliberately: that
function takes scalar parameters and cannot broadcast over a parameter grid.
Convention kept identical -- dt*drift + sigma_eff*sqrt(dt)*xi, clipped at +/-50.

Not a paper figure; written to locate the switching regime in parameter space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from _common import savefig
from dci import ReducedParams, potential, find_fixed_points

DT = 0.01
T_TOTAL = 300.0                    # matches the trace window of the slice figures
T_BURN = 30.0                      # discard: L starts at 0, i.e. on the barrier top
N_STEPS = int(T_TOTAL / DT)
N_BURN = int(T_BURN / DT)
N_TRAJ = 10

R_ON = R_OFF = 1.0
R_SUM = R_ON + R_OFF               # r = 2  ->  bifurcation at a = 2

# Offset from 0 so no grid point lands exactly on a = r, where f'(0) = 0 and the
# settling time is genuinely infinite (critical slowing down).
A_GRID = np.linspace(0.05, 6.0, 24)
SIG_GRID = np.linspace(0.25, 3.0, 24)

THRESHOLD = 1.0                    # amplitude hysteresis for committing a percept
CLIP = 50.0
SEED = 3


def sweep():
    """Alternation rate over the (a, sigma_eff) grid, in switches per unit time."""
    rng = np.random.default_rng(SEED)
    A = A_GRID[:, None, None]                       # (na, 1, 1)
    S = SIG_GRID[None, :, None]                     # (1, ns, 1)
    shape = (A_GRID.size, SIG_GRID.size, N_TRAJ)

    L = np.zeros(shape)
    committed = np.ones(shape)                      # arbitrary initial commitment
    count = np.zeros(shape, dtype=np.int64)

    sqdt = np.sqrt(DT)
    bias = R_ON - R_OFF

    for t in range(N_STEPS):
        Lc = np.clip(L, -CLIP, CLIP)
        det = A * Lc + R_ON * np.exp(-Lc) - R_OFF * np.exp(Lc) + bias
        L = L + DT * det + S * sqdt * rng.standard_normal(shape)

        new = np.where(L > THRESHOLD, 1.0,
                       np.where(L < -THRESHOLD, -1.0, committed))
        if t >= N_BURN:
            count += (new != committed)
        committed = new

    t_measured = (N_STEPS - N_BURN) * DT
    return count.sum(axis=2) / (N_TRAJ * t_measured)


def geometry():
    """Per-a barrier height dV (analytic, noise-free)."""
    dV = np.full(A_GRID.size, np.nan)
    for i, a in enumerate(A_GRID):
        p = ReducedParams(r_on=R_ON, r_off=R_OFF, a=float(a), w=1.0)
        fps = find_fixed_points(p)
        stable = [x for x, s in fps if s]
        unstable = [x for x, s in fps if not s]
        if len(stable) >= 2 and unstable:
            V = potential(np.array(stable + [unstable[0]]), p, shift=False)
            dV[i] = float(V[-1] - np.min(V[:-1]))
    return dV


def main():
    rate = sweep()                                  # (na, ns)
    dV = geometry()

    D = 0.5 * SIG_GRID ** 2                         # noise energy
    ratio_map = dV[:, None] / D[None, :]            # Kramers exponent dV/D

    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    X, Y = np.meshgrid(A_GRID, SIG_GRID, indexing="ij")

    rmask = np.ma.masked_where(rate <= 0, rate)
    pc = ax.pcolormesh(X, Y, rmask, norm=LogNorm(), cmap="viridis", shading="auto")
    fig.colorbar(pc, ax=ax, label="alternation rate (switches / unit time)")

    cs = ax.contour(X, Y, ratio_map, levels=[1, 2, 4, 8], colors="white",
                    linewidths=1.3, linestyles="--")
    ax.clabel(cs, fmt=lambda v: f"$\\Delta V/D$ = {v:g}", fontsize=8)

    ax.axvline(R_SUM, color="tab:red", lw=2)
    ax.axvspan(A_GRID[0], R_SUM, color="tab:red", alpha=0.12, lw=0)
    ax.text(R_SUM - 0.12, SIG_GRID[-1] * 0.97,
            "monostable ($a < r$):\nno second state\nto switch to",
            color="tab:red", fontsize=9, ha="right", va="top")
    ax.set_title("Where the model alternates\n"
                 "(white dashed = analytic iso-rate curves)", fontsize=11)

    ax.set_xlabel("loop strength $a$        ($r$ = 2)")
    ax.set_ylabel("effective noise $\\sigma_{eff}$ = $w^*_{int}\\,\\sigma_{noise}$")
    ax.set_xlim(A_GRID[0], A_GRID[-1])
    ax.set_ylim(SIG_GRID[0], SIG_GRID[-1])

    fig.suptitle(
        "Regime map: loop strength and noise together determine switching\n"
        f"($r_{{on}}$ = $r_{{off}}$ = 1, $\\mu_{{noise}}$ = 0, "
        f"T = {T_TOTAL:g} per cell, {N_TRAJ} trajectories, "
        f"hysteresis threshold = {THRESHOLD:g})",
        fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "fig_regime_map.png")


if __name__ == "__main__":
    main()
