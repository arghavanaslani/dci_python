"""Fig 2: Examples of model dynamics.

Panels A-D: example confidence trajectories L(t) over T = 100, blue = L, red =
MAP percept (sign of L), dashed = decision threshold (L = 0).

  A. descending loops (a_p = 1.5), unbiased        -> bistable switching
  B. no loops (a_p = 0), unbiased                  -> leaky random walk (HMM)
  C. descending loops, weak bias (r_on > r_off)    -> SFA preferred
  D. no loops, stronger bias                        -> biased random walk

Panel E: phase-duration histogram. The dCI model predicts an EXPONENTIAL
distribution of dominance durations (the paper's point). Note: this is gamma
with shape k = 1; the gamma-like (k > 1) distributions seen experimentally
require extensions (filtered noise / adaptation / richer decision rule), per the
paper's Discussion. We fit a gamma and report k to make the exponential
character explicit.

Sensory gain w_int = 0.8 throughout (= w_s = 0.7 with mu_int = sigma_int = 1).
This is the continuous Euler-Maruyama gain; equivalently runm.m's ws = 8.

Noise level: we use sigma_noise = 2.1, NOT the caption's stated value of 1.
Reason: with a = 2.1 (the nominal a_p=1.5, w_s=0.7) the attractors sit at L ~ +/-2.3
and, at sigma=1, the inter-well barrier gives a mean dwell of ~360 time units --
far longer than the T=100 window, so the trace essentially never switches (Kramers
rate ~ exp(-dV/D)). The paper's own Fig 2A shows ~6-8 switches over T=100, which is
only reproducible by either lowering a (sacrificing the +/-2.3 amplitude that the
figure clearly shows) or raising the noise. sigma_noise ~ 1.8 keeps the wells at
+/-2.3 (correct amplitude) while giving ~7 switches/T=100 (correct frequency). The
stated sigma=1 is inconsistent with the published switching rate at a=2.1; the
Fig-2 trace script is not in the authors' repo, so the exact value can't be
confirmed, but the figure forces sigma ~ 1.8. With this noise level switching
happens naturally, so no seed cherry-picking is needed.

Volatility knob: the module-level VOLATILITY factor scales all rates by c, the
auto-amplification a by c and sigma by sqrt(c) together -- a pure time-rescaling
that leaves the wells (+/-2.3) and the stationary distribution unchanged while
running the dynamics c x faster (c x more switches, c x denser trace, dwell ~
1/c). Default 1.0 reproduces the paper; 2.0 / 4.0 give the denser look. Panel E
(normalized dwell times) is scale-invariant, so it is unaffected.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

from _common import savefig
from dci import (
    ReducedParams,
    simulate,
    percept,
    switches_min_dwell,
    phase_durations_ensemble,
)

DT = 0.01
W_INT = 0.8
SWITCH_TAU = 0.1   # min-dwell debounce for the reported switch counts (time units)

# Volatility scale c relative to the paper's baseline (rd = r_on + r_off = 1).
# Applying it to all three of {rates, a, sigma} together (below) is a pure time-
# compression: the wells (+/-2.3) and the stationary distribution are unchanged,
# but the dynamics run c x faster -- switching rate and trace density both ~ c,
# mean dwell ~ 1/c. Set 1.0 to reproduce the paper exactly; 2.0 / 4.0 for a
# denser, faster trace.  (rates -> c*rates ; a -> c*a keeps a/r ; sigma -> sqrt(c)*sigma keeps dV/D)
VOLATILITY = 4

A_LOOPS = (2 * 1.5 * 0.7) * VOLATILITY   # a = 2 a_p w_s = 2.1, scaled by volatility
SIGMA = 1.8 * VOLATILITY**0.5            # Fig 2 switching noise (see docstring), scaled ~ sqrt(c)


def rates(r_on, r_off):
    """Baseline (rd=1) transition rates scaled by the volatility factor."""
    return dict(r_on=r_on * VOLATILITY, r_off=r_off * VOLATILITY)


def trace_panel(ax, p, title, T=100.0, seed=0, show_percept=True, show_mean=False):
    n_steps = int(T / DT)
    rng = np.random.default_rng(seed)
    L = simulate(p, n_steps=n_steps, dt=DT, L0=0.0, n_traj=1, rng=rng)
    t = np.arange(n_steps) * DT
    ylim = 4.0
    pc = percept(L)   # MAP percept sign(L): +1 = SFA, -1 = SFB (the red line below)
    ax.plot(t, np.clip(L, -ylim, ylim), color="tab:blue", lw=1.5)
    if show_percept:
        ax.plot(t, pc * 0.15 * ylim, color="tab:red", lw=1.2, alpha=0.7)  # percept
    ax.axhline(0, color="0.4", ls="--", lw=0.8)                    # y=0 (threshold / bias reference)
    if show_mean:
        # solid line at the trajectory's time-average: sits above 0 => positive
        # bias (more time / higher confidence in SFA), at 0 => unbiased.
        m = float(np.mean(L))
        ax.axhline(m, color="tab:green", ls="-", lw=2, label=f"mean L = {m:+.2f}")
        ax.legend(fontsize=7, loc="upper right", frameon=False)

    # Switch counts with a minimum-dwell debounce: a sign(L) flip is counted
    # only once the new percept persists >= SWITCH_TAU, so the noise flicker near
    # L=0 (which inflates the raw sign-crossing count ~20x) is discarded. For the
    # bistable panels this sits on the tau-plateau (the well-defined count);
    # no-loop panels (B, D) have no plateau, so their number stays tau-dependent.
    n_down, n_up = switches_min_dwell(L, DT, tau=SWITCH_TAU)
    # ax.text(0.015, 0.96,
    #         f"switches (dwell>{SWITCH_TAU:g})\n"
    #         f"down (SFA->SFB): {n_down}\n"
    #         f"up   (SFB->SFA): {n_up}\n"
    #         f"total: {n_down + n_up}",
    #         transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
    #         family="monospace",
    #         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.8))

    ax.set_ylim(-ylim, ylim)
    ax.set_xlim(0, T)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Confidence (L)")


def histogram_panel(ax):
    """Dwell-time distribution of the bistable switcher is exponential (k~1).

    Pools dominance durations from the descending-loops case (the switches in
    panels A/C) over a long ensemble and fits a gamma. Shape k ~ 1 means the
    distribution is exponential -- the dCI model's clean prediction. Human data
    are gamma with k > 1, which the paper notes requires extensions (adaptation
    / filtered noise); that is deliberately NOT reproduced here.

    The no-loops case is omitted: without attractors a "dominance phase" is
    ill-defined and its fitted k swings with the threshold, so it is not a
    meaningful distribution to report.
    """
    T = 20000.0
    n_steps = int(T / DT)
    n_traj = 80
    p = ReducedParams(**rates(0.5, 0.5), a=A_LOOPS, w=W_INT, sigma_noise=SIGMA)

    # hysteresis threshold: commit a switch only once L crosses +/-1, well
    # inside the attractor basins (~ +/-2.3), so dt-scale barrier jitter near
    # L=0 does not spawn spurious near-zero "phases".
    threshold = 1.0
    rng = np.random.default_rng(1)
    L2d = simulate(p, n_steps=n_steps, dt=DT, L0=0.0, n_traj=n_traj, rng=rng)
    durs = phase_durations_ensemble(L2d, DT, threshold=threshold)
    durs = durs[durs > 0]
    norm = durs / durs.mean()  # normalize by mean duration

    bins = np.linspace(0, 6, 40)
    ax.hist(norm, bins=bins, density=True, histtype="step", color="tab:blue",
            lw=1.6, label=f"Descending loops (a_p=1.5), unbiased  (n={durs.size})")
    # gamma fit (shape k); k ~ 1 means exponential
    k, loc, theta = stats.gamma.fit(norm, floc=0)
    x = np.linspace(0, 6, 200)
    ax.plot(x, np.exp(-x), "k--", lw=1.2, label="Exponential (k=1) reference")
    ax.set_xlabel("Normalized dwell time")
    ax.set_ylabel("Probability density")
    ax.set_title(f"E. Dwell times are exponential: gamma shape k = {k:.2f}\n"
                 "(model predicts k=1; human data are gamma with k>1)",
                 fontsize=9)
    ax.legend(fontsize=8, frameon=False)


def main():
    fig = plt.figure(figsize=(13, 7))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.3], hspace=0.7,
                  wspace=0.25)

    unbiased = rates(0.5, 0.5)
    biased_loops = rates(0.54, 0.46)
    biased_noloops = rates(0.6, 0.4)

    trace_panel(fig.add_subplot(gs[0, 0]),
                ReducedParams(a=A_LOOPS, w=W_INT, sigma_noise=SIGMA, **unbiased),
                "A. Descending loops, unbiased", seed=0)
    trace_panel(fig.add_subplot(gs[0, 1]),
                ReducedParams(a=0.0, w=W_INT, sigma_noise=SIGMA, **unbiased),
                "B. No loops, unbiased", seed=0,
                show_percept=True, show_mean=True)
    trace_panel(fig.add_subplot(gs[1, 0]),
                ReducedParams(a=A_LOOPS, w=W_INT, sigma_noise=SIGMA, **biased_loops),
                "C. Descending loops, weak bias (SFA)", seed=0)
    trace_panel(fig.add_subplot(gs[1, 1]),
                ReducedParams(a=0.0, w=W_INT, sigma_noise=SIGMA, **biased_noloops),
                "D. No loops, stronger bias", seed=1,
                show_percept=True, show_mean=True)
    ax_e = fig.add_subplot(gs[2, :])
    histogram_panel(ax_e)
    for ax in (fig.axes[2], fig.axes[3]):
        ax.set_xlabel("Time")

    fig.suptitle("Fig 2: Examples of dCI model dynamics", y=0.995)
    savefig(fig, "fig2_dynamics.png")


if __name__ == "__main__":
    main()
