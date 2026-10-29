"""S4: parameter recovery.

Reproduces the authors' recovery experiment (run_param_recovery.m): generate
fake intermittent-presentation survival curves from known {ws, alpha, a} (with
r = r_on + r_off fixed at 2), recover the parameters by fitting the survival
curves, and scatter true vs estimated. Good recovery = points on the diagonal.

Panels:
  A. loop strength a            -- true vs estimated
  B. bias alpha = r_on/r        -- true vs estimated
  C. net bistability a - r      -- true vs estimated  (the CI-strength axis)
  D. example survival curves    -- data (points) vs fitted model (lines)

The two scientifically meaningful recovered quantities are a - r (how far past
the pitchfork the system sits) and alpha (the prior bias); panel A/C and B are
the direct read-outs.
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import savefig
from dci.recovery import (
    run_recovery, survival_curve, TON, TOFF_LIST, R_FIXED,
)

# Recovery settings (validated by profiling; scaled down from the paper's
# Nexp=5000/Nmod=2000/Nit=50 to stay tractable in pure Python).
N_SETS = 8
N_TRAJ = 1000
N_REPEAT = 12
N_RESTARTS = 6      # multi-start guard against the ws-a local minimum (set-4 failure)
SEED = 0


def _scatter(ax, true, est, label, unit=""):
    true, est = np.asarray(true), np.asarray(est)
    lo = min(true.min(), est.min())
    hi = max(true.max(), est.max())
    pad = 0.08 * (hi - lo + 1e-9)
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, "0.6", ls="--", lw=1, zorder=0)   # y = x
    ax.scatter(true, est, s=40, color="tab:blue", edgecolor="k", lw=0.5, zorder=2)
    r = np.corrcoef(true, est)[0, 1] if true.size > 1 else np.nan
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(f"True {label}{unit}")
    ax.set_ylabel(f"Estimated {label}{unit}")
    ax.set_title(f"{label}   (r = {r:.2f})", fontsize=10)


def main():
    res = run_recovery(n_sets=N_SETS, n_traj=N_TRAJ, n_repeat=N_REPEAT,
                       n_restarts=N_RESTARTS, seed=SEED)

    a_t = [r["true"]["a"] for r in res]
    a_e = [r["est"]["a"] for r in res]
    al_t = [r["true"]["alpha"] for r in res]
    al_e = [r["est"]["alpha"] for r in res]
    amr_t = [r["true"]["a_minus_r"] for r in res]
    amr_e = [r["est"]["a_minus_r"] for r in res]

    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    _scatter(axs[0, 0], a_t, a_e, "loop strength a")
    _scatter(axs[0, 1], al_t, al_e, "bias alpha")
    _scatter(axs[1, 0], amr_t, amr_e, "net bistability a - r")

    # Panel D: survival-curve fit for the first recovered set.
    ax = axs[1, 1]
    r0 = res[0]
    (dp, do) = r0["data"]
    e = r0["est"]
    mrng = np.random.default_rng(12345)
    mp, mo = survival_curve(e["ws"], e["alpha"], e["a"], TON, TOFF_LIST, R_FIXED,
                            n_traj=2000, n_repeat=20, rng=mrng)
    x = np.array(TOFF_LIST)
    ax.plot(x, dp, "o", color="tab:red", label="data: SFA (prperson)")
    ax.plot(x, do, "o", color="tab:blue", label="data: SFB (prpersoff)")
    ax.plot(x, mp, "-", color="tab:red", lw=1.4, label="fit: SFA")
    ax.plot(x, mo, "-", color="tab:blue", lw=1.4, label="fit: SFB")
    ax.set_xscale("log")
    ax.set_xlabel("OFF duration Toff (steps)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("D. Example fit: survival curves", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle(
        f"S4: parameter recovery (r fixed = {R_FIXED:g}, N = {N_SETS} sets)",
        y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "figS4_recovery.png")


if __name__ == "__main__":
    main()
