"""Teaching figure: history-dependence across the five iso-rate samples.

fig_isorate_samples.py drew the traces and the duration histograms, which settle
the SHAPE of the duration distribution (order-blind). This asks the complementary
question -- does a duration depend on the ones before it? -- using the
"cumulative history" measure of Pastukhov, Garcia-Rodriguez, Haenicke, Guillamon,
Deco & Braun (2013), Front Comput Neurosci 7:17 (resources/fncom-07-00017.pdf),
their Eq 6 and Figure 2.

The dominance record S_x(t) is passed through a leaky integrator of time constant
tau_H, giving H_x(t) in [0, 1] -- large only when x has dominated both
substantially and recently. H is read at each reversal and correlated against
ln(T) of the period that follows, for tau_H swept over a wide range. gamma_H =
tau_H / T_dom is the normalised timescale, which they find pinned near 0.5 across
three displays, with |c_H| of 0.1-0.4.

Plotted SIGNED rather than as the paper's mean absolute value. Since H_y = 1-H_x
the four pairings collapse to two independent numbers, and adaptation predicts
opposite signs for them:

    SELF  = corr(H_x, ln T_x)   x is tired -> x's own next turn is SHORT (neg)
    RIVAL = corr(H_x, ln T_y)   x is tired -> the rival's turn is LONG  (pos)

Their ratio distinguishes mechanisms -- |self| >> |rival| means fatigue is
percept-specific, |self| ~ |rival| means the two share one resource. Averaging
the magnitudes, as the paper does, discards both the sign and that ratio; it is
the right call for a low-variance fitting constraint and the wrong one for
reading mechanism. The paper's statistic is still printed for comparison.

Our SDE must give zero for both. It has one state variable, and at every
commitment that variable sits on the threshold, +/-THRESHOLD -- the same state
every time, so no information survives a reversal and the durations are i.i.d.
An adaptation model (Laing-Chow) has a second, slow state variable that does
survive; that is where history-dependence comes from.

The extremes are taken over a sweep, and the extreme of a noisy curve is not
zero even when the curve's expectation is. The shuffled band (durations resampled
with replacement, order destroyed, identical pipeline) is therefore the
reference, not zero -- this is their Figure 2B. Signing the statistic makes that
null two-sided and centred on zero, where taking absolute values first folds it
onto the positive side and pushes the whole band upward.

H is evaluated in closed form between reversals rather than by filtering the
million-sample trace: S is piecewise constant, so H_{i+1} = 1 - (1-H_i)*exp(-T/t)
while dominant and H_i*exp(-T/t) while suppressed. Correlations accumulate as
running sums so the (tau, replicate, period) array is never held in memory.

Not a paper figure; written to test the fan's samples for history-dependence.
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import savefig
from dci import committed_state, switch_indices
from fig_isorate_samples import build_rows, DT, THRESHOLD, T_TOTAL

N_SHUFFLE = 300
H_BURN = 10                        # periods discarded while H equilibrates
GAMMA_GRID = np.logspace(-2.0, np.log10(20.0), 40)   # tau_H / T_dom
MIN_N = 20                         # below this no statistic is attempted
SEED = 23

HUMAN_CH = (0.1, 0.4)
HUMAN_GAMMA_H = 0.5

C_SELF = "tab:blue"
C_RIVAL = "tab:red"


def duration_sequence(L):
    """Ordered (durations, percept signs) using the same rule as the regime map."""
    s = committed_state(L, THRESHOLD)
    sw = switch_indices(L, THRESHOLD)
    if sw.size < 2:
        return np.empty(0), np.empty(0)
    return np.diff(sw) * DT, s[sw[:-1] + 1]


def history_corr(durs, signs, taus):
    """Signed (self, rival) correlations for each tau and replicate.

    durs is (n_rep, n_per). self is corr(H_x, ln T) over x-dominant periods,
    rival is the same H_x against the periods where the rival runs.
    """
    n_rep, n_per = durs.shape
    H = np.zeros((taus.size, n_rep))
    inv = 1.0 / taus[:, None]

    acc = {g: dict(n=0,
                   sx=np.zeros((taus.size, n_rep)),
                   sxx=np.zeros((taus.size, n_rep)),
                   sxy=np.zeros((taus.size, n_rep)),
                   sy=np.zeros(n_rep), syy=np.zeros(n_rep))
           for g in (1, -1)}

    for i in range(n_per):
        T = durs[:, i]
        g = 1 if signs[i] > 0 else -1
        if i >= H_BURN:
            y = np.log(T)
            a = acc[g]
            a["n"] += 1
            a["sx"] += H
            a["sxx"] += H * H
            a["sxy"] += H * y[None, :]
            a["sy"] += y
            a["syy"] += y * y
        decay = np.exp(-T[None, :] * inv)
        H = 1.0 - (1.0 - H) * decay if g > 0 else H * decay

    out = []
    for g in (1, -1):
        a = acc[g]
        n = a["n"]
        num = n * a["sxy"] - a["sx"] * a["sy"][None, :]
        den = np.sqrt(np.maximum(n * a["sxx"] - a["sx"] ** 2, 1e-30)
                      * np.maximum(n * a["syy"] - a["sy"] ** 2, 1e-30)[None, :])
        out.append(num / den)
    return out[0], out[1]


def analyse(rows):
    rng = np.random.default_rng(SEED)
    for r in rows:
        durs, signs = duration_sequence(r["L"])
        r["seq"] = durs
        if durs.size < MIN_N:
            r["sweep"] = None
            continue

        taus = GAMMA_GRID * durs.mean()
        self_c, rival_c = history_corr(durs[None, :], signs, taus)
        r["self"] = self_c[:, 0]
        r["rival"] = rival_c[:, 0]
        r["absmean"] = 0.5 * (np.abs(r["self"]) + np.abs(r["rival"]))

        shuf = rng.choice(durs, size=(N_SHUFFLE, durs.size), replace=True)
        ns, nr = history_corr(shuf, signs, taus)
        r["null_lo"] = np.minimum(np.percentile(ns, 2.5, axis=1),
                                  np.percentile(nr, 2.5, axis=1))
        r["null_hi"] = np.maximum(np.percentile(ns, 97.5, axis=1),
                                  np.percentile(nr, 97.5, axis=1))

        r["self_min"] = float(r["self"].min())            # adaptation: negative
        r["rival_max"] = float(r["rival"].max())          # adaptation: positive
        r["self_null"] = float(np.percentile(ns.min(axis=0), 2.5))
        r["rival_null"] = float(np.percentile(nr.max(axis=0), 97.5))

        j = int(np.argmax(r["absmean"]))
        r["c_H"] = float(r["absmean"][j])
        r["gamma_H"] = float(GAMMA_GRID[j])
        r["c_H_null"] = float(np.percentile(
            (0.5 * (np.abs(ns) + np.abs(nr))).max(axis=0), 97.5))
        r["sweep"] = True


def print_table(rows):
    head = (f"{'dV/D':>6} {'n':>6} {'T_dom':>8} | {'self min':>9} {'null':>8} | "
            f"{'rival max':>10} {'null':>8} | {'|c_H|':>7} {'null':>7} {'gam_H':>7}")
    print(head)
    print("-" * len(head))
    for r in rows:
        if r["sweep"] is None:
            print(f"{r['c']:6.2f} {r['seq'].size:6d}   -- too few durations to "
                  f"measure --")
            continue
        print(f"{r['c']:6.2f} {r['seq'].size:6d} {r['seq'].mean():8.2f} | "
              f"{r['self_min']:9.3f} {r['self_null']:8.3f} | "
              f"{r['rival_max']:10.3f} {r['rival_null']:8.3f} | "
              f"{r['c_H']:7.3f} {r['c_H_null']:7.3f} {r['gamma_H']:7.2f}")
    print()
    print("adaptation predicts  self < 0  and  rival > 0, each beyond its null.")
    print(f"humans (Pastukhov 2013, Table 1): |c_H| = {HUMAN_CH[0]}-{HUMAN_CH[1]}, "
          f"gamma_H ~ {HUMAN_GAMMA_H}")


def main():
    rows = build_rows()
    analyse(rows)
    print_table(rows)

    fig, axs = plt.subplots(1, len(rows), figsize=(16.0, 5.2), sharey=True)

    for i, (ax, r) in enumerate(zip(axs, rows)):
        if r["sweep"]:
            ax.axhspan(*HUMAN_CH, color="tab:green", alpha=0.14, lw=0,
                       label="where humans land")
            ax.axhspan(-HUMAN_CH[1], -HUMAN_CH[0], color="tab:green", alpha=0.14,
                       lw=0)
            ax.axvline(HUMAN_GAMMA_H, color="tab:green", lw=1.3, ls=":")
            ax.fill_between(GAMMA_GRID, r["null_lo"], r["null_hi"], color="0.65",
                            alpha=0.6, lw=0, label="shuffled null (95%)")
            ax.axhline(0.0, color="0.3", lw=1.0)
            ax.plot(GAMMA_GRID, r["self"], color=C_SELF, lw=2.0,
                    label="self: $corr(H_x, \\ln T_x)$")
            ax.plot(GAMMA_GRID, r["rival"], color=C_RIVAL, lw=2.0,
                    label="rival: $corr(H_x, \\ln T_y)$")
            ax.set_xscale("log")
            ax.set_ylim(-0.45, 0.45)
            ax.set_title(f"{i + 1}.   $\\Delta V/D$ = {r['c']:g}\n"
                         f"$n$ = {r['seq'].size} durations", fontsize=10.5)
            if i == len(rows) - 1:
                ax.legend(fontsize=8, loc="lower center", framealpha=0.92)
        else:
            ax.text(0.5, 0.5, f"{r['seq'].size} duration\nno sequence to test",
                    transform=ax.transAxes, fontsize=10, ha="center", va="center",
                    color="0.35")
            ax.set_xticks([])
            ax.set_title(f"{i + 1}.   $\\Delta V/D$ = {r['c']:g}", fontsize=10.5)

        ax.set_xlabel("$\\gamma_H = \\tau_H / T_{dom}$")
        if i == 0:
            ax.set_ylabel("correlation with $\\ln T$")

    fig.suptitle(
        "History-dependence, signed: does a tired percept shorten its own next "
        "turn (blue $<$ 0) and lengthen its rival's (red $>$ 0)?\n"
        "adaptation would push blue down and red up, past the grey; "
        "grey = the same durations with their order shuffled",
        fontsize=13, y=0.99)
    fig.text(0.5, 0.015,
             f"$T$ = {T_TOTAL:g}, threshold = {THRESHOLD:g}, {N_SHUFFLE} shuffles; "
             "human values from Pastukhov et al. 2013, Table 1",
             ha="center", fontsize=9, color="0.35")
    fig.subplots_adjust(left=0.062, right=0.99, top=0.78, bottom=0.15, wspace=0.12)
    savefig(fig, "fig_isorate_stats.png")


if __name__ == "__main__":
    main()
