"""Diagnostics extracted from simulated confidence trajectories.

The perceptual decision is MAP (paper, "Perceptual decision"): the percept is
sign(L) (SFA if L > 0, SFB if L < 0). A perceptual switch is a (sustained)
sign change of L. From the sequence of switches we derive phase (dominance)
durations, predominance, mean phase durations and alternation rate -- the
quantities underlying Figs 2E, 5 and 7.

Hysteresis (``threshold``): a raw sign criterion counts every dt-scale
re-crossing of L = 0 near a switch as a separate "phase", which floods the
duration distribution with spurious near-zero values. A perceptual switch is
instead a *sustained* change: the committed percept flips to +1 only once L
rises above +threshold and to -1 only once L falls below -threshold. With
threshold = 0 this reduces to forward-filled sign(L).
"""

from __future__ import annotations

import numpy as np


def committed_state(L, threshold: float = 0.0):
    """Debounced percept (+1/-1) using hysteresis at +/- threshold.

    Vectorized per 1D trajectory: mark commitment events (L > +thr -> +1,
    L < -thr -> -1) and forward-fill the most recent one.
    """
    L = np.asarray(L, dtype=float)
    ev = np.zeros(L.shape)
    ev[L > threshold] = 1.0
    ev[L < -threshold] = -1.0
    idx = np.where(ev != 0, np.arange(ev.size), 0)
    np.maximum.accumulate(idx, out=idx)
    s = ev[idx]
    s[s == 0] = 1.0  # before the first commitment, default to +1
    return s


def percept(L):
    """MAP percept sign(L), zeros carried forward (threshold-free)."""
    return committed_state(L, threshold=0.0)


def switch_indices(L, threshold: float = 0.0):
    """Indices t where the (debounced) percept changes between t and t+1."""
    s = committed_state(L, threshold)
    return np.where(np.diff(s) != 0)[0]


def switches_min_dwell(L, dt: float, tau: float = 0.5):
    """Directed switch counts after a minimum-dwell (temporal) debounce.

    A sign(L) flip is committed only once the new percept has persisted for at
    least ``tau`` (time units); briefer flips -- the noise-driven flicker as L
    grazes 0 -- are ignored. This debounces by *duration*, complementing the
    amplitude hysteresis of ``committed_state`` (which debounces by requiring
    |L| to reach a threshold), and matches how many switches a viewer can
    actually resolve in a trace.

    Returns
    -------
    (n_down, n_up) : int, int
        Counts of +1 -> -1 (SFA -> SFB, "down") and -1 -> +1 (SFB -> SFA, "up").

    Notes
    -----
    For a bistable trajectory the total is flat for ``tau`` anywhere between the
    flicker timescale and the dwell timescale (a plateau) -- that plateau is the
    well-defined switch count. For a non-bistable (no-loops) trace no such
    plateau exists and the count stays ``tau``-dependent.
    """
    L = np.asarray(L, dtype=float)
    if L.size == 0:
        return 0, 0
    s = np.where(L >= 0, 1, -1)
    min_len = max(1, int(round(tau / dt)))
    cur = int(s[0])
    run_val = int(s[0])
    run_len = 1
    n_down = n_up = 0
    for i in range(1, s.size):
        if s[i] == run_val:
            run_len += 1
        else:
            run_val = int(s[i])
            run_len = 1
        if run_val != cur and run_len >= min_len:
            if cur > 0:
                n_down += 1     # +1 -> -1
            else:
                n_up += 1       # -1 -> +1
            cur = run_val
    return n_down, n_up


def phase_durations(L, dt: float, threshold: float = 0.0):
    """Dominance durations (time between consecutive switches), one trajectory.

    The first and last (censored) segments are excluded, since only intervals
    bounded by two switches have a well-defined duration.
    """
    sw = switch_indices(L, threshold)
    if sw.size < 2:
        return np.empty(0)
    return np.diff(sw) * dt


def phase_durations_ensemble(L2d, dt: float, threshold: float = 0.0):
    """Pool phase durations across all trajectories (rows of a 2D array)."""
    L2d = np.atleast_2d(L2d)
    if L2d.shape[0] == 0:
        return np.empty(0)
    return np.concatenate([phase_durations(row, dt, threshold) for row in L2d])


def phase_durations_by_percept(L, dt: float, threshold: float = 0.0):
    """Return (durations_SFA, durations_SFB) for the L>0 and L<0 percepts."""
    s = committed_state(L, threshold)
    sw = switch_indices(L, threshold)
    if sw.size < 2:
        return np.empty(0), np.empty(0)
    durs = np.diff(sw) * dt
    # segment between switch k and k+1 holds the state value just after switch k
    seg_sign = s[sw[:-1] + 1]
    return durs[seg_sign > 0], durs[seg_sign < 0]


def relative_predominance(L, threshold: float = 0.0):
    """Fraction of time spent in the SFA percept (committed state > 0). Fig 5A/7A."""
    s = committed_state(L, threshold)
    return float(np.mean(s > 0))


def mean_phase_duration_by_percept(L2d, dt: float, threshold: float = 0.0):
    """(mean SFA duration, mean SFB duration) pooled over trajectories. Fig 5B/7B."""
    L2d = np.atleast_2d(L2d)
    sfa, sfb = [], []
    for row in L2d:
        a, b = phase_durations_by_percept(row, dt, threshold)
        sfa.append(a)
        sfb.append(b)
    sfa = np.concatenate(sfa) if sfa else np.empty(0)
    sfb = np.concatenate(sfb) if sfb else np.empty(0)
    mean_a = float(np.mean(sfa)) if sfa.size else np.nan
    mean_b = float(np.mean(sfb)) if sfb.size else np.nan
    return mean_a, mean_b


def alternation_rate(L2d, dt: float, threshold: float = 0.0):
    """Mean number of switches per unit time across trajectories. Fig 5C/5D."""
    L2d = np.atleast_2d(L2d)
    total_switches = sum(switch_indices(row, threshold).size for row in L2d)
    total_time = L2d.shape[0] * (L2d.shape[1] - 1) * dt
    return total_switches / total_time if total_time > 0 else np.nan
