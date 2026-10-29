"""Parameter recovery (S4): fit {ws, alpha, a} from intermittent survival curves.

Port of the authors' MATLAB recovery pipeline (run_param_recovery.m,
recovparam.m, objfun.m, partostatdir.m, stattopardir.m). The volatility
r = r_on + r_off is held FIXED (r = 2, as in run_param_recovery.m); the three
recovered parameters are

    ws     -- sensory gain / noise amplitude   (paper draw range 20..100)
    alpha  -- bias, r_on / (r_on + r_off)       (draw range 0.5..0.9)
    a      -- loop strength                      (draw range [1,3]*r = 2..6)

with r_on = alpha*r, r_off = (1-alpha)*r.

Data are intermittent-presentation *survival probabilities*: for a fixed ON
duration ``Ton`` and each OFF gap ``Toff`` in TOFF_LIST, prperson = P(an SFA
percept survives the gap) and prpersoff = P(an SFB percept survives). Recovery
fits {ws, alpha, a} to minimize the squared error between model and data
survival curves (r fixed).

Noise convention mirrors runm.m exactly: the update is
``L += dt*(drift + ws*xi)`` (dt-scaled noise, w = ws), which at dt = 0.01 equals
the sqrt(dt) Euler--Maruyama scheme with w_int = ws/10. We estimate the survival
probabilities from an *ensemble* of n_traj trajectories run in parallel (rather
than one long trajectory as in runm.m) -- statistically equivalent for a
stationary process and far faster in NumPy. This is validated against the scalar
runm.m port ``run_intermittent_single`` in the tests.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .model import DT_DEFAULT

_CLIP = 50.0

R_FIXED = 2.0
TON = 10
TOFF_LIST = (1, 10, 50, 200, 500)

# Fit bounds (wider than the true-parameter draw ranges, as in recovparam.m).
_LB = np.array([20.0, 0.16, 0.0])    # ws, alpha, a
_UB = np.array([100.0, 0.95, 12.0])


def params_to_reduced(ws, alpha, a, r=R_FIXED):
    """(ws, alpha, a) -> (r_on, r_off, a, ws).  partostatdir analog (r fixed)."""
    r_on = alpha * r
    r_off = (1.0 - alpha) * r
    return r_on, r_off, a, ws


def reduced_to_params(r_on, r_off, a):
    """(r_on, r_off, a) -> (ws-less) (alpha, a).  stattopardir analog (bias + loop)."""
    alpha = r_on / (r_on + r_off)
    return alpha, a


def survival_curve(ws, alpha, a, Ton=TON, Toff_list=TOFF_LIST, r=R_FIXED,
                   dt=DT_DEFAULT, n_traj=1000, n_repeat=15, rng=None):
    """Survival probabilities across Toff for one parameter set (vectorized).

    Returns
    -------
    (prperson, prpersoff) : ndarray, ndarray  (each length len(Toff_list))
        Survival probability of the SFA (L>0) and SFB (L<0) percepts across the
        OFF gap, pooled over the ensemble.
    """
    if rng is None:
        rng = np.random.default_rng()
    r_on, r_off, a, ws = params_to_reduced(ws, alpha, a, r)
    bias = r_on - r_off
    prperson = np.empty(len(Toff_list))
    prpersoff = np.empty(len(Toff_list))

    for j, Toff in enumerate(Toff_list):
        period = Ton + Toff
        n_time = period * (n_repeat + 1)
        L = np.zeros(n_traj)
        samples = []
        for t in range(1, n_time):
            Lc = np.clip(L, -_CLIP, _CLIP)
            det = a * Lc + r_on * np.exp(-Lc) - r_off * np.exp(Lc) + bias
            if (t % period) < Ton:
                L = L + dt * det + dt * ws * rng.standard_normal(n_traj)
            else:
                L = L + dt * det
            # sample sign(L) once per period, near the end of the ON block
            if (t + 1) % period == (Ton - 1):
                samples.append(np.sign(L))

        S = np.asarray(samples)                 # (n_samples, n_traj)
        cur, nxt = S[:-1], S[1:]
        n_plus = np.count_nonzero(cur > 0)
        n_minus = np.count_nonzero(cur < 0)
        prperson[j] = 1.0 - np.count_nonzero((cur > 0) & (nxt < 0)) / (n_plus + 1e-4)
        prpersoff[j] = 1.0 - np.count_nonzero((cur < 0) & (nxt > 0)) / (n_minus + 1e-4)
    return prperson, prpersoff


def _sse(model, data):
    (mp, mo), (dp, do) = model, data
    return float(np.sum((mp - dp) ** 2) + np.sum((mo - do) ** 2))


def fit_survival(data, Ton=TON, Toff_list=TOFF_LIST, r=R_FIXED,
                 n_traj=1000, n_repeat=15, n_restarts=3, model_seed=0, rng=None):
    """Fit (ws, alpha, a) to data survival curves; r fixed.

    Uses multi-start bounded Nelder--Mead (scipy's derivative-free analog of the
    paper's ``patternsearch``). Common random numbers: every objective evaluation
    reuses ``model_seed`` so the objective is deterministic; the data are
    generated with a *different* seed (see ``recover_once``) so recovery error is
    honest rather than trivially zero.
    """
    if rng is None:
        rng = np.random.default_rng()

    def objective(x):
        ws, alpha, a = x
        mrng = np.random.default_rng(model_seed)
        model = survival_curve(ws, alpha, a, Ton, Toff_list, r, DT_DEFAULT,
                               n_traj, n_repeat, mrng)
        return _sse(model, data)

    best = None
    bounds = list(zip(_LB, _UB))
    for _ in range(n_restarts):
        x0 = _LB + rng.random(3) * (_UB - _LB)
        res = minimize(objective, x0, method="Nelder-Mead", bounds=bounds,
                       options=dict(maxfev=300, xatol=1e-2, fatol=1e-4))
        if best is None or res.fun < best.fun:
            best = res
    ws, alpha, a = best.x
    r_on, r_off, _, _ = params_to_reduced(ws, alpha, a, r)
    return dict(ws=float(ws), alpha=float(alpha), a=float(a),
                r_on=float(r_on), r_off=float(r_off),
                a_minus_r=float(a - r), sse=float(best.fun))


def recover_once(ws_true, alpha_true, a_true, r=R_FIXED, n_traj=1000,
                 n_repeat=15, n_restarts=3, data_seed=1, model_seed=0):
    """Generate data at known params, then recover; return {'true','est','data'}."""
    drng = np.random.default_rng(data_seed)
    data = survival_curve(ws_true, alpha_true, a_true, TON, TOFF_LIST, r,
                          DT_DEFAULT, n_traj, n_repeat, drng)
    est = fit_survival(data, TON, TOFF_LIST, r, n_traj, n_repeat,
                       n_restarts, model_seed)
    true = dict(ws=ws_true, alpha=alpha_true, a=a_true, a_minus_r=a_true - r)
    return dict(true=true, est=est, data=data)


def _informative(prperson, prpersoff):
    """Paper's criterion: first-Toff survival not too flippy and not too stable."""
    sp_on, sp_off = prperson[0], prpersoff[0]
    too_flippy = (sp_on < 0.65) and (sp_off < 0.65)
    too_stable = max(sp_on, sp_off) > 0.85
    return not (too_flippy or too_stable)


def run_recovery(n_sets=8, r=R_FIXED, n_traj=1000, n_repeat=15, n_restarts=3,
                 seed=0):
    """Recover from n_sets random true parameter sets (informative-range only).

    Returns a list of per-set dicts from ``recover_once``.
    """
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_sets):
        # draw informative true params (cheap screening ensemble)
        while True:
            ws = rng.uniform(20, 100)
            alpha = rng.uniform(0.5, 0.9)
            a = r * rng.uniform(1.0, 3.0)
            pp, po = survival_curve(ws, alpha, a, TON, TOFF_LIST, r, DT_DEFAULT,
                                    n_traj=300, n_repeat=10, rng=rng)
            if _informative(pp, po):
                break
        results.append(recover_once(ws, alpha, a, r, n_traj, n_repeat,
                                    n_restarts, data_seed=1000 + i, model_seed=i))
    return results
