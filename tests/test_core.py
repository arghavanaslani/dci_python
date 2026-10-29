"""Validation of the core dCI engine against analytic limits.

Run with:  python -m pytest dci_python/tests/test_core.py   (if pytest present)
or simply: python dci_python/tests/test_core.py             (plain asserts)

Because no MATLAB is available we validate against the paper's analytic results
rather than against runm.m bit-for-bit:

* Eq 7 : with no descending loops (a = 0) the only fixed point is the prior,
         L* = log(r_on / r_off).
* Eq 8 : the unbiased system undergoes a pitchfork at a_p = r / w_s, i.e.
         a = 2r; below -> 1 fixed point, above -> 3 (two stable + one unstable).
* The deterministic flow (sigma_noise = 0) relaxes to the prior fixed point.
* Parameter algebra round-trips (StructuralParams -> ReducedParams).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dci import (  # noqa: E402
    ReducedParams,
    StructuralParams,
    pitchfork_threshold_ap,
    simulate,
    find_fixed_points,
)


def test_prior_fixed_point_eq7():
    """a = 0 -> single fixed point at log(r_on/r_off) (Eq 7)."""
    p = ReducedParams(r_on=0.6, r_off=0.4, a=0.0, w=0.0)
    fps = find_fixed_points(p)
    assert len(fps) == 1, f"expected 1 fixed point, got {len(fps)}"
    L_star, stable = fps[0]
    assert stable
    expected = np.log(0.6 / 0.4)
    assert abs(L_star - expected) < 1e-6, (L_star, expected)


def test_pitchfork_threshold_eq8():
    """Unbiased system: 1 fixed point below a=2r, 3 above (pitchfork, Eq 8)."""
    r = 0.5  # r_on = r_off = r
    w_s = 1.0
    ap_crit = pitchfork_threshold_ap(r, w_s)  # = 0.5
    assert abs(ap_crit - 0.5) < 1e-12

    # below threshold: a = 2*ap*w_s < 2r -> one fixed point
    p_below = ReducedParams(r_on=r, r_off=r, a=2 * (ap_crit * 0.8) * w_s, w=0.0)
    assert len(find_fixed_points(p_below)) == 1

    # above threshold -> three fixed points (two stable, one unstable at 0)
    p_above = ReducedParams(r_on=r, r_off=r, a=2 * (ap_crit * 1.5) * w_s, w=0.0)
    fps = find_fixed_points(p_above)
    assert len(fps) == 3, [round(f[0], 3) for f in fps]
    n_stable = sum(s for _, s in fps)
    assert n_stable == 2
    # symmetric about 0
    Ls = sorted(f[0] for f in fps)
    assert abs(Ls[0] + Ls[2]) < 1e-6
    assert abs(Ls[1]) < 1e-6


def test_deterministic_relaxes_to_prior():
    """With no noise, L relaxes to the prior fixed point log(r_on/r_off)."""
    p = ReducedParams(r_on=0.6, r_off=0.4, a=0.0, w=1.0,
                      mu_noise=0.0, sigma_noise=0.0)
    L = simulate(p, n_steps=20000, dt=0.01, L0=0.0, n_traj=1)
    expected = np.log(0.6 / 0.4)
    assert abs(L[-1] - expected) < 1e-3, (L[-1], expected)


def test_param_algebra_roundtrip():
    """Structural -> reduced derives the documented coefficients."""
    s = StructuralParams(w_s=0.9, a_s=0.3, a_p=1.2, r_on=0.55, r_off=0.45,
                         mu_int=1.0, sigma_int=1.0)
    r = s.to_reduced()
    assert abs(r.a - 2 * 1.2 * 0.9) < 1e-12
    w_int = (2 * 1.0 / 1.0**2) * (2 * 0.9 - 1)
    assert abs(s.w_int - w_int) < 1e-12
    assert abs(r.w - w_int * (1 + 2 * 0.9 * 0.3)) < 1e-12
    assert abs(r.bias - 0.1) < 1e-12


def test_bistable_simulation_switches():
    """Strong descending loops -> trajectory visits both attractors (bistable)."""
    # unbiased, strong loops (a > 2r), realistic continuous gain w_int ~ 0.8
    p = ReducedParams(r_on=0.5, r_off=0.5, a=1.5, w=0.8,
                      mu_noise=0.0, sigma_noise=1.0)
    rng = np.random.default_rng(0)
    L = simulate(p, n_steps=200000, dt=0.01, L0=0.0, n_traj=1, rng=rng)
    fps = find_fixed_points(p)
    stable_L = sorted(f[0] for f in fps if f[1])
    assert len(stable_L) == 2
    # trajectory should spend time near both stable fixed points
    near_pos = np.mean(L > 0.5 * stable_L[1])
    near_neg = np.mean(L < 0.5 * stable_L[0])
    assert near_pos > 0.05 and near_neg > 0.05, (near_pos, near_neg)


def test_committed_state_hysteresis():
    """Hysteresis debouncing: small jitter at 0 does not flip the percept."""
    from dci import committed_state, switch_indices
    # rises past +1, jitters around 0 (should NOT switch), then drops past -1
    L = np.array([0.0, 0.5, 1.2, 0.3, -0.4, 0.2, -1.5, -0.3, 0.6])
    s = committed_state(L, threshold=1.0)
    # one genuine switch (+1 -> -1) at the -1.5 sample
    assert switch_indices(L, threshold=1.0).size == 1
    # raw sign would have counted several spurious crossings
    assert switch_indices(L, threshold=0.0).size > 1


def test_switches_min_dwell_ignores_flicker():
    """A sub-tau flip is discarded; a sustained flip counts once."""
    from dci import switches_min_dwell
    dt = 0.01
    # +1 for 1.0s, a 0.03s negative flicker (< tau) that returns to +1, then a
    # sustained 1.0s negative segment (>= tau).
    L = np.concatenate([
        np.full(100, 1.0),
        np.full(3, -1.0),     # flicker: must NOT count at tau = 0.5
        np.full(100, 1.0),
        np.full(100, -1.0),   # sustained: counts as one down switch
    ])
    assert switches_min_dwell(L, dt, tau=0.5) == (1, 0)
    # tau = 0 is the raw sign-crossing count: +->-, -->+, +->- = 2 down, 1 up
    assert switches_min_dwell(L, dt, tau=0.0) == (2, 1)


def test_phase_durations_exponential():
    """Bistable dwell times follow an ~exponential (gamma shape k ~ 1)."""
    from scipy import stats
    from dci import phase_durations_ensemble
    # a = 1.5 (moderate wells) gives plenty of switches quickly and a clean
    # exponential; very deep wells would need far longer runs for statistics.
    p = ReducedParams(r_on=0.5, r_off=0.5, a=1.5, w=0.8, sigma_noise=1.0)
    rng = np.random.default_rng(3)
    L2d = simulate(p, n_steps=100000, dt=0.01, n_traj=40, L0=0.0, rng=rng)
    durs = phase_durations_ensemble(L2d, 0.01, threshold=0.8)
    durs = durs[durs > 0]
    assert durs.size > 500, durs.size
    k, loc, theta = stats.gamma.fit(durs / durs.mean(), floc=0)
    assert 0.6 < k < 1.5, f"gamma shape k={k:.2f} not ~ exponential"


def test_em_equals_runm_scaling():
    """EM with gain w_int == runm.m dt-scaling with ws = 10*w_int (dt=0.01).

    Driving both update rules with the *same* noise sequence must give an
    identical trajectory, confirming the noise-convention reconciliation.
    """
    dt = 0.01
    w_int = 0.8
    ws = 10 * w_int
    r_on, r_off, a = 0.5, 0.5, 1.5
    bias = r_on - r_off
    rng = np.random.default_rng(7)
    xi = rng.standard_normal(5000)

    L_em = L_runm = 0.0
    sqdt = np.sqrt(dt)
    for k in range(5000):
        det = a * L_em + r_on * np.exp(-L_em) - r_off * np.exp(L_em) + bias
        L_em = L_em + dt * det + w_int * sqdt * xi[k]
        det2 = a * L_runm + r_on * np.exp(-L_runm) - r_off * np.exp(L_runm) + bias
        L_runm = L_runm + dt * (det2 + ws * xi[k])
    assert abs(L_em - L_runm) < 1e-9, (L_em, L_runm)


def test_recovery_param_mapping_roundtrip():
    """params_to_reduced and reduced_to_params invert each other (r fixed)."""
    from dci import params_to_reduced, reduced_to_params
    r_on, r_off, a, ws = params_to_reduced(ws=50.0, alpha=0.7, a=4.0, r=2.0)
    assert abs(r_on - 1.4) < 1e-12 and abs(r_off - 0.6) < 1e-12
    assert ws == 50.0
    alpha, a2 = reduced_to_params(r_on, r_off, a)
    assert abs(alpha - 0.7) < 1e-12 and abs(a2 - 4.0) < 1e-12


def test_survival_curve_matches_runm():
    """Vectorized ensemble survival ~ the scalar runm.m port (same convention)."""
    from dci import survival_curve
    from dci.model import run_intermittent_single
    ws, alpha, a, r = 40.0, 0.7, 4.0, 2.0
    r_on, r_off = alpha * r, (1 - alpha) * r
    Ton, Toff = 10, 50
    # ensemble estimate
    pp, po = survival_curve(ws, alpha, a, Ton=Ton, Toff_list=(Toff,), r=r,
                            n_traj=4000, n_repeat=20,
                            rng=np.random.default_rng(0))
    # scalar faithful port (one long trajectory)
    prperson, prpersoff = run_intermittent_single(
        ws, r_on, r_off, a, Ton, Toff, Nrepeat=2000,
        rng=np.random.default_rng(1))
    assert abs(pp[0] - prperson) < 0.05, (pp[0], prperson)
    assert abs(po[0] - prpersoff) < 0.05, (po[0], prpersoff)


def test_recovery_alpha_selfconsistency():
    """Recovering from data at known params returns the bias alpha (robust param)."""
    from dci import survival_curve, fit_survival
    Toff = (1, 10, 50, 200)   # drop 500 to keep the test quick
    data = survival_curve(50.0, 0.8, 4.0, Toff_list=Toff, n_traj=500, n_repeat=12,
                          rng=np.random.default_rng(7))
    est = fit_survival(data, Toff_list=Toff, n_traj=500, n_repeat=12,
                       n_restarts=3, model_seed=0, rng=np.random.default_rng(0))
    assert abs(est["alpha"] - 0.8) < 0.08, est["alpha"]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
