"""Core dynamical circular inference (dCI) integrator.

Implements the stochastic differential equation (Eq 6 of Leptourgos et al.,
2020) via the Euler--Maruyama scheme the paper states it uses:

    L <- L + dt * (a*L + r_on*exp(-L) - r_off*exp(L) + (r_on - r_off)
                   + w * mu_noise)
         + w * sigma_noise * sqrt(dt) * xi          (xi ~ Normal(0, 1))

i.e. the deterministic part scales with dt and the diffusion with sqrt(dt).
The sensory observation is S ~ Normal(mu_noise, sigma_noise^2); its mean
contributes a drift (w*mu_noise) and its fluctuation a diffusion
(w*sigma_noise). Here ``w`` is the sensory gain w*_int (= w_int when there are
no ascending loops), an O(1) quantity -- the value quoted in the paper figures
(e.g. w_int = 0.8 for Fig 2A).

Relation to the MATLAB ``runm.m``: that code writes the update as
``L += dt*(drift + ws*randn)`` with dt = 0.01 and ``ws = 10*w_int`` (the ``*10``
in partostatdir.m). Since dt*10 = 0.1 = sqrt(dt) at dt = 0.01, runm.m's noise
term ``dt*ws*randn`` equals ``sqrt(dt)*w_int*randn`` -- exactly the diffusion
above. So this integrator reproduces runm.m when called with w = ws/10, and is
the correct (dt-independent) Euler--Maruyama scheme in general.

During OFF periods (intermittent presentation) no sensory input is delivered,
so the entire ``w * S`` term (drift and diffusion) is switched off via
``stim_mask``.
"""

from __future__ import annotations

import numpy as np

from .params import ReducedParams

DT_DEFAULT = 0.01

# Guardrail against the stiff -r_off*exp(L) term overflowing under explicit
# Euler. Legitimate dynamics stay well within +/-10 (the paper's figures span
# |L| <~ 6), so clipping here never touches real trajectories; it only tames
# pathological blow-ups into a finite value instead of inf/nan.
_CLIP = 50.0


def drift(L, p: ReducedParams):
    """Deterministic drift f(L) = -Phi(L) + a*L (Eqs 5-6), no sensory term.

    This is the right-hand side that governs the dynamics in the absence of
    sensory input (OFF periods), and whose integral defines the energy
    landscape (see ``landscape.py``).
    """
    Lc = np.clip(L, -_CLIP, _CLIP)
    return (p.a * Lc
            + p.r_on * np.exp(-Lc)
            - p.r_off * np.exp(Lc)
            + (p.r_on - p.r_off))


def simulate(p: ReducedParams,
             n_steps: int,
             dt: float = DT_DEFAULT,
             stim_mask: np.ndarray | None = None,
             L0: float = 0.0,
             n_traj: int = 1,
             rng: np.random.Generator | None = None):
    """Integrate the dCI SDE for ``n_steps`` steps and ``n_traj`` trajectories.

    The realizations are independent, so they are advanced as a vector in time
    (the only Python-level loop is over time); this is what makes long parameter
    sweeps tractable in pure NumPy.

    Parameters
    ----------
    p : ReducedParams
        Model parameters (rates, auto-amplification a, gain w, noise stats).
    n_steps : int
        Number of Euler steps. Total simulated time is n_steps * dt.
    dt : float
        Integration step (0.01 in the paper).
    stim_mask : ndarray of bool, shape (n_steps,), optional
        True where the stimulus is present (sensory noise injected). None means
        continuous presentation (stimulus always on).
    L0 : float
        Initial log-odds (0 in the paper).
    n_traj : int
        Number of independent trajectories to run in parallel.
    rng : np.random.Generator, optional
        Random generator (defaults to np.random.default_rng()).

    Returns
    -------
    ndarray, shape (n_traj, n_steps) or (n_steps,) if n_traj == 1.
        The log-odds trajectory(ies), including the initial value at index 0.
    """
    if rng is None:
        rng = np.random.default_rng()
    if stim_mask is None:
        stim_on = np.ones(n_steps, dtype=bool)
    else:
        stim_on = np.asarray(stim_mask, dtype=bool)
        if stim_on.shape[0] != n_steps:
            raise ValueError("stim_mask length must equal n_steps")

    out = np.empty((n_traj, n_steps), dtype=np.float64)
    L = np.full(n_traj, float(L0), dtype=np.float64)
    out[:, 0] = L

    a, r_on, r_off = p.a, p.r_on, p.r_off
    w, mu, sigma = p.w, p.mu_noise, p.sigma_noise
    bias = r_on - r_off
    sqdt = np.sqrt(dt)
    diff_coef = w * sigma * sqdt   # diffusion amplitude per step
    drift_sens = w * mu            # sensory mean -> drift

    for t in range(1, n_steps):
        Lc = np.clip(L, -_CLIP, _CLIP)
        det = a * Lc + r_on * np.exp(-Lc) - r_off * np.exp(Lc) + bias
        if stim_on[t - 1]:
            xi = rng.standard_normal(n_traj)
            L = L + dt * (det + drift_sens) + diff_coef * xi
        else:
            L = L + dt * det
        out[:, t] = L

    if n_traj == 1:
        return out[0]
    return out


def make_intermittent_mask(n_steps: int, ton_steps: int, toff_steps: int) -> np.ndarray:
    """Boolean stimulus mask for intermittent presentation.

    Repeating blocks of ``ton_steps`` ON followed by ``toff_steps`` OFF.
    """
    period = ton_steps + toff_steps
    phase = np.arange(n_steps) % period
    return phase < ton_steps


def run_intermittent_single(w, r_on, r_off, a, Ton, Toff, Nrepeat,
                            dt: float = DT_DEFAULT, rng=None):
    """Faithful single-trajectory port of the original MATLAB ``runm.m``.

    Returns the survival probabilities (probability that a percept persists
    across one OFF gap) for the two interpretations.

    Parameters mirror runm.m exactly (dt-scaling of the noise): ``w`` is its
    ``ws`` argument, which relates to the continuous gain as ws = 10*w_int at
    dt = 0.01. Ton/Toff are the ON/OFF durations *in integration steps*, Nrepeat
    the number of ON/OFF cycles. Kept as a faithful reference for validation;
    the vectorized survival-probability routine lives in ``analysis.py``.

    Returns
    -------
    (prperson, prpersoff) : float, float
        Survival probability of interpretation SFA (L>0) and SFB (L<0).
    """
    if rng is None:
        rng = np.random.default_rng()
    period = Ton + Toff
    Ntime = min(period * (Nrepeat + 1), 1_000_000)

    L = 0.0
    resp = []
    bias = r_on - r_off
    for t in range(1, Ntime):  # mirror MATLAB t = 1 .. Ntime-1
        mask_on = (t % period) < Ton
        n = rng.standard_normal() if mask_on else 0.0
        Lc = min(max(L, -_CLIP), _CLIP)
        L = L + dt * (a * Lc + r_on * np.exp(-Lc) - r_off * np.exp(Lc)
                      + bias + w * n)
        if not np.isfinite(L):
            return 0.5, 0.5
        if (t + 1) % period == (Ton - 1):
            resp.append(np.sign(L))

    resp = np.asarray(resp)
    d = np.diff(resp)
    prpersoff = 1.0 - np.sum(d > 0) / (np.sum(resp == -1) + 1e-4)
    prperson = 1.0 - np.sum(d < 0) / (np.sum(resp == 1) + 1e-4)
    return prperson, prpersoff
