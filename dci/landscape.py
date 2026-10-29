"""Deterministic ground truth for validating the stochastic engine.

These quantities depend only on the drift f(L) = -Phi(L) + a*L and require no
stochastic simulation. They give the analytic reference the switching model is
checked against: where the attractor wells sit (fixed points), how deep they
are (the potential barrier -> Kramers escape rate), and where bistability
appears (the pitchfork). Not tied to any figure -- a validation utility.

The system is a gradient flow dL/dt = f(L) = -dV/dL with potential

    V(L) = -a*L^2/2 + r_on*exp(-L) + r_off*exp(L) - (r_on - r_off)*L

(the negative integral of the drift). L flows downhill toward minima of V;
minima are stable fixed points, maxima are unstable ones.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .params import ReducedParams


def potential(L, p: ReducedParams, shift: bool = True):
    """Energy landscape V(L) (integral of -drift).

    If ``shift`` is True the curve is offset so its minimum over the evaluated
    points is 0 (for clean plotting, as in Fig 3).
    """
    L = np.asarray(L, dtype=float)
    V = (-0.5 * p.a * L**2
         + p.r_on * np.exp(-L)
         + p.r_off * np.exp(L)
         - (p.r_on - p.r_off) * L)
    if shift:
        V = V - np.min(V)
    return V


def drift_deriv(L, p: ReducedParams):
    """f'(L) = a - r_on*exp(-L) - r_off*exp(L). Stability: stable if f'(L) < 0."""
    L = np.asarray(L, dtype=float)
    return p.a - p.r_on * np.exp(-L) - p.r_off * np.exp(L)


def _drift_scalar(L, p: ReducedParams):
    return (p.a * L + p.r_on * np.exp(-L) - p.r_off * np.exp(L)
            + (p.r_on - p.r_off))


def find_fixed_points(p: ReducedParams,
                      L_range: tuple[float, float] = (-20.0, 20.0),
                      n_grid: int = 4000):
    """Locate fixed points of the deterministic flow (roots of the drift).

    Returns
    -------
    list of (L_star, stable) tuples, sorted by L_star.
        ``stable`` is True when f'(L_star) < 0 (a potential well).
    """
    grid = np.linspace(L_range[0], L_range[1], n_grid)
    f = _drift_scalar(grid, p)
    roots = []
    sign = np.sign(f)
    # bracket sign changes and refine
    idx = np.where(np.diff(sign) != 0)[0]
    for i in idx:
        a_, b_ = grid[i], grid[i + 1]
        try:
            r = brentq(_drift_scalar, a_, b_, args=(p,), xtol=1e-10)
        except ValueError:
            continue
        stable = drift_deriv(r, p) < 0
        roots.append((float(r), bool(stable)))
    # exact zeros on the grid (rare)
    for i in np.where(f == 0)[0]:
        roots.append((float(grid[i]), bool(drift_deriv(grid[i], p) < 0)))
    roots.sort(key=lambda t: t[0])
    # de-duplicate near-identical roots
    deduped = []
    for r, s in roots:
        if not deduped or abs(r - deduped[-1][0]) > 1e-6:
            deduped.append((r, s))
    return deduped
