"""Parameter algebra for the dynamical circular inference (dCI) model.

Reference: Leptourgos, Bouttier, Jardri & Deneve (2020), PLoS Comput Biol,
"A functional theory of bistable perception based on dynamical circular
inference". Equation numbers below refer to the main text / Table 1.

The model integrates a log-odds variable L (confidence in interpretation
SFA vs SFB):

    dL/dt = -Phi(L) + a * L + w_star_int * S                         (Eq 6)

with the nonlinear leak

    -Phi(L) = (r_on * exp(-L) - r_off * exp(L)) + (r_on - r_off)      (Eq 5)

and S the (noisy) sensory observation, S ~ Normal(mu_noise, sigma_noise^2).

Two parameter "views" are provided:

* StructuralParams: the interpretable quantities of the generative model
  (feed-forward weight w_s, ascending/descending loop strengths a_s/a_p,
  likelihood mean/std mu_int/sigma_int, transition rates r_on/r_off).
  From these the simulation coefficients are *derived*.

* ReducedParams: the 4 free parameters actually fed to the integrator
  (r_on, r_off, a, w) -- this matches the arguments of the original MATLAB
  ``runm.m`` (where its ``ws`` argument is the sensory gain ``w``, i.e.
  w_star_int, NOT the feed-forward weight w_s).

Derived coefficients (Table 1):

    w_int      = (2 * mu_int / sigma_int**2) * (2 * w_s - 1)   # gain, no asc. loops
    w_star_int = w_int * (1 + 2 * w_s * a_s)                   # gain, with asc. loops
    a          = 2 * a_p * w_s                                 # auto-amplification
    b          = r_on - r_off                                  # bias
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ReducedParams:
    """The 4 free parameters fed directly to the integrator.

    Attributes
    ----------
    r_on, r_off : float
        Transition rates 0->1 and 1->0 (Eq 2, 3). Their sum r = r_on + r_off
        sets the volatility; their difference b = r_on - r_off sets the bias.
    a : float
        Auto-amplification from descending loops (a = 2 a_p w_s).
    w : float
        Sensory gain multiplying the observation S (this is w_star_int; it is
        the ``ws`` argument of the MATLAB runm.m).
    mu_noise : float
        Mean (drift) of the sensory observation distribution. 0 = ambiguous.
    sigma_noise : float
        Std of the sensory observation distribution (1.0 in the paper).
    """

    r_on: float
    r_off: float
    a: float
    w: float
    mu_noise: float = 0.0
    sigma_noise: float = 1.0

    @property
    def r(self) -> float:
        """Total rate r = r_on + r_off (volatility)."""
        return self.r_on + self.r_off

    @property
    def bias(self) -> float:
        """Bias b = r_on - r_off (Table 1)."""
        return self.r_on - self.r_off

    @property
    def prior_fixed_point(self) -> float:
        """Fixed point of the loop-free system, L = log(r_on / r_off) (Eq 7)."""
        return math.log(self.r_on / self.r_off)


@dataclass(frozen=True)
class StructuralParams:
    """Interpretable generative-model parameters; derives the simulation gains.

    Defaults follow the paper's "Simulations" section: sigma_noise = sigma_int
    = 1, mu_int = 1, L0 = 0.
    """

    w_s: float = 0.9          # feed-forward weight, P(X3D=1|X2D=1) in [0.5, 1]
    a_s: float = 0.0          # ascending (climbing) loop strength
    a_p: float = 0.0          # descending loop strength
    r_on: float = 0.5         # transition rate 0->1
    r_off: float = 0.5        # transition rate 1->0
    mu_int: float = 1.0       # mean of internal likelihood (+/- mu_int)
    sigma_int: float = 1.0    # std of internal likelihood
    mu_noise: float = 0.5     # drift of sensory observation (0 = ambiguous)
    sigma_noise: float = 1.0  # std of sensory observation

    @property
    def w_int(self) -> float:
        """Sensory gain without ascending loops (Table 1)."""
        return (2.0 * self.mu_int / self.sigma_int**2) * (2.0 * self.w_s - 1.0)

    @property
    def w_star_int(self) -> float:
        """Sensory gain including ascending loops (Eq 6 / Table 1)."""
        return self.w_int * (1.0 + 2.0 * self.w_s * self.a_s)

    @property
    def a(self) -> float:
        """Auto-amplification from descending loops, a = 2 a_p w_s."""
        return 2.0 * self.a_p * self.w_s

    def to_reduced(self) -> ReducedParams:
        """Collapse to the 4-parameter integrator interface."""
        return ReducedParams(
            r_on=self.r_on,
            r_off=self.r_off,
            a=self.a,
            w=self.w_star_int,
            mu_noise=self.mu_noise,
            sigma_noise=self.sigma_noise,
        )


def pitchfork_threshold_ap(r: float, w_s: float) -> float:
    """Descending-loop strength at the pitchfork bifurcation, a_p = r / w_s (Eq 8).

    Here ``r`` is the common value of the two rates (r_on = r_off = r), as in
    Eq 8. Equivalently the auto-amplification at threshold is a = 2 a_p w_s = 2r,
    where f'(0) = a - r_on - r_off changes sign. At this value the prior fixed
    point loses stability and two symmetric attractors appear.
    """
    return r / w_s
