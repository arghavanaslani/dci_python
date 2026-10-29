"""Python replication of the dynamical circular inference (dCI) model.

Leptourgos, Bouttier, Jardri & Deneve (2020), PLoS Comput Biol.
"""

from .params import (
    ReducedParams,
    StructuralParams,
    pitchfork_threshold_ap,
)
from .model import (
    simulate,
    drift,
    make_intermittent_mask,
    run_intermittent_single,
    DT_DEFAULT,
)
from .landscape import (
    potential,
    drift_deriv,
    find_fixed_points,
)
from .analysis import (
    committed_state,
    percept,
    switch_indices,
    switches_min_dwell,
    phase_durations,
    phase_durations_ensemble,
    phase_durations_by_percept,
    relative_predominance,
    mean_phase_duration_by_percept,
    alternation_rate,
)
from .recovery import (
    survival_curve,
    fit_survival,
    recover_once,
    run_recovery,
    params_to_reduced,
    reduced_to_params,
)

__all__ = [
    "ReducedParams",
    "StructuralParams",
    "pitchfork_threshold_ap",
    "simulate",
    "drift",
    "make_intermittent_mask",
    "run_intermittent_single",
    "DT_DEFAULT",
    "potential",
    "drift_deriv",
    "find_fixed_points",
    "committed_state",
    "percept",
    "switch_indices",
    "switches_min_dwell",
    "phase_durations",
    "phase_durations_ensemble",
    "phase_durations_by_percept",
    "relative_predominance",
    "mean_phase_duration_by_percept",
    "alternation_rate",
    "survival_curve",
    "fit_survival",
    "recover_once",
    "run_recovery",
    "params_to_reduced",
    "reduced_to_params",
]
