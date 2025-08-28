"""
stochid: Stochastic and Deterministic ODE/SDE Identification Framework
"""

# Expose configs
from .config import BistableConfig, VDPConfig

# Expose simulators
from .simulators import (
    simulate_bistable_records,
    simulate_vdp_records,
    em_step_bistable,
    vdp_drift_np,
)

# Expose datasets + loaders
from .datasets import (
    SigmaMeanDatasetBistable,
    SigmaMeanDatasetVDP,
    build_loaders,
    collate_sigma,
)

# Expose models + integrators
from .models import Drift1D, DriftND
from .integrators import UTIntegrator1D, StateIntegratorND

# Expose training/eval utilities
from .train import train_bistable, train_vdp
from .eval import whitebox_test_bistable, rollout_truth_vdp, rollout_learned_state, pct_error
from .viz import plot_overlaid_traj

__all__ = [
    # Configs
    "BistableConfig", "VDPConfig",
    # Simulators
    "simulate_bistable_records", "simulate_vdp_records", "em_step_bistable", "vdp_drift_np",
    # Datasets
    "SigmaMeanDatasetBistable", "SigmaMeanDatasetVDP", "build_loaders", "collate_sigma",
    # Models
    "Drift1D", "DriftND",
    # Integrators
    "UTIntegrator1D", "StateIntegratorND",
    # Training/Eval
    "train_bistable", "train_vdp",
    "whitebox_test_bistable", "rollout_truth_vdp", "rollout_learned_state", "pct_error",
    # Viz
    "plot_overlaid_traj",
]
