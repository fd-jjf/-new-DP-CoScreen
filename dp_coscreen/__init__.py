"""
DP-CoScreen: Differentially Private Orthogonal Feature Screening for Industrial IoT Data
"""

__version__ = "1.0.0"
__author__ = "DP-CoScreen Research Team"

from .privacy import (
    calibrate_agm,
    compute_l2_sensitivity,
    sample_symmetric_gaussian_noise,
    PrivacyBudget,
)
from .schur import (
    project_psd_cone,
    schur_orthogonal_screening,
    compute_spectral_suboptimality_bound,
)
from .preprocessing import (
    PHYSICAL_DOMAIN_BOUNDS,
    FixedFeatureMap,
    normalize_attribute,
    clip_augmented_vectors,
    partition_dirichlet_noniid,
)
from .screening import DPCoScreen
from .evaluation import evaluate_classifier, aggregate_split_statistics

__all__ = [
    "DPCoScreen",
    "calibrate_agm",
    "compute_l2_sensitivity",
    "sample_symmetric_gaussian_noise",
    "PrivacyBudget",
    "project_psd_cone",
    "schur_orthogonal_screening",
    "compute_spectral_suboptimality_bound",
    "PHYSICAL_DOMAIN_BOUNDS",
    "FixedFeatureMap",
    "normalize_attribute",
    "clip_augmented_vectors",
    "partition_dirichlet_noniid",
    "evaluate_classifier",
    "aggregate_split_statistics",
]
