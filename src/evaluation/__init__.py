"""Evaluation metrics and publication plotting for General-Purpose Dual-Uncertainty RAG."""
from .metrics import evaluate_policy_on_split, compute_expected_calibration_error, compute_selective_risk_aurc
from .plotting import generate_all_publication_figures
