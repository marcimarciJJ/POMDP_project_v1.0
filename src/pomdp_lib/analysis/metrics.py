"""
Performance Metrics for POMDP Solver Comparison.
Compute quantitative metrics from solver results.
"""

import numpy as np
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from solvers.unified_interface import SolverResult


def compute_policy_agreement(policy1: np.ndarray, policy2: np.ndarray) -> float:
    """
    Compute fraction of states where two policies agree.

    Args:
        policy1: first policy array
        policy2: second policy array

    Returns:
        Agreement rate in [0, 1]
    """
    return float(np.mean(policy1 == policy2))


def compute_value_l2(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute L2 distance between two value functions.

    Args:
        v1: first value function
        v2: second value function

    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(v1 - v2))


def compute_value_linf(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute L-infinity (max) distance between two value functions.

    Args:
        v1: first value function
        v2: second value function

    Returns:
        Maximum absolute difference
    """
    return float(np.max(np.abs(v1 - v2)))


def compute_solver_metrics(result: SolverResult) -> Dict[str, Any]:
    """
    Compute summary metrics for a single solver result.

    Args:
        result: SolverResult from a solver

    Returns:
        Dictionary of metric name -> value
    """
    return {
        'solver_name': result.solver_name,
        'time_ms': float(result.convergence_time * 1000),
        'iterations': int(result.iterations),
        'final_bellman_error': float(result.final_bellman_error),
        'is_converged': bool(result.is_converged),
        'J_b0': float(result.convergence_trace[-1]) if result.convergence_trace else 0.0,
        'mean_value': float(np.mean(result.value_function)),
        'std_value': float(np.std(result.value_function)),
        'min_value': float(np.min(result.value_function)),
        'max_value': float(np.max(result.value_function)),
    }


def compute_pairwise_metrics(results: Dict[str, SolverResult]) -> Dict[str, Dict]:
    """
    Compute pairwise comparison metrics for all solver pairs.

    Args:
        results: dictionary mapping solver name to SolverResult

    Returns:
        Nested dictionary: pair_key -> metric_name -> value
    """
    names = list(results.keys())
    pairwise = {}

    for i, s1 in enumerate(names):
        for s2 in names[i + 1:]:
            r1, r2 = results[s1], results[s2]
            key = f"{s1}_vs_{s2}"
            pairwise[key] = {
                'policy_agreement': compute_policy_agreement(r1.policy, r2.policy),
                'value_l2_distance': compute_value_l2(r1.value_function, r2.value_function),
                'value_linf_distance': compute_value_linf(r1.value_function, r2.value_function),
                'value_mean_abs_diff': float(np.mean(np.abs(r1.value_function - r2.value_function))),
            }

    return pairwise


def summarize_all(results: Dict[str, SolverResult]) -> Dict[str, Any]:
    """
    Generate a full metrics summary for all solver results.

    Args:
        results: dictionary mapping solver name to SolverResult

    Returns:
        Summary dictionary with per-solver and pairwise metrics
    """
    return {
        'per_solver': {name: compute_solver_metrics(r) for name, r in results.items()},
        'pairwise': compute_pairwise_metrics(results),
    }
