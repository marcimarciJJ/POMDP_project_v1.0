"""
Tests for POMDP analysis tools.
Validates comparator, metrics, and result formatting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from pomdp_lib.models import model_3x3
from pomdp_lib.solvers.qmdp import QMDPSolver
from pomdp_lib.solvers.pbvi import PBVISolver
from pomdp_lib.analysis.comparator import MultiSolverComparator
from pomdp_lib.analysis.metrics import (
    compute_policy_agreement,
    compute_value_l2,
    compute_solver_metrics,
    compute_pairwise_metrics,
)


def make_model_3x3():
    return {
        'S': model_3x3.S,
        'A': model_3x3.A,
        'O': model_3x3.O,
        'T': model_3x3.T,
        'Z': model_3x3.Z,
        'R': model_3x3.R,
        'gamma': model_3x3.gamma,
        'b0': model_3x3.b0,
        'meta': {'width': 3, 'height': 3, 'environment_id': '3x3'}
    }


@pytest.fixture
def solver_results():
    """Provide solved QMDP and PBVI results for tests."""
    model = make_model_3x3()
    qmdp = QMDPSolver(model).solve(epsilon=1e-4, max_iterations=50)
    pbvi = PBVISolver(model).solve(epsilon=1e-3, max_iterations=20, num_belief_points=10)
    return {'QMDP': qmdp, 'PBVI': pbvi}


# ---- Metric Tests ----

def test_policy_agreement_self(solver_results):
    """A policy should agree with itself 100%."""
    r = solver_results['QMDP']
    agreement = compute_policy_agreement(r.policy, r.policy)
    assert np.isclose(agreement, 1.0)


def test_policy_agreement_range(solver_results):
    """Policy agreement between two different solvers should be in [0, 1]."""
    r1, r2 = solver_results['QMDP'], solver_results['PBVI']
    agreement = compute_policy_agreement(r1.policy, r2.policy)
    assert 0.0 <= agreement <= 1.0


def test_value_l2_self(solver_results):
    """L2 distance of value function to itself should be 0."""
    v = solver_results['QMDP'].value_function
    assert np.isclose(compute_value_l2(v, v), 0.0)


def test_value_l2_nonneg(solver_results):
    """L2 distance between two value functions must be non-negative."""
    v1 = solver_results['QMDP'].value_function
    v2 = solver_results['PBVI'].value_function
    assert compute_value_l2(v1, v2) >= 0.0


def test_solver_metrics_keys(solver_results):
    """Solver metrics should contain required keys."""
    metrics = compute_solver_metrics(solver_results['QMDP'])
    required = {'solver_name', 'time_ms', 'iterations', 'final_bellman_error',
                'is_converged', 'J_b0', 'mean_value', 'std_value'}
    assert required.issubset(metrics.keys())


def test_pairwise_metrics_structure(solver_results):
    """Pairwise metrics should have entries for each pair of solvers."""
    pairwise = compute_pairwise_metrics(solver_results)
    assert 'QMDP_vs_PBVI' in pairwise
    entry = pairwise['QMDP_vs_PBVI']
    assert 'policy_agreement' in entry
    assert 'value_l2_distance' in entry


# ---- Comparator Tests ----

def test_comparator_init(solver_results):
    """MultiSolverComparator should initialize without errors."""
    comparator = MultiSolverComparator(solver_results)
    assert comparator is not None


def test_comparator_compare_policies(solver_results):
    """compare_policies should return a non-empty dict."""
    comparator = MultiSolverComparator(solver_results)
    result = comparator.compare_policies()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_comparator_compare_values(solver_results):
    """compare_values should return a non-empty dict."""
    comparator = MultiSolverComparator(solver_results)
    result = comparator.compare_values()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_comparator_compare_efficiency(solver_results):
    """compare_efficiency should list all solver names."""
    comparator = MultiSolverComparator(solver_results)
    result = comparator.compare_efficiency()
    assert 'QMDP' in result
    assert 'PBVI' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
