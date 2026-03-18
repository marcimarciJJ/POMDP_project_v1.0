"""
Tests for POMDP solver correctness.
Validates QMDP and PBVI on the 3x3 model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from pomdp_lib.models import model_3x3
from pomdp_lib.solvers.qmdp import QMDPSolver
from pomdp_lib.solvers.pbvi import PBVISolver


def make_model_3x3():
    """Build model dict from the 3x3 module."""
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


# ---- QMDP Tests ----

def test_qmdp_returns_result():
    """QMDP should return a SolverResult object."""
    model = make_model_3x3()
    solver = QMDPSolver(model)
    result = solver.solve(epsilon=1e-4, max_iterations=50)
    assert result is not None


def test_qmdp_policy_shape():
    """QMDP policy should have shape (nS,)."""
    model = make_model_3x3()
    nS = len(model['S'])
    solver = QMDPSolver(model)
    result = solver.solve(epsilon=1e-4, max_iterations=50)
    assert result.policy.shape == (nS,)


def test_qmdp_policy_valid_actions():
    """QMDP policy actions must be valid action indices."""
    model = make_model_3x3()
    nA = len(model['A'])
    solver = QMDPSolver(model)
    result = solver.solve(epsilon=1e-4, max_iterations=50)
    assert np.all(result.policy >= 0)
    assert np.all(result.policy < nA)


def test_qmdp_value_function_shape():
    """QMDP value function should have shape (nS,)."""
    model = make_model_3x3()
    nS = len(model['S'])
    solver = QMDPSolver(model)
    result = solver.solve(epsilon=1e-4, max_iterations=50)
    assert result.value_function.shape == (nS,)


def test_qmdp_convergence_trace():
    """QMDP convergence trace must be non-empty."""
    model = make_model_3x3()
    solver = QMDPSolver(model)
    result = solver.solve(epsilon=1e-4, max_iterations=50)
    assert len(result.convergence_trace) > 0


def test_qmdp_timing():
    """QMDP convergence_time should be positive."""
    model = make_model_3x3()
    solver = QMDPSolver(model)
    result = solver.solve(epsilon=1e-4, max_iterations=50)
    assert result.convergence_time >= 0.0


def test_qmdp_iterations():
    """QMDP iterations should be at least 1."""
    model = make_model_3x3()
    solver = QMDPSolver(model)
    result = solver.solve(epsilon=1e-4, max_iterations=50)
    assert result.iterations >= 1


# ---- PBVI Tests ----

def test_pbvi_returns_result():
    """PBVI should return a SolverResult object."""
    model = make_model_3x3()
    solver = PBVISolver(model)
    result = solver.solve(epsilon=1e-3, max_iterations=20, num_belief_points=10)
    assert result is not None


def test_pbvi_policy_shape():
    """PBVI policy should have shape (nS,)."""
    model = make_model_3x3()
    nS = len(model['S'])
    solver = PBVISolver(model)
    result = solver.solve(epsilon=1e-3, max_iterations=20, num_belief_points=10)
    assert result.policy.shape == (nS,)


def test_pbvi_policy_valid_actions():
    """PBVI policy actions must be valid action indices."""
    model = make_model_3x3()
    nA = len(model['A'])
    solver = PBVISolver(model)
    result = solver.solve(epsilon=1e-3, max_iterations=20, num_belief_points=10)
    assert np.all(result.policy >= 0)
    assert np.all(result.policy < nA)


def test_pbvi_value_function_shape():
    """PBVI value function should have shape (nS,)."""
    model = make_model_3x3()
    nS = len(model['S'])
    solver = PBVISolver(model)
    result = solver.solve(epsilon=1e-3, max_iterations=20, num_belief_points=10)
    assert result.value_function.shape == (nS,)


def test_pbvi_convergence_trace():
    """PBVI convergence trace must be non-empty."""
    model = make_model_3x3()
    solver = PBVISolver(model)
    result = solver.solve(epsilon=1e-3, max_iterations=20, num_belief_points=10)
    assert len(result.convergence_trace) > 0


def test_both_solvers_same_num_states():
    """QMDP and PBVI should return the same number of states."""
    model = make_model_3x3()
    qmdp = QMDPSolver(model)
    pbvi = PBVISolver(model)
    r_q = qmdp.solve(epsilon=1e-4, max_iterations=50)
    r_p = pbvi.solve(epsilon=1e-3, max_iterations=20, num_belief_points=10)
    assert r_q.num_states == r_p.num_states


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
