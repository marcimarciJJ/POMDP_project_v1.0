"""
Tests for POMDP model validation.
Checks structure, normalization, and correctness of models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from pomdp_lib.models.model_3x3 import S, A, O, T, Z, R, b0, gamma
from pomdp_lib.models.model_5x5 import (S as S5, A as A5, O as O5,
                                         T as T5, Z as Z5, R as R5,
                                         b0 as b05, gamma as gamma5)


def test_model_3x3_shapes():
    """Check that 3x3 model arrays have the correct shapes."""
    nS = len(S)
    nA = len(A)
    nO = len(O)
    assert T.shape == (nA, nS, nS), f"T shape mismatch: {T.shape}"
    assert Z.shape == (nO, nS), f"Z shape mismatch: {Z.shape}"
    assert R.shape == (nA, nS), f"R shape mismatch: {R.shape}"
    assert len(b0) == nS


def test_model_3x3_transition_normalization():
    """Each row of T (for each action) must sum to 1."""
    T_sums = np.sum(T, axis=2)  # (nA, nS)
    assert np.allclose(T_sums, 1.0, atol=1e-6), "Transitions not normalized"


def test_model_3x3_observation_normalization():
    """For each state, observation probabilities must sum to 1."""
    Z_sums = np.sum(Z, axis=0)  # (nS,)
    assert np.allclose(Z_sums, 1.0, atol=1e-6), "Observations not normalized"


def test_model_3x3_belief_normalization():
    """Initial belief must sum to 1."""
    assert np.isclose(np.sum(b0), 1.0, atol=1e-6), "b0 not normalized"


def test_model_3x3_discount():
    """Discount factor must be in (0, 1)."""
    assert 0.0 < gamma < 1.0


def test_model_5x5_shapes():
    """Check that 5x5 model arrays have the correct shapes."""
    nS = len(S5)
    nA = len(A5)
    nO = len(O5)
    assert T5.shape == (nA, nS, nS)
    assert Z5.shape == (nO, nS)
    assert R5.shape == (nA, nS)
    assert len(b05) == nS


def test_model_5x5_transition_normalization():
    """Each row of T for 5x5 must sum to 1."""
    T_sums = np.sum(T5, axis=2)
    assert np.allclose(T_sums, 1.0, atol=1e-6)


def test_model_5x5_observation_normalization():
    """Observations for 5x5 must sum to 1 per state."""
    Z_sums = np.sum(Z5, axis=0)
    assert np.allclose(Z_sums, 1.0, atol=1e-6)


def test_model_nonneg_probabilities():
    """All transition and observation probabilities must be non-negative."""
    assert np.all(T >= 0), "Negative transitions in 3x3"
    assert np.all(Z >= 0), "Negative observations in 3x3"
    assert np.all(T5 >= 0), "Negative transitions in 5x5"
    assert np.all(Z5 >= 0), "Negative observations in 5x5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
