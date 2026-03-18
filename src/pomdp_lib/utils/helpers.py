"""
Helper utility functions for POMDP computations.
Common operations reused across solvers and analysis tools.
"""

import numpy as np
from typing import List, Optional, Tuple


def normalize_belief(b: np.ndarray) -> np.ndarray:
    """
    Normalize a belief vector so it sums to 1.

    Args:
        b: belief vector (may not sum to 1)

    Returns:
        Normalized belief vector
    """
    total = np.sum(b)
    if total < 1e-12:
        # If all zeros, return uniform belief
        return np.ones_like(b) / len(b)
    return b / total


def compute_belief_update(b: np.ndarray, a: int, o: int,
                          T: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Compute updated belief after taking action a and observing o.

    Uses Bayes' rule: b'(s') = eta * O(o|s',a) * sum_s T(s,a,s') * b(s)

    Args:
        b: current belief vector, shape (nS,)
        a: action index
        o: observation index
        T: transition matrix, shape (nA, nS, nS)
        Z: observation matrix, shape (nO, nS)

    Returns:
        Updated belief vector, shape (nS,)
    """
    # Predict step: sum over current states
    b_pred = T[a].T @ b  # (nS,)

    # Update step: weight by observation probability
    b_new = Z[o, :] * b_pred

    return normalize_belief(b_new)


def sample_from_distribution(probs: np.ndarray) -> int:
    """
    Sample an index from a probability distribution.

    Args:
        probs: probability distribution, must sum to 1

    Returns:
        Sampled index
    """
    return int(np.random.choice(len(probs), p=probs))


def compute_j_b0(b0: np.ndarray, value_function: np.ndarray) -> float:
    """
    Compute J(b0) = expected value under initial belief.

    Args:
        b0: initial belief vector
        value_function: value function over states

    Returns:
        Scalar expected value
    """
    return float(np.dot(b0, value_function))


def compute_policy_entropy(policy: np.ndarray, nA: int) -> float:
    """
    Compute entropy of a deterministic policy distribution over actions.

    Args:
        policy: policy array mapping states to actions
        nA: number of actions

    Returns:
        Shannon entropy of action distribution
    """
    counts = np.bincount(policy, minlength=nA).astype(float)
    probs = counts / counts.sum()
    # Avoid log(0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def bellman_error(V_old: np.ndarray, V_new: np.ndarray) -> float:
    """
    Compute the Bellman error (max absolute difference).

    Args:
        V_old: value function from previous iteration
        V_new: value function from current iteration

    Returns:
        Max absolute difference
    """
    return float(np.max(np.abs(V_new - V_old)))
