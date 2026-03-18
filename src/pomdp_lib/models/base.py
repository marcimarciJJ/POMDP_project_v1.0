"""
Base class for POMDP models.
Defines the interface and validation for all POMDP environments.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class POMDPModel:
    """
    Base class for POMDP models.
    A POMDP is defined by the 7-tuple (S, A, T, R, Omega, O, gamma).
    """

    def __init__(self, name: str = "POMDP"):
        """Initialize POMDP model with empty components."""
        self.name = name
        self.S: List[Any] = []
        self.A: List[str] = []
        self.O: List[str] = []
        self.T: Optional[np.ndarray] = None
        self.Z: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.gamma: float = 0.95
        self.b0: Optional[np.ndarray] = None
        self.meta: Dict[str, Any] = {}

    @property
    def nS(self) -> int:
        """Number of states."""
        return len(self.S)

    @property
    def nA(self) -> int:
        """Number of actions."""
        return len(self.A)

    @property
    def nO(self) -> int:
        """Number of observations."""
        return len(self.O)

    def to_dict(self) -> Dict[str, Any]:
        """Export model as a dictionary for use by solvers."""
        return {
            'S': self.S,
            'A': self.A,
            'O': self.O,
            'T': self.T,
            'Z': self.Z,
            'R': self.R,
            'gamma': self.gamma,
            'b0': self.b0,
            'meta': self.meta
        }

    def validate(self) -> bool:
        """
        Validate POMDP structure.
        Checks shapes and probability normalization.
        """
        try:
            assert self.T is not None, "Transition matrix T is None"
            assert self.Z is not None, "Observation matrix Z is None"
            assert self.R is not None, "Reward matrix R is None"
            assert self.b0 is not None, "Initial belief b0 is None"

            assert self.T.shape == (self.nA, self.nS, self.nS), \
                f"T shape {self.T.shape} != ({self.nA},{self.nS},{self.nS})"
            assert self.Z.shape == (self.nO, self.nS), \
                f"Z shape {self.Z.shape} != ({self.nO},{self.nS})"
            assert self.R.shape == (self.nA, self.nS), \
                f"R shape {self.R.shape} != ({self.nA},{self.nS})"
            assert len(self.b0) == self.nS, \
                f"b0 length {len(self.b0)} != {self.nS}"

            # Check that transition rows sum to 1
            T_sums = np.sum(self.T, axis=2)
            assert np.allclose(T_sums, 1.0, atol=1e-6), "Transition rows do not sum to 1"

            # Check that observation columns sum to 1
            Z_sums = np.sum(self.Z, axis=0)
            assert np.allclose(Z_sums, 1.0, atol=1e-6), "Observation columns do not sum to 1"

            # Check that initial belief sums to 1
            b0_sum = np.sum(self.b0)
            assert np.isclose(b0_sum, 1.0, atol=1e-6), "b0 does not sum to 1"

            return True

        except AssertionError as e:
            print(f"Validation error: {e}")
            return False
