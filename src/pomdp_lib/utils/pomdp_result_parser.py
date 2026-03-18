"""
POMDP Result Parser - Parse .policy and .alpha files from pomdp-solve.
Used after running the external exact solver to read back solutions.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class POMDPResultParser:
    """
    Parse result files output by the pomdp-solve C library.
    Supports .alpha (alpha vectors) and .pg (policy graph) file formats.
    """

    @staticmethod
    def parse_alpha_file(alpha_file: str, nS: int) -> Tuple[np.ndarray, List[int]]:
        """
        Parse a .alpha file containing alpha vectors.

        Each alpha vector entry has:
        - A line with the action index
        - A line with nS space-separated floats

        Args:
            alpha_file: path to the .alpha file
            nS: number of states

        Returns:
            Tuple of (alpha_matrix, actions) where
            alpha_matrix has shape (K, nS) and actions is a list of K integers
        """
        path = Path(alpha_file)
        if not path.exists():
            raise FileNotFoundError(f"Alpha file not found: {alpha_file}")

        alphas = []
        actions = []

        with open(path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0
        while i < len(lines):
            try:
                # Action index on its own line
                action = int(lines[i])
                i += 1
                if i >= len(lines):
                    break
                # Alpha vector values on next line
                values = list(map(float, lines[i].split()))
                if len(values) == nS:
                    alphas.append(values)
                    actions.append(action)
                i += 1
            except (ValueError, IndexError):
                i += 1
                continue

        if not alphas:
            return np.zeros((1, nS)), [0]

        return np.array(alphas), actions

    @staticmethod
    def parse_pg_file(pg_file: str) -> Dict[int, Dict[str, Any]]:
        """
        Parse a .pg (policy graph) file from pomdp-solve.

        Args:
            pg_file: path to the .pg file

        Returns:
            Dictionary mapping node id to {action, transitions}
        """
        path = Path(pg_file)
        if not path.exists():
            raise FileNotFoundError(f"PG file not found: {pg_file}")

        graph = {}

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        node_id = int(parts[0])
                        action = int(parts[1])
                        transitions = [int(x) for x in parts[2:]]
                        graph[node_id] = {
                            'action': action,
                            'transitions': transitions
                        }
                    except (ValueError, IndexError):
                        continue

        return graph

    @staticmethod
    def extract_policy_from_alphas(alpha_matrix: np.ndarray,
                                   actions: List[int],
                                   nS: int) -> np.ndarray:
        """
        Extract a state-based policy from alpha vectors.

        For each state s, pick the alpha vector that gives highest value
        when b is the unit vector at s.

        Args:
            alpha_matrix: shape (K, nS)
            actions: action for each alpha vector
            nS: number of states

        Returns:
            policy array of shape (nS,)
        """
        policy = np.zeros(nS, dtype=int)
        for s in range(nS):
            b_s = np.zeros(nS)
            b_s[s] = 1.0
            values = alpha_matrix @ b_s
            best_idx = int(np.argmax(values))
            policy[s] = actions[best_idx]
        return policy

    @staticmethod
    def extract_value_function_from_alphas(alpha_matrix: np.ndarray,
                                           b0: np.ndarray) -> np.ndarray:
        """
        Extract value function from alpha vectors using initial belief.

        Args:
            alpha_matrix: shape (K, nS)
            b0: initial belief vector

        Returns:
            value function array of shape (nS,)
        """
        nS = alpha_matrix.shape[1]
        value = np.zeros(nS)
        for s in range(nS):
            b_s = np.zeros(nS)
            b_s[s] = 1.0
            value[s] = float(np.max(alpha_matrix @ b_s))
        return value
