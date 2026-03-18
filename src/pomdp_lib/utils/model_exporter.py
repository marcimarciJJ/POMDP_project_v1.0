"""
Model Exporter - Export POMDP models to .pomdp file format.
The .pomdp format is used by the external pomdp-solve C library.
"""

from pathlib import Path
from typing import Any, Dict
import numpy as np


class ModelExporter:
    """
    Export POMDP models to standard .pomdp file format.
    Used to interface with external exact solvers like pomdp-solve.
    """

    @staticmethod
    def export(model_data: Dict[str, Any], output_file: str) -> None:
        """
        Export a POMDP model dictionary to a .pomdp file.

        Args:
            model_data: dictionary with keys S, A, O, T, Z, R, gamma
            output_file: path to write the .pomdp file
        """
        S = model_data['S']
        A = model_data['A']
        O = model_data['O']
        T = model_data['T']
        Z = model_data['Z']
        R = model_data['R']
        gamma = model_data['gamma']

        nS = len(S)
        nA = len(A)
        nO = len(O)

        # Create parent directories if they don't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            # Header section
            f.write(f"discount: {gamma}\n")
            f.write("values: reward\n")
            f.write(f"states: {nS}\n")
            f.write(f"actions: {nA}\n")
            f.write(f"observations: {nO}\n\n")

            # Transition probabilities
            f.write("# Transition probabilities T(a, s, s')\n")
            for a in range(nA):
                for s in range(nS):
                    for s_next in range(nS):
                        prob = float(T[a, s, s_next])
                        if prob > 1e-10:
                            f.write(f"T: {a} : {s} : {s_next} {prob:.8f}\n")

            f.write("\n")

            # Observation probabilities
            f.write("# Observation probabilities Z(o, s')\n")
            for o in range(nO):
                for s_next in range(nS):
                    prob = float(Z[o, s_next])
                    if prob > 1e-10:
                        f.write(f"O: * : {s_next} : {o} {prob:.8f}\n")

            f.write("\n")

            # Rewards
            f.write("# Reward function R(a, s)\n")
            for a in range(nA):
                for s in range(nS):
                    r = float(R[a, s])
                    if abs(r) > 1e-10:
                        f.write(f"R: {a} : {s} : * : * {r:.6f}\n")

        print(f"Model exported to {output_file}")

    @staticmethod
    def export_with_labels(model_data: Dict[str, Any], output_file: str) -> None:
        """
        Export model with human-readable action/observation labels.

        Args:
            model_data: dictionary with keys S, A, O, T, Z, R, gamma
            output_file: path to write the .pomdp file
        """
        S = model_data['S']
        A = model_data['A']
        O = model_data['O']
        T = model_data['T']
        Z = model_data['Z']
        R = model_data['R']
        gamma = model_data['gamma']

        nS = len(S)
        nA = len(A)
        nO = len(O)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write(f"discount: {gamma}\n")
            f.write("values: reward\n")
            f.write(f"states: {' '.join(str(s) for s in S)}\n")
            f.write(f"actions: {' '.join(A)}\n")
            f.write(f"observations: {' '.join(O)}\n\n")

            for a_idx, a_name in enumerate(A):
                for s in range(nS):
                    for s_next in range(nS):
                        prob = float(T[a_idx, s, s_next])
                        if prob > 1e-10:
                            f.write(f"T: {a_name} : {s} : {s_next} {prob:.8f}\n")

            f.write("\n")

            for o_idx, o_name in enumerate(O):
                for s_next in range(nS):
                    prob = float(Z[o_idx, s_next])
                    if prob > 1e-10:
                        f.write(f"O: * : {s_next} : {o_name} {prob:.8f}\n")

            f.write("\n")

            for a_idx, a_name in enumerate(A):
                for s in range(nS):
                    r = float(R[a_idx, s])
                    if abs(r) > 1e-10:
                        f.write(f"R: {a_name} : {s} : * : * {r:.6f}\n")

        print(f"Model exported with labels to {output_file}")
