"""
POMDP File Format Exporter
Convert NumPy matrix representation to standard .pomdp file format.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any


class POMDPFileExporter:
    """Export POMDP model to standard .pomdp file format."""
    
    @staticmethod
    def export(model_data: Dict[str, Any], output_file: str):
        """Export model to .pomdp file."""
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
            f.write(f"states: {nS}\n")
            f.write(f"actions: {nA}\n")
            f.write(f"observations: {nO}\n\n")
            
            f.write("# Transition probabilities\n")
            for a in range(nA):
                for s in range(nS):
                    for s_next in range(nS):
                        prob = T[a, s, s_next]
                        if prob > 1e-10:
                            f.write(f"T: {A[a]} : {s} : {s_next} {prob}\n")
            
            f.write("\n")
            
            f.write("# Observation probabilities\n")
            for o in range(nO):
                for s_next in range(nS):
                    prob = Z[o, s_next]
                    if prob > 1e-10:
                        f.write(f"Z: * : {s_next} : {O[o]} {prob}\n")
            
            f.write("\n")
            
            f.write("# Rewards\n")
            for a in range(nA):
                for s in range(nS):
                    r = R[a, s]
                    if r != 0:
                        f.write(f"R: {A[a]} : {s} : * : * {r}\n")
        
        print(f"Model exported to {output_file}")
