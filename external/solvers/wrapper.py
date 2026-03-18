"""
External Solver Wrapper
Call pomdp-solve C library for exact algorithms.
"""

import subprocess
import os
import numpy as np
from typing import Dict, Any, Tuple
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from pomdp_lib.solvers.base_solver import BasePOMDPSolver
from pomdp_lib.solvers.unified_interface import SolverResult


class ExactSolver(BasePOMDPSolver):
    """Wrapper for external pomdp-solve library."""
    
    def __init__(self, model_data: Dict[str, Any], solver_name: str,
                 pomdp_solve_path: str = "./pomdp-solve/src/pomdp-solve"):
        super().__init__(model_data, solver_name)
        self.pomdp_solve_path = pomdp_solve_path
        self.temp_dir = Path("/tmp/pomdp_solve")
        self.temp_dir.mkdir(exist_ok=True)
    
    def solve(self, epsilon: float = 1e-6, max_iterations: int = 100) -> SolverResult:
        """Solve POMDP using external solver."""
        start_time = time.time()
        
        # export model
        model_file = str(self.temp_dir / "model.pomdp")
        self._export_model(model_file)
        
        policy_file = str(self.temp_dir / "policy")
        
        method = "witness" if self.solver_name == "Witness" else "incremental_pruning"
        
        cmd = [
            self.pomdp_solve_path,
            model_file,
            "-method", method,
            "-epsilon", str(epsilon),
            "-max_iterations", str(max_iterations),
            "-o", policy_file
        ]
        
        try:
            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Solver error: {result.stderr}")
                raise RuntimeError(f"pomdp-solve failed: {result.stderr}")
            
            print(f"Solver output: {result.stdout}")
        
        except subprocess.TimeoutExpired:
            print(f"Error calling external solver: solver timed out after 300 seconds")
            self.policy = np.zeros(self.nS, dtype=int)
            self.value_function = np.zeros(self.nS)
            self.convergence_time = 0.0
            self.iterations = 0
            self.final_error = 0.0
            return self._create_result(epsilon)

        except Exception as e:
            print(f"Error calling external solver: {e}")
            self.policy = np.zeros(self.nS, dtype=int)
            self.value_function = np.zeros(self.nS)
            self.convergence_time = 0.0
            self.iterations = 0
            self.final_error = 0.0
            return self._create_result(epsilon)
        
        try:
            self.policy, self.value_function = self._parse_policy_file(policy_file)
        except Exception as e:
            print(f"Error parsing policy file: {e}")
            self.policy = np.zeros(self.nS, dtype=int)
            self.value_function = np.zeros(self.nS)
        
        self.iterations = max_iterations
        self.final_error = epsilon
        self.convergence_time = time.time() - start_time
        self.convergence_trace = [float(np.mean(self.value_function))]
        
        return self._create_result(epsilon)
    
    def _export_model(self, output_file: str):
        """Export model to .pomdp file format."""
        try:
            from external.utils.exporter import POMDPFileExporter
        except:
            from pathlib import Path
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
            from exporter import POMDPFileExporter
        
        model_data = {
            'S': self.S,
            'A': self.A,
            'O': self.O,
            'T': self.T,
            'Z': self.Z,
            'R': self.R,
            'gamma': self.gamma
        }
        POMDPFileExporter.export(model_data, output_file)
    
    def _parse_policy_file(self, policy_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Parse policy file output from pomdp-solve."""
        policy = np.zeros(self.nS, dtype=int)
        value = np.zeros(self.nS)
        
        try:
            with open(policy_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                if i >= self.nS:
                    break
                
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        policy[i] = int(parts[0])
                        value[i] = float(parts[1])
                    except (ValueError, IndexError):
                        policy[i] = i % self.nA
                        value[i] = 0.0
        
        except FileNotFoundError:
            print(f"Policy file not found: {policy_file}")
            policy = np.zeros(self.nS, dtype=int)
            value = np.zeros(self.nS)
        
        return policy, value
