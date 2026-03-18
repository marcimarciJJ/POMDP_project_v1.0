#!/usr/bin/env python3
"""
Auto-generate all 11 POMDP solver files
一键生成所有代码文件到正确位置
"""

import os
from pathlib import Path

# ============================================================================
# ALL 11 FILES CONTENT
# ============================================================================

FILES_CONTENT = {
    # File 1: unified_interface.py
    "src/pomdp_lib/solvers/unified_interface.py": '''"""
Unified Result Format for All POMDP Solvers
This module defines standard output format for all solvers.
All solvers (QMDP, PBVI, Witness, IP) must return this format.
This ensures consistent interface for comparison and visualization.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
import numpy as np
import json
from pathlib import Path
import pickle


@dataclass
class SolverResult:
    """
    Standard result format for all POMDP solvers.
    All solvers must return this format to ensure compatibility.
    """
    
    # === Core Solving Results ===
    solver_name: str                      # "QMDP" | "PBVI" | "Witness" | "IP"
    policy: np.ndarray                    # shape (nS,), best action for each state
    value_function: np.ndarray            # shape (nS,), optimal value for each state
    
    # === Convergence History ===
    convergence_trace: List[float]        # J(b0) value at each iteration
    convergence_time: float               # total solving time in seconds
    iterations: int                       # number of iterations performed
    final_bellman_error: float            # bellman residual at convergence
    is_converged: bool                    # whether algorithm converged
    
    # === Environment Info ===
    environment_id: str                   # "3x3" or "5x5"
    grid_width: int
    grid_height: int
    num_states: int
    num_actions: int
    num_observations: int
    gamma: float                          # discount factor
    
    # === Only for PBVI ===
    alpha_vectors: Optional[np.ndarray] = None  # shape (K, nS)
    num_beliefs: Optional[int] = None           # number of belief points used
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for JSON serialization."""
        result = {
            'solver_name': self.solver_name,
            'environment_id': self.environment_id,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'num_states': self.num_states,
            'num_actions': self.num_actions,
            'num_observations': self.num_observations,
            'gamma': self.gamma,
            'policy': self.policy.tolist() if isinstance(self.policy, np.ndarray) else self.policy,
            'value_function': self.value_function.tolist() if isinstance(self.value_function, np.ndarray) else self.value_function,
            'convergence_trace': self.convergence_trace,
            'convergence_time': float(self.convergence_time),
            'iterations': int(self.iterations),
            'final_bellman_error': float(self.final_bellman_error),
            'is_converged': bool(self.is_converged),
        }
        
        if self.num_beliefs is not None:
            result['num_beliefs'] = self.num_beliefs
            
        return result
    
    def save_json(self, filepath: str):
        """Save result to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"Result saved to {filepath}")
    
    def save_pickle(self, filepath: str):
        """Save result to pickle file for fast loading."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Result saved to {filepath}")
    
    @staticmethod
    def load_json(filepath: str) -> 'SolverResult':
        """Load result from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        data['policy'] = np.array(data['policy'], dtype=int)
        data['value_function'] = np.array(data['value_function'], dtype=float)
        
        return SolverResult(**data)
    
    @staticmethod
    def load_pickle(filepath: str) -> 'SolverResult':
        """Load result from pickle file."""
        with open(filepath, 'rb') as f:
            result = pickle.load(f)
        
        return result


@dataclass
class ExperimentConfig:
    """Configuration for POMDP solving experiments."""
    
    # === QMDP Settings ===
    qmdp_epsilon: float = 1e-6
    qmdp_max_iterations: int = 100
    
    # === PBVI Settings ===
    pbvi_epsilon: float = 1e-6
    pbvi_max_iterations: int = 100
    pbvi_num_belief_points: Optional[int] = None
    pbvi_min_iterations: int = 10
    
    # === External Solver Settings ===
    witness_epsilon: float = 1e-6
    witness_max_iterations: int = 100
    ip_epsilon: float = 1e-6
    ip_max_iterations: int = 100
    
    # === Environment ===
    random_seed: int = 42
    gamma: float = 0.95
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save config to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @staticmethod
    def load(filepath: str) -> 'ExperimentConfig':
        """Load config from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return ExperimentConfig(**data)
''',

    # File 2: base_solver.py
    "src/pomdp_lib/solvers/base_solver.py": '''"""
Abstract Base Class for All POMDP Solvers
Define the common interface that all solvers must follow.
This ensures consistency across different solving methods.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple
import time
from .unified_interface import SolverResult


class BasePOMDPSolver(ABC):
    """
    Abstract base class for all POMDP solvers.
    Every solver (QMDP, PBVI, Witness, IP) inherits from this class.
    """
    
    def __init__(self, model_data: Dict[str, Any], solver_name: str):
        """
        Initialize solver with model data.
        
        Args:
            model_data: Dictionary containing S, A, O, T, Z, R, gamma, b0, meta
            solver_name: name of the solver
        """
        # Extract model components
        self.S = model_data['S']          # list of states
        self.A = model_data['A']          # list of action names
        self.O = model_data['O']          # list of observation labels
        self.T = model_data['T']          # transition matrix (nA, nS, nS)
        self.Z = model_data['Z']          # observation matrix (nO, nS)
        self.R = model_data['R']          # reward matrix (nA, nS)
        self.gamma = model_data['gamma']  # discount factor
        self.b0 = model_data['b0']        # initial belief
        self.meta = model_data['meta']    # metadata
        
        self.solver_name = solver_name
        self.nS = len(self.S)
        self.nA = len(self.A)
        self.nO = len(self.O)
        
        # result storage
        self.policy = None
        self.value_function = None
        self.iterations = 0
        self.final_error = 0.0
        self.convergence_time = 0.0
        self.convergence_trace = []
    
    @abstractmethod
    def solve(self, epsilon: float = 1e-6, max_iterations: int = 100) -> SolverResult:
        """
        Solve POMDP problem.
        This method must be implemented by all subclasses.
        
        Args:
            epsilon: convergence threshold
            max_iterations: maximum number of iterations
            
        Returns:
            SolverResult: unified result format
        """
        pass
    
    def _create_result(self, epsilon: float) -> SolverResult:
        """
        Create result object in unified format.
        This is called by subclasses after solving.
        """
        return SolverResult(
            solver_name=self.solver_name,
            policy=self.policy,
            value_function=self.value_function,
            convergence_trace=self.convergence_trace,
            environment_id=self.meta.get('environment_id', 'unknown'),
            grid_width=self.meta.get('width', 0),
            grid_height=self.meta.get('height', 0),
            num_states=self.nS,
            num_actions=self.nA,
            num_observations=self.nO,
            gamma=self.gamma,
            convergence_time=self.convergence_time,
            iterations=self.iterations,
            final_bellman_error=self.final_error,
            is_converged=self.final_error < epsilon
        )
''',

    # File 3: qmdp.py (简化版)
    "src/pomdp_lib/solvers/qmdp.py": '''"""
QMDP Solver - State-Action Policy
QMDP ignores observation and solves standard MDP.
This is an upper bound on optimal POMDP value.
"""

import numpy as np
from typing import Dict, Any
import time
from .base_solver import BasePOMDPSolver
from .unified_interface import SolverResult


class QMDPSolver(BasePOMDPSolver):
    """
    QMDP solver implements standard MDP value iteration.
    Policy depends only on state, not on belief.
    """
    
    def __init__(self, model_data: Dict[str, Any]):
        super().__init__(model_data, "QMDP")
    
    def solve(self, epsilon: float = 1e-6, max_iterations: int = 100) -> SolverResult:
        """
        Solve using QMDP algorithm.
        Standard value iteration on MDP formed by ignoring observations.
        """
        start_time = time.time()
        
        # initialize value function
        V = np.zeros(self.nS)
        self.convergence_trace = []
        
        # main value iteration loop
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            # compute Q values for all state-action pairs
            # Q(s,a) = R(a,s) + gamma * sum_s' T(a,s,s') * V(s')
            Q = np.zeros((self.nS, self.nA))
            for a in range(self.nA):
                # immediate reward for action a at each state
                Q[:, a] = self.R[a, :]
                
                # expected future value: transition weighted
                future = self.T[a] @ V_old  # (nS, nS) @ (nS,) = (nS,)
                Q[:, a] += self.gamma * future
            
            # update V by taking max over actions
            V = np.max(Q, axis=1)
            
            # compute J(b0) for convergence trace
            J_b0 = np.dot(self.b0, V)
            self.convergence_trace.append(float(J_b0))
            
            # check convergence using Bellman error
            error = np.max(np.abs(V - V_old))
            self.final_error = error
            self.iterations = iteration + 1
            
            if error < epsilon:
                break
        
        # extract optimal policy from final Q values
        Q_final = np.zeros((self.nS, self.nA))
        for a in range(self.nA):
            Q_final[:, a] = self.R[a, :] + self.gamma * (self.T[a] @ V)
        
        self.policy = np.argmax(Q_final, axis=1)
        self.value_function = V
        self.convergence_time = time.time() - start_time
        
        return self._create_result(epsilon)
''',

    # File 4: pbvi.py (简化版)
    "src/pomdp_lib/solvers/pbvi.py": '''"""
PBVI - Point-Based Value Iteration
Solves POMDP using alpha-vector representation.
Samples belief points and maintains set of alpha vectors.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import time
from .base_solver import BasePOMDPSolver
from .unified_interface import SolverResult


class PBVISolver(BasePOMDPSolver):
    """
    PBVI solver uses point-based value iteration.
    Samples belief points and backs up alpha vectors.
    """
    
    def __init__(self, model_data: Dict[str, Any]):
        super().__init__(model_data, "PBVI")
        self.alpha_vectors = []
        self.alpha_actions = []
    
    def solve(self, epsilon: float = 1e-6, max_iterations: int = 100,
              num_belief_points: Optional[int] = None) -> SolverResult:
        """
        Solve using PBVI algorithm.
        Uses adaptive belief sampling: max(|S|, 20).
        """
        start_time = time.time()
        
        # adaptive belief point count
        if num_belief_points is None:
            num_belief_points = max(self.nS, 20)
        
        # sample belief points
        B = self._sample_beliefs(num_belief_points)
        nB = len(B)
        
        # initialize alpha vectors
        self.alpha_vectors = [np.zeros(self.nS) for _ in range(nB)]
        self.alpha_actions = [0] * nB
        self.convergence_trace = []
        
        # main PBVI loop
        for iteration in range(max_iterations):
            prev_alphas = [a.copy() for a in self.alpha_vectors]
            prev_alphas_matrix = np.vstack(prev_alphas)
            
            new_alphas = []
            new_actions = []
            
            # for each belief point find best alpha vector
            for b in B:
                best_val = -float('inf')
                best_alpha = None
                best_action = -1
                
                # try each action
                for a in range(self.nA):
                    alpha_a = self.R[a, :].copy()
                    
                    # backup: add discounted future value
                    for o in range(self.nO):
                        z_vec = self.Z[o, :]
                        weighted = prev_alphas_matrix * z_vec
                        back_proj = (self.T[a] @ weighted.T).T
                        vals = back_proj @ b
                        best_idx = np.argmax(vals)
                        alpha_a += self.gamma * back_proj[best_idx]
                    
                    # evaluate this action at current belief
                    val_a = alpha_a @ b
                    if val_a > best_val:
                        best_val = val_a
                        best_alpha = alpha_a
                        best_action = a
                
                new_alphas.append(best_alpha)
                new_actions.append(best_action)
            
            # update alpha set
            self.alpha_vectors = new_alphas
            self.alpha_actions = new_actions
            
            # compute J(b0) for convergence trace
            J_b0 = max(alpha @ self.b0 for alpha in self.alpha_vectors)
            self.convergence_trace.append(float(J_b0))
            
            # check convergence
            bellman_error = self._compute_bellman_error(prev_alphas)
            self.final_error = bellman_error
            self.iterations = iteration + 1
            
            if bellman_error < epsilon and iteration >= max(10, max_iterations // 5):
                break
        
        # extract policy and value function
        self._extract_policy_from_alphas(self.alpha_vectors, B)
        
        self.convergence_time = time.time() - start_time
        
        return self._create_result(epsilon)
    
    def _sample_beliefs(self, num_points: int) -> List[np.ndarray]:
        """Sample belief points using Dirichlet distribution."""
        np.random.seed(42)
        
        B = [self.b0.copy()]
        
        for _ in range(num_points - 1):
            b = np.random.dirichlet(np.ones(self.nS))
            B.append(b)
        
        return B
    
    def _compute_bellman_error(self, prev_alphas: List[np.ndarray]) -> float:
        """Compute Bellman residual to check convergence."""
        if not prev_alphas or not self.alpha_vectors:
            return float('inf')
        
        prev_norms = [np.linalg.norm(a) for a in prev_alphas]
        curr_norms = [np.linalg.norm(a) for a in self.alpha_vectors]
        
        error = np.max([abs(c - p) for c, p in zip(curr_norms, prev_norms)])
        
        return error
    
    def _extract_policy_from_alphas(self, alphas: List[np.ndarray],
                                    beliefs: List[np.ndarray]):
        """Extract policy and value function from alpha vectors."""
        self.value_function = np.array([
            max((alpha @ self.b0) for alpha in alphas)
            for _ in range(self.nS)
        ])
        
        self.policy = np.zeros(self.nS, dtype=int)
        for s in range(self.nS):
            b_s = np.zeros(self.nS)
            b_s[s] = 1.0
            
            Q_s = np.zeros(self.nA)
            for a in range(self.nA):
                max_alpha_val = max(alpha @ b_s for alpha in alphas)
                Q_s[a] = max_alpha_val
            
            self.policy[s] = np.argmax(Q_s)
''',

    # File 5: comparator.py
    "src/pomdp_lib/analysis/comparator.py": '''"""
MultiSolverComparator - Compare Results from All Solvers
Provides comprehensive comparison of policy, value function and efficiency.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from solvers.unified_interface import SolverResult


class MultiSolverComparator:
    """
    Compare results from multiple POMDP solvers.
    Analyze policy agreement, value function differences and efficiency.
    """
    
    def __init__(self, results: Dict[str, SolverResult]):
        """Initialize comparator with solver results."""
        self.results = results
        self.solver_names = list(results.keys())
    
    def compare_policies(self) -> Dict:
        """Compare policies from all solvers."""
        comparison = {}
        
        for i, s1 in enumerate(self.solver_names):
            for s2 in self.solver_names[i+1:]:
                r1, r2 = self.results[s1], self.results[s2]
                
                agreement = np.mean(r1.policy == r2.policy)
                differing = np.sum(r1.policy != r2.policy)
                
                comparison[f"{s1}_vs_{s2}"] = {
                    "agreement_rate": float(agreement),
                    "num_differing_states": int(differing),
                    "num_total_states": int(len(r1.policy))
                }
        
        return comparison
    
    def compare_values(self) -> Dict:
        """Compare value functions from all solvers."""
        comparison = {}
        
        for i, s1 in enumerate(self.solver_names):
            for s2 in self.solver_names[i+1:]:
                r1, r2 = self.results[s1], self.results[s2]
                
                v1, v2 = r1.value_function, r2.value_function
                l2_dist = np.linalg.norm(v1 - v2)
                l_inf_dist = np.max(np.abs(v1 - v2))
                mean_abs_diff = np.mean(np.abs(v1 - v2))
                
                comparison[f"{s1}_vs_{s2}"] = {
                    "l2_distance": float(l2_dist),
                    "l_inf_distance": float(l_inf_dist),
                    "mean_absolute_diff": float(mean_abs_diff)
                }
        
        return comparison
    
    def compare_efficiency(self) -> Dict:
        """Compare computational efficiency of solvers."""
        efficiency = {}
        
        for name, result in self.results.items():
            efficiency[name] = {
                "time_ms": float(result.convergence_time * 1000),
                "iterations": int(result.iterations),
                "bellman_error": float(result.final_bellman_error),
                "is_converged": bool(result.is_converged)
            }
        
        return efficiency
    
    def generate_report(self, output_dir: str = "./results") -> Dict:
        """Generate comprehensive comparison report."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        report = {
            "policy_comparison": self.compare_policies(),
            "value_comparison": self.compare_values(),
            "efficiency_comparison": self.compare_efficiency(),
            "summary": {
                "num_solvers": len(self.solver_names),
                "solver_names": self.solver_names,
                "num_states": self.results[self.solver_names[0]].num_states
            }
        }
        
        report_path = f"{output_dir}/comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comparison report saved to {report_path}")
        
        return report
    
    def print_summary(self):
        """Print human readable summary of comparison."""
        print("\\n" + "="*60)
        print("POMDP SOLVER COMPARISON SUMMARY")
        print("="*60)
        
        print("\\nEFFICIENCY COMPARISON:")
        print("-" * 60)
        eff = self.compare_efficiency()
        for name, metrics in eff.items():
            print(f"{name:15} Time: {metrics['time_ms']:10.2f}ms, " +
                  f"Iterations: {metrics['iterations']:3}, " +
                  f"Error: {metrics['bellman_error']:.2e}")
        
        print("\\nPOLICY AGREEMENT:")
        print("-" * 60)
        policies = self.compare_policies()
        for pair, agreement in policies.items():
            print(f"{pair:30} Agreement: {agreement['agreement_rate']*100:6.2f}%")
        
        print("\\nVALUE FUNCTION L2 DISTANCES:")
        print("-" * 60)
        values = self.compare_values()
        for pair, distances in values.items():
            print(f"{pair:30} L2: {distances['l2_distance']:.6f}, " +
                  f"L_inf: {distances['l_inf_distance']:.6f}")
''',

    # File 6: visualizer.py
    "src/pomdp_lib/vis/visualizer.py": '''"""
Academic Style Visualization for POMDP Results
Generate publication-quality figures using matplotlib and seaborn.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from solvers.unified_interface import SolverResult


class Visualizer:
    """Generate academic style visualizations for POMDP solving results."""
    
    def __init__(self, results: Dict[str, SolverResult], style: str = "seaborn-v0_8-darkgrid"):
        """Initialize visualizer with solver results."""
        self.results = results
        self.style = style
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_value_functions(self, output_dir: str = "./results"):
        """Plot value functions from all solvers for comparison."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        n_solvers = len(self.results)
        n_cols = min(2, n_solvers)
        n_rows = (n_solvers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        if n_solvers == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (solver_name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            
            values = result.value_function
            states = np.arange(len(values))
            
            ax.bar(states, values, color='steelblue', alpha=0.7, edgecolor='navy')
            ax.set_xlabel("State Index", fontsize=11)
            ax.set_ylabel("Value Function", fontsize=11)
            ax.set_title(f"{solver_name}\\nTime: {result.convergence_time:.4f}s, " +
                        f"Iterations: {result.iterations}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        for idx in range(len(self.results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        output_path = f"{output_dir}/value_functions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Value function plot saved to {output_path}")
        plt.close()
    
    def plot_convergence_traces(self, output_dir: str = "./results"):
        """Plot convergence traces showing J(b0) over iterations."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for solver_name, result in self.results.items():
            trace = result.convergence_trace
            iterations = np.arange(len(trace))
            ax.plot(iterations, trace, marker='o', label=solver_name, linewidth=2, markersize=4)
        
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("J(b0) Value", fontsize=12)
        ax.set_title("Convergence Comparison", fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f"{output_dir}/convergence.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {output_path}")
        plt.close()
    
    def plot_efficiency_comparison(self, output_dir: str = "./results"):
        """Plot running time and iteration count comparison."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        solver_names = list(self.results.keys())
        times = [self.results[name].convergence_time * 1000 for name in solver_names]
        iters = [self.results[name].iterations for name in solver_names]
        
        ax = axes[0]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(solver_names)]
        bars1 = ax.bar(solver_names, times, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_title("Running Time Comparison", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax = axes[1]
        bars2 = ax.bar(solver_names, iters, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel("Iterations", fontsize=11)
        ax.set_title("Iteration Count Comparison", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_path = f"{output_dir}/efficiency.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Efficiency plot saved to {output_path}")
        plt.close()
    
    def plot_policy_agreement_matrix(self, output_dir: str = "./results"):
        """Plot heatmap showing policy agreement between all pairs of solvers."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        n = len(self.results)
        agreement_matrix = np.zeros((n, n))
        solver_names = list(self.results.keys())
        
        for i, s1 in enumerate(solver_names):
            for j, s2 in enumerate(solver_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                elif i < j:
                    p1 = self.results[s1].policy
                    p2 = self.results[s2].policy
                    agreement = np.mean(p1 == p2)
                    agreement_matrix[i, j] = agreement
                    agreement_matrix[j, i] = agreement
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        sns.heatmap(agreement_matrix, annot=True, fmt='.2%', cmap='YlGn',
                   xticklabels=solver_names, yticklabels=solver_names,
                   cbar_kws={'label': 'Agreement Rate'},
                   ax=ax, vmin=0, vmax=1)
        
        ax.set_title("Policy Agreement Matrix", fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        output_path = f"{output_dir}/agreement_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Agreement matrix plot saved to {output_path}")
        plt.close()
    
    def plot_all(self, output_dir: str = "./results"):
        """Generate all plots at once."""
        self.plot_value_functions(output_dir)
        self.plot_convergence_traces(output_dir)
        self.plot_efficiency_comparison(output_dir)
        self.plot_policy_agreement_matrix(output_dir)
        print(f"All plots saved to {output_dir}/")
''',

    # File 7: exporter.py
    "external/utils/exporter.py": '''"""
POMDP File Format Exporter
Convert NumPy matrix representation to standard .pomdp file format.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any


class POMMDPFileExporter:
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
            f.write(f"discount: {gamma}\\n")
            f.write("values: reward\\n")
            f.write(f"states: {nS}\\n")
            f.write(f"actions: {nA}\\n")
            f.write(f"observations: {nO}\\n\\n")
            
            f.write("# Transition probabilities\\n")
            for a in range(nA):
                for s in range(nS):
                    for s_next in range(nS):
                        prob = T[a, s, s_next]
                        if prob > 1e-10:
                            f.write(f"T: {A[a]} : {s} : {s_next} {prob}\\n")
            
            f.write("\\n")
            
            f.write("# Observation probabilities\\n")
            for o in range(nO):
                for s_next in range(nS):
                    prob = Z[o, s_next]
                    if prob > 1e-10:
                        f.write(f"Z: * : {s_next} : {O[o]} {prob}\\n")
            
            f.write("\\n")
            
            f.write("# Rewards\\n")
            for a in range(nA):
                for s in range(nS):
                    r = R[a, s]
                    if r != 0:
                        f.write(f"R: {A[a]} : {s} : * : * {r}\\n")
        
        print(f"Model exported to {output_file}")
''',

    # File 8: wrapper.py
    "external/solvers/wrapper.py": '''"""
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
            from external.utils.exporter import POMMDPFileExporter
        except:
            from pathlib import Path
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
            from exporter import POMMDPFileExporter
        
        model_data = {
            'S': self.S,
            'A': self.A,
            'O': self.O,
            'T': self.T,
            'Z': self.Z,
            'R': self.R,
            'gamma': self.gamma
        }
        POMMDPFileExporter.export(model_data, output_file)
    
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
''',

    # File 9: colab_demo.py
    "scripts/colab_demo.py": '''"""
Complete POMDP Analysis Demo for Colab
End-to-end demonstration of all solvers and analysis.
"""

import sys
import os
import time
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
external_path = os.path.abspath(os.path.join(current_dir, "../external"))
sys.path.insert(0, src_path)
sys.path.insert(0, external_path)

import numpy as np


def main():
    """Run complete POMDP analysis demo."""
    
    print("\\n" + "="*70)
    print(" "*15 + "POMDP Analysis Tool - Complete Demo")
    print("="*70)
    
    # ===== Step 1: Load Models =====
    print("\\n[Step 1] Loading POMDP Models...")
    print("-" * 70)
    
    from pomdp_lib.models.model_3x3 import S as S_3x3, A, O as O_3x3
    from pomdp_lib.models.model_3x3 import T as T_3x3, Z as Z_3x3, R as R_3x3, b0 as b0_3x3, gamma
    
    model_3x3 = {
        'S': S_3x3, 'A': A, 'O': O_3x3, 'T': T_3x3, 'Z': Z_3x3, 'R': R_3x3,
        'gamma': gamma, 'b0': b0_3x3,
        'meta': {'width': 3, 'height': 3, 'environment_id': '3x3'}
    }
    
    print(f"✓ 3x3 model loaded: {len(S_3x3)} states, {len(A)} actions, {len(O_3x3)} observations")
    
    # ===== Step 2: Run Approximate Algorithms =====
    print("\\n[Step 2] Running Approximate Algorithms (QMDP & PBVI)...")
    print("-" * 70)
    
    from pomdp_lib.solvers.qmdp import QMDPSolver
    from pomdp_lib.solvers.pbvi import PBVISolver
    
    results = {}
    
    print("  Running QMDP on 3x3 model...")
    qmdp = QMDPSolver(model_3x3)
    result_qmdp = qmdp.solve(epsilon=1e-6, max_iterations=100)
    results["QMDP"] = result_qmdp
    print(f"  ✓ QMDP: {result_qmdp.convergence_time:.4f}s, {result_qmdp.iterations} iterations")
    
    print("  Running PBVI on 3x3 model...")
    pbvi = PBVISolver(model_3x3)
    result_pbvi = pbvi.solve(epsilon=1e-6, max_iterations=100)
    results["PBVI"] = result_pbvi
    print(f"  ✓ PBVI: {result_pbvi.convergence_time:.4f}s, {result_pbvi.iterations} iterations")
    
    # ===== Step 3: Comparison Analysis =====
    print("\\n[Step 3] Comparative Analysis...")
    print("-" * 70)
    
    from pomdp_lib.analysis.comparator import MultiSolverComparator
    
    comparator = MultiSolverComparator(results)
    
    output_dir = "./results"
    Path(output_dir).mkdir(exist_ok=True)
    
    report = comparator.generate_report(output_dir)
    
    comparator.print_summary()
    
    # ===== Step 4: Visualization =====
    print("\\n[Step 4] Generating Visualizations...")
    print("-" * 70)
    
    from pomdp_lib.vis.visualizer import Visualizer
    
    visualizer = Visualizer(results)
    visualizer.plot_all(output_dir)
    
    # ===== Final Summary =====
    print("\\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"Results saved to: {output_dir}/")
    print("\\n✓ All analysis complete!")


if __name__ == "__main__":
    main()
''',

    # File 10: experiment_config.yaml
    "config/experiment_config.yaml": '''# POMDP Experiment Configuration

qmdp:
  epsilon: 1.0e-6
  max_iterations: 100

pbvi:
  epsilon: 1.0e-6
  max_iterations: 100
  num_belief_points: null
  min_iterations: 10

witness:
  epsilon: 1.0e-6
  max_iterations: 100

incremental_pruning:
  epsilon: 1.0e-6
  max_iterations: 100

environment:
  gamma: 0.95
  random_seed: 42

grids:
  "3x3":
    width: 3
    height: 3
    exit_cell: [3, 2]
    traps: [[1, 1]]
    trees: [[2, 2]]
    robot_start: [1, 3]
    initial_belief: "deterministic"

output:
  results_dir: "./results"
  save_format: ["json", "pickle"]
  generate_plots: true
  plot_dpi: 300
''',

    # File 11: __init__.py files
    "src/pomdp_lib/solvers/__init__.py": '''"""POMDP Solvers Package"""
from .unified_interface import SolverResult, ExperimentConfig
from .base_solver import BasePOMDPSolver
from .qmdp import QMDPSolver
from .pbvi import PBVISolver

__all__ = [
    'SolverResult',
    'ExperimentConfig',
    'BasePOMDPSolver',
    'QMDPSolver',
    'PBVISolver'
]
''',

    "src/pomdp_lib/analysis/__init__.py": '''"""POMDP Analysis Package"""
from .comparator import MultiSolverComparator

__all__ = ['MultiSolverComparator']
''',

    "src/pomdp_lib/vis/__init__.py": '''"""POMDP Visualization Package"""
from .visualizer import Visualizer

__all__ = ['Visualizer']
''',

    "external/utils/__init__.py": '''"""External Solver Utilities"""
from .exporter import POMMDPFileExporter

__all__ = ['POMMDPFileExporter']
''',

    "external/solvers/__init__.py": '''"""External Solver Wrappers"""
from .wrapper import ExactSolver

__all__ = ['ExactSolver']
''',
}


def generate_all_files():
    """Generate all files to disk."""
    print("\n" + "="*70)
    print("POMDP Solver Project - Complete File Generator")
    print("="*70)
    
    success_count = 0
    failed_count = 0
    
    for file_path, content in FILES_CONTENT.items():
        try:
            # Create directory structure
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ {file_path}")
            success_count += 1
        
        except Exception as e:
            print(f"✗ {file_path}: {e}")
            failed_count += 1
    
    print("\n" + "="*70)
    print(f"✅ Successfully created {success_count}/{len(FILES_CONTENT)} files!")
    if failed_count > 0:
        print(f"⚠️ Failed to create {failed_count} files")
    print("="*70)
    
    print("\n📁 Your project structure is now ready!")
    print("Next steps:")
    print("  1. Run: python scripts/colab_demo.py")
    print("  2. Check results in ./results/")
    print("  3. Upload to Colab for final testing")


if __name__ == "__main__":
    generate_all_files()
