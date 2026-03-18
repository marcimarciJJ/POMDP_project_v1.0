"""
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
