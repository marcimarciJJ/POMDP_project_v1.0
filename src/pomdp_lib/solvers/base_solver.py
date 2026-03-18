"""
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
