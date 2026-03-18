"""
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
        self.value_function = np.zeros(self.nS)
        for s in range(self.nS):
            b_s = np.zeros(self.nS)
            b_s[s] = 1.0
            self.value_function[s] = max(alpha @ b_s for alpha in alphas)
        
        self.policy = np.zeros(self.nS, dtype=int)
        for s in range(self.nS):
            b_s = np.zeros(self.nS)
            b_s[s] = 1.0
            
            Q_s = np.zeros(self.nA)
            for a in range(self.nA):
                max_alpha_val = max(alpha @ b_s for alpha in alphas)
                Q_s[a] = max_alpha_val
            
            self.policy[s] = np.argmax(Q_s)
