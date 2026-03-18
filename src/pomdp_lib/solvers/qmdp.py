"""
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
