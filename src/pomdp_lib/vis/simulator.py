"""
Belief State Simulator for POMDP.
Simulate agent interactions in a POMDP model, tracking belief state over time.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helpers import compute_belief_update, sample_from_distribution


class BeliefSimulator:
    """
    Simulate agent execution of a POMDP policy with belief tracking.
    Tracks belief state, actions, observations and cumulative reward.
    """

    def __init__(self, model: Dict[str, Any]):
        """
        Initialize simulator with POMDP model.

        Args:
            model: POMDP model dictionary with S, A, O, T, Z, R, gamma, b0
        """
        self.S = model['S']
        self.A = model['A']
        self.O = model['O']
        self.T = model['T']
        self.Z = model['Z']
        self.R = model['R']
        self.gamma = model['gamma']
        self.b0 = model['b0'].copy()

        self.nS = len(self.S)
        self.nA = len(self.A)
        self.nO = len(self.O)

        # Episode tracking
        self.belief: np.ndarray = self.b0.copy()
        self.true_state: int = sample_from_distribution(self.b0)
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.discount: float = 1.0

        # History
        self.belief_history: List[np.ndarray] = [self.belief.copy()]
        self.state_history: List[int] = [self.true_state]
        self.action_history: List[int] = []
        self.obs_history: List[int] = []
        self.reward_history: List[float] = []

    def reset(self) -> np.ndarray:
        """
        Reset simulator to initial state.

        Returns:
            Initial belief vector
        """
        self.belief = self.b0.copy()
        self.true_state = sample_from_distribution(self.b0)
        self.step_count = 0
        self.total_reward = 0.0
        self.discount = 1.0
        self.belief_history = [self.belief.copy()]
        self.state_history = [self.true_state]
        self.action_history = []
        self.obs_history = []
        self.reward_history = []
        return self.belief.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, int]:
        """
        Take one step using a given action.

        Args:
            action: action index to execute

        Returns:
            Tuple of (new_belief, reward, observation)
        """
        s = self.true_state

        # Sample next state from transition
        next_state = sample_from_distribution(self.T[action, s, :])

        # Sample observation from observation model
        obs_probs = self.Z[:, next_state]
        obs = sample_from_distribution(obs_probs)

        # Get reward
        reward = float(self.R[action, s])

        # Update belief using Bayes rule
        self.belief = compute_belief_update(self.belief, action, obs, self.T, self.Z)

        # Update tracking
        self.true_state = next_state
        self.total_reward += self.discount * reward
        self.discount *= self.gamma
        self.step_count += 1

        # Store history
        self.belief_history.append(self.belief.copy())
        self.state_history.append(next_state)
        self.action_history.append(action)
        self.obs_history.append(obs)
        self.reward_history.append(reward)

        return self.belief.copy(), reward, obs

    def run_policy(self, policy: np.ndarray, max_steps: int = 50) -> Dict[str, Any]:
        """
        Run a full episode using a given policy.

        The policy is applied to the most likely state (MAP estimate).

        Args:
            policy: array mapping state index -> action index
            max_steps: maximum number of steps

        Returns:
            Episode summary dictionary
        """
        self.reset()

        for _ in range(max_steps):
            # MAP estimate of current state
            map_state = int(np.argmax(self.belief))
            action = int(policy[map_state])
            self.step(action)

        return {
            'total_reward': self.total_reward,
            'steps': self.step_count,
            'belief_history': self.belief_history,
            'state_history': self.state_history,
            'action_history': self.action_history,
            'obs_history': self.obs_history,
            'reward_history': self.reward_history,
        }
