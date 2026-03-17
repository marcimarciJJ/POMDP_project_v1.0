import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Literal  

class FoggyForestGenerator:
    """
    Foggy Forest Environment Generator.
    """

    def __init__(self, 
                 width: int, 
                 height: int, 
                 exit_cell: Tuple[int, int], 
                 traps: List[Tuple[int, int]], 
                 trees: List[Tuple[int, int]], 
                 robot_start: Optional[Tuple[int, int]] = None,
                 initial_belief_type: Literal["deterministic", "uniform", "safe_uniform"] = "deterministic",
                 p_move: float = 0.8,
                 gamma: float = 0.95):
        self.width = width
        self.height = height
        self.exit_cell = exit_cell
        self.traps = set(traps)
        self.trees = set(trees)
        self.robot_start = robot_start
        self.initial_belief_type = initial_belief_type 
        self.p_move = p_move
        self.gamma = gamma
        self.data: Dict[str, Any] = {}
        self.A = ["N", "S", "W", "E", "Stay"]

        """
        Initialize the environment configuration.
        
        Args:
            width: grid width (x dimension)
            height: grid height (y dimension)
            exit_cell: goal position (x, y)
            traps: list of trap positions
            trees: list of tree positions (obstacles, excluded from state space)
            robot_start: initial position of robot (required if initial_belief_type="deterministic")
            initial_belief_type: how to initialize belief distribution
                - "deterministic": agent position is known exactly at robot_start
                - "uniform": agent position is completely uncertain (uniform over all states)
                - "safe_uniform": agent position is uniform over non-trap, non-exit states
            p_move: probability of successful movement (default 0.8)
            gamma: discount factor (default 0.95)
        """


    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _in_bounds(self, x: int, y: int) -> bool:
        return 1 <= x <= self.width and 1 <= y <= self.height

    def generate(self) -> Dict[str, Any]:
        # 1. Build States S (exclude trees)
        S: List[Tuple[int, int]] = []
        for x in range(1, self.width + 1):
            for y in range(1, self.height + 1):
                if (x, y) in self.trees: continue
                S.append((x, y))
        S.sort()
        
        nS = len(S)
        nA = len(self.A)
        idx_map: Dict[Tuple[int, int], int] = {s: i for i, s in enumerate(S)}

        # 2. Build Transitions T
        T = np.zeros((nA, nS, nS), dtype=float)
        for ai, action in enumerate(self.A):
            for si, s in enumerate(S):
                if s == self.exit_cell or s in self.traps or action == "Stay":
                    T[ai, si, si] = 1.0
                    continue
                x, y = s
                if action == "N": nxt = (x, y + 1)
                elif action == "S": nxt = (x, y - 1)
                elif action == "W": nxt = (x - 1, y)
                elif action == "E": nxt = (x + 1, y)
                else: nxt = (x, y)

                if self._in_bounds(*nxt) and nxt not in self.trees:
                    T[ai, si, idx_map[nxt]] += self.p_move
                    T[ai, si, si] += (1.0 - self.p_move)
                else:
                    T[ai, si, si] = 1.0

        # 3. Build Rewards R
        move_cost, stay_cost = -1.0, -0.5
        R = np.zeros((nA, nS), dtype=float)
        for ai, a in enumerate(self.A):
            for si, s in enumerate(S):
                if s == self.exit_cell or s in self.traps:
                    R[ai, si] = 0.0
                    continue
                base = stay_cost if a == "Stay" else move_cost
                exp_r = 0.0
                for sj in range(nS):
                    prob = T[ai, si, sj]
                    if prob > 0:
                        s_next = S[sj]
                        bonus = 10.0 if s_next == self.exit_cell else (-10.0 if s_next in self.traps else 0.0)
                        exp_r += prob * (base + bonus)
                R[ai, si] = exp_r

        # 4. Build Observations Z
        def get_label(s):
            active = []
            if s == self.exit_cell or s[0] in [1, self.width] or s[1] in [1, self.height]:
                active.append("see-clearing")
            if s in self.traps or any(self._manhattan(s, t) == 1 for t in self.traps):
                active.append("hear-wind")
            if any(self._manhattan(s, t) == 1 for t in self.trees):
                active.append("hear-bird")
            
            if not active: return "none"
            base_order = {"see-clearing": 0, "hear-wind": 1, "hear-bird": 2}
            active.sort(key=lambda k: base_order[k])
            return "+".join(active)

        all_labels = [get_label(s) for s in S]
        OBS = sorted(list(set(all_labels)))
        nO = len(OBS)
        Z = np.zeros((nO, nS), dtype=float)
        state_obs_map = {}
        for si, label in enumerate(all_labels):
            Z[OBS.index(label), si] = 1.0
            state_obs_map[S[si]] = label

        # 5. Initial Belief b0 
        b0 = np.zeros(nS, dtype=float)
        if self.initial_belief_type == "deterministic" and self.robot_start in idx_map:
            b0[idx_map[self.robot_start]] = 1.0
        elif self.initial_belief_type == "safe_uniform":
            safe_indices = [i for i, s in enumerate(S) if s not in self.traps and s != self.exit_cell]
            b0[safe_indices] = 1.0 / len(safe_indices)
        else: # "uniform" 
            b0[:] = 1.0 / nS

        self.data = {"S": S, "A": self.A, "O": OBS, "T": T, "Z": Z, "R": R, 
                     "gamma": self.gamma, "b0": b0, "state_obs_map": state_obs_map,
                     "meta": {"width": self.width, "height": self.height, "EXIT": self.exit_cell,
                              "TRAPS": list(self.traps), "TREES": list(self.trees)}}
        return self.data

    def save_to_file(self, filename: str):
        if not self.data: self.generate()
        path = Path(filename)
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w") as f:
            f.write("import numpy as np\n\ngamma = " + str(self.data['gamma']) + "\n")
            f.write(f"S = {self.data['S']!r}\nA = {self.data['A']!r}\nO = {self.data['O']!r}\n")
            for name in ['T', 'Z', 'R', 'b0']:
                f.write(f"{name} = np.array({self.data[name].tolist()})\n")