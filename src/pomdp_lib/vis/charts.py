"""
Specialized chart generation for POMDP analysis.
Provides chart functions used by the Visualizer.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from solvers.unified_interface import SolverResult


def plot_grid_policy(result: SolverResult, width: int, height: int,
                     ax: Optional[plt.Axes] = None,
                     trees: Optional[List[Tuple[int, int]]] = None) -> plt.Axes:
    """
    Plot policy as arrows on a 2D grid.

    Args:
        result: solver result containing policy
        width: grid width
        height: grid height
        ax: matplotlib axes to plot on
        trees: list of tree (obstacle) positions to mark

    Returns:
        Matplotlib axes
    """
    action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→', 4: '·'}  # N, S, W, E, Stay
    action_dx = {0: 0, 1: 0, 2: -0.3, 3: 0.3, 4: 0}
    action_dy = {0: 0.3, 1: -0.3, 2: 0, 3: 0, 4: 0}

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # Draw grid
    for x in range(width + 1):
        ax.axvline(x, color='gray', linewidth=0.5, alpha=0.5)
    for y in range(height + 1):
        ax.axhline(y, color='gray', linewidth=0.5, alpha=0.5)

    # Draw arrows for each state
    policy = result.policy
    nS = result.num_states
    for s in range(min(nS, width * height)):
        a = int(policy[s])
        # Estimate grid position from state index
        x = (s % width) + 0.5
        y = (s // width) + 0.5
        sym = action_symbols.get(a, '?')
        ax.text(x, y, sym, ha='center', va='center', fontsize=14, fontweight='bold')

    ax.set_title(f"Policy - {result.solver_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return ax


def plot_value_heatmap(result: SolverResult, width: int, height: int,
                       ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot value function as a heatmap on a 2D grid.

    Args:
        result: solver result containing value_function
        width: grid width
        height: grid height
        ax: matplotlib axes to plot on

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    nS = result.num_states
    values = result.value_function[:nS]

    # Reshape to grid (best effort)
    grid_values = np.zeros((height, width))
    for s in range(min(nS, width * height)):
        row = s // width
        col = s % width
        if row < height and col < width:
            grid_values[row, col] = values[s]

    im = ax.imshow(grid_values, origin='lower', cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax, label='Value')
    ax.set_title(f"Value Function - {result.solver_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return ax


def plot_convergence_comparison(results: Dict[str, SolverResult],
                                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot convergence traces for multiple solvers on one axes.

    Args:
        results: dictionary of solver name -> SolverResult
        ax: matplotlib axes to plot on

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    for name, result in results.items():
        trace = result.convergence_trace
        if trace:
            ax.plot(range(len(trace)), trace, marker='o', label=name,
                    linewidth=2, markersize=3)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("J(b₀)", fontsize=11)
    ax.set_title("Convergence Comparison", fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    return ax
