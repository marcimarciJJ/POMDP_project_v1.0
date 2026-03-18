"""
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
            ax.set_title(f"{solver_name}\nTime: {result.convergence_time:.4f}s, " +
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
