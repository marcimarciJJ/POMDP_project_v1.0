"""
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
        print("\n" + "="*60)
        print("POMDP SOLVER COMPARISON SUMMARY")
        print("="*60)
        
        print("\nEFFICIENCY COMPARISON:")
        print("-" * 60)
        eff = self.compare_efficiency()
        for name, metrics in eff.items():
            print(f"{name:15} Time: {metrics['time_ms']:10.2f}ms, " +
                  f"Iterations: {metrics['iterations']:3}, " +
                  f"Error: {metrics['bellman_error']:.2e}")
        
        print("\nPOLICY AGREEMENT:")
        print("-" * 60)
        policies = self.compare_policies()
        for pair, agreement in policies.items():
            print(f"{pair:30} Agreement: {agreement['agreement_rate']*100:6.2f}%")
        
        print("\nVALUE FUNCTION L2 DISTANCES:")
        print("-" * 60)
        values = self.compare_values()
        for pair, distances in values.items():
            print(f"{pair:30} L2: {distances['l2_distance']:.6f}, " +
                  f"L_inf: {distances['l_inf_distance']:.6f}")
