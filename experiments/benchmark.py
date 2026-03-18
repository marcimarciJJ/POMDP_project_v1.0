"""
Performance benchmark script.
Measures solving time and convergence speed for QMDP and PBVI.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from pomdp_lib.models import model_3x3, model_5x5
from pomdp_lib.solvers.qmdp import QMDPSolver
from pomdp_lib.solvers.pbvi import PBVISolver


def benchmark_solver(solver_class, model, n_runs: int = 5, **kwargs):
    """
    Benchmark a solver by running it multiple times.

    Args:
        solver_class: solver class to instantiate
        model: POMDP model dictionary
        n_runs: number of benchmark runs
        **kwargs: solver arguments

    Returns:
        Dictionary with timing statistics
    """
    times = []
    iterations = []
    for _ in range(n_runs):
        solver = solver_class(model)
        result = solver.solve(**kwargs)
        times.append(result.convergence_time * 1000)
        iterations.append(result.iterations)

    return {
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'mean_iterations': float(np.mean(iterations)),
    }


def main():
    """Run benchmarks on 3x3 and 5x5 models."""
    models = {
        '3x3': {
            'S': model_3x3.S, 'A': model_3x3.A, 'O': model_3x3.O,
            'T': model_3x3.T, 'Z': model_3x3.Z, 'R': model_3x3.R,
            'gamma': model_3x3.gamma, 'b0': model_3x3.b0,
            'meta': {'width': 3, 'height': 3, 'environment_id': '3x3'}
        },
        '5x5': {
            'S': model_5x5.S, 'A': model_5x5.A, 'O': model_5x5.O,
            'T': model_5x5.T, 'Z': model_5x5.Z, 'R': model_5x5.R,
            'gamma': model_5x5.gamma, 'b0': model_5x5.b0,
            'meta': {'width': 5, 'height': 5, 'environment_id': '5x5'}
        }
    }

    n_runs = 3

    print("\n" + "="*70)
    print("POMDP Solver Performance Benchmark")
    print(f"Runs per solver: {n_runs}")
    print("="*70)

    for env_id, model in models.items():
        nS = len(model['S'])
        print(f"\n[{env_id}] {nS} states")
        print("-" * 50)

        for solver_name, solver_class, kwargs in [
            ('QMDP', QMDPSolver, {'epsilon': 1e-6, 'max_iterations': 100}),
            ('PBVI', PBVISolver, {'epsilon': 1e-6, 'max_iterations': 100,
                                  'num_belief_points': max(nS, 20)}),
        ]:
            stats = benchmark_solver(solver_class, model, n_runs=n_runs, **kwargs)
            print(f"  {solver_name:6}: {stats['mean_time_ms']:.2f}ms ± "
                  f"{stats['std_time_ms']:.2f}ms | "
                  f"{stats['mean_iterations']:.1f} iterations avg")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
