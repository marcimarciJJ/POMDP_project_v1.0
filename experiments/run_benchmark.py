"""
Main experiment script for POMDP analysis.
Runs all solvers on the 3x3 and 5x5 Foggy Forest models
and generates a comprehensive comparison report.
"""

import sys
import os
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np

from pomdp_lib.models import model_3x3, model_5x5
from pomdp_lib.solvers.qmdp import QMDPSolver
from pomdp_lib.solvers.pbvi import PBVISolver
from pomdp_lib.analysis.comparator import MultiSolverComparator
from pomdp_lib.vis.visualizer import Visualizer
from pomdp_lib.utils.io import save_results


def build_model(module, env_id: str, width: int, height: int) -> dict:
    """Build a model dictionary from a model module."""
    return {
        'S': module.S,
        'A': module.A,
        'O': module.O,
        'T': module.T,
        'Z': module.Z,
        'R': module.R,
        'gamma': module.gamma,
        'b0': module.b0,
        'meta': {
            'width': width,
            'height': height,
            'environment_id': env_id
        }
    }


def run_experiment(model: dict, output_dir: str) -> dict:
    """
    Run QMDP and PBVI on a POMDP model and generate outputs.

    Args:
        model: POMDP model dictionary
        output_dir: directory to save results

    Returns:
        Dictionary of solver results
    """
    env_id = model['meta']['environment_id']
    print(f"\n{'='*60}")
    print(f"Running experiment: {env_id} model")
    print(f"States: {len(model['S'])}, Actions: {len(model['A'])}, Observations: {len(model['O'])}")
    print(f"{'='*60}")

    results = {}

    # QMDP
    print("\n[QMDP] Solving...")
    qmdp_solver = QMDPSolver(model)
    results['QMDP'] = qmdp_solver.solve(epsilon=1e-6, max_iterations=100)
    r = results['QMDP']
    print(f"  ✓ Converged in {r.iterations} iterations, {r.convergence_time:.4f}s, "
          f"error={r.final_bellman_error:.2e}")

    # PBVI
    print("\n[PBVI] Solving...")
    pbvi_solver = PBVISolver(model)
    results['PBVI'] = pbvi_solver.solve(epsilon=1e-6, max_iterations=100)
    r = results['PBVI']
    print(f"  ✓ Converged in {r.iterations} iterations, {r.convergence_time:.4f}s, "
          f"error={r.final_bellman_error:.2e}")

    # Comparison
    print("\n[Analysis] Comparing solvers...")
    comparator = MultiSolverComparator(results)
    comparator.print_summary()
    report = comparator.generate_report(output_dir)

    # Visualization
    print("\n[Visualization] Generating plots...")
    try:
        viz = Visualizer(results)
        viz.plot_all(output_dir)
    except Exception as e:
        print(f"  Warning: visualization failed ({e})")

    # Save results
    serializable = {
        name: res.to_dict() for name, res in results.items()
    }
    save_results(serializable, f"{output_dir}/results.json", format="json")

    return results


def main():
    """Run experiments on both 3x3 and 5x5 models."""
    print("\n" + "="*70)
    print(" "*15 + "POMDP Analysis Experiment Runner")
    print("="*70)

    base_dir = Path(__file__).parent.parent / "data" / "results"

    # 3x3 model
    model3 = build_model(model_3x3, '3x3', 3, 3)
    run_experiment(model3, str(base_dir / "3x3"))

    # 5x5 model
    model5 = build_model(model_5x5, '5x5', 5, 5)
    run_experiment(model5, str(base_dir / "5x5"))

    print("\n" + "="*70)
    print("All experiments complete!")
    print(f"Results saved to: {base_dir}")


if __name__ == "__main__":
    main()
