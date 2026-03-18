"""
Multi-solver comparison script.
Directly compare QMDP and PBVI on the Foggy Forest environments.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from pomdp_lib.models import model_3x3
from pomdp_lib.solvers.qmdp import QMDPSolver
from pomdp_lib.solvers.pbvi import PBVISolver
from pomdp_lib.analysis.comparator import MultiSolverComparator
from pomdp_lib.analysis.metrics import summarize_all


def main():
    """Compare all solvers on the 3x3 model."""
    model = {
        'S': model_3x3.S,
        'A': model_3x3.A,
        'O': model_3x3.O,
        'T': model_3x3.T,
        'Z': model_3x3.Z,
        'R': model_3x3.R,
        'gamma': model_3x3.gamma,
        'b0': model_3x3.b0,
        'meta': {'width': 3, 'height': 3, 'environment_id': '3x3'}
    }

    print("\n" + "="*70)
    print("Solver Comparison on 3x3 Foggy Forest Model")
    print("="*70)

    results = {}
    results['QMDP'] = QMDPSolver(model).solve(epsilon=1e-6, max_iterations=100)
    results['PBVI'] = PBVISolver(model).solve(epsilon=1e-6, max_iterations=100,
                                               num_belief_points=20)

    comparator = MultiSolverComparator(results)
    comparator.print_summary()

    summary = summarize_all(results)
    print("\nPer-solver metrics:")
    for name, m in summary['per_solver'].items():
        print(f"  {name}: time={m['time_ms']:.2f}ms, iter={m['iterations']}, "
              f"J(b0)={m['J_b0']:.4f}")

    print("\nPairwise metrics:")
    for pair, m in summary['pairwise'].items():
        print(f"  {pair}: agreement={m['policy_agreement']*100:.1f}%, "
              f"L2={m['value_l2_distance']:.4f}")


if __name__ == "__main__":
    main()
