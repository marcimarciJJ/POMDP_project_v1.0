"""
Complete POMDP Analysis Demo for Colab
End-to-end demonstration of all solvers and analysis.
"""

import sys
import os
import time
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
external_path = os.path.abspath(os.path.join(current_dir, "../external"))
sys.path.insert(0, src_path)
sys.path.insert(0, external_path)

import numpy as np


def main():
    """Run complete POMDP analysis demo."""
    
    print("\n" + "="*70)
    print(" "*15 + "POMDP Analysis Tool - Complete Demo")
    print("="*70)
    
    # ===== Step 1: Load Models =====
    print("\n[Step 1] Loading POMDP Models...")
    print("-" * 70)
    
    from pomdp_lib.models.model_3x3 import S as S_3x3, A, O as O_3x3
    from pomdp_lib.models.model_3x3 import T as T_3x3, Z as Z_3x3, R as R_3x3, b0 as b0_3x3, gamma
    
    model_3x3 = {
        'S': S_3x3, 'A': A, 'O': O_3x3, 'T': T_3x3, 'Z': Z_3x3, 'R': R_3x3,
        'gamma': gamma, 'b0': b0_3x3,
        'meta': {'width': 3, 'height': 3, 'environment_id': '3x3'}
    }
    
    print(f"✓ 3x3 model loaded: {len(S_3x3)} states, {len(A)} actions, {len(O_3x3)} observations")
    
    # ===== Step 2: Run Approximate Algorithms =====
    print("\n[Step 2] Running Approximate Algorithms (QMDP & PBVI)...")
    print("-" * 70)
    
    from pomdp_lib.solvers.qmdp import QMDPSolver
    from pomdp_lib.solvers.pbvi import PBVISolver
    
    results = {}
    
    print("  Running QMDP on 3x3 model...")
    qmdp = QMDPSolver(model_3x3)
    result_qmdp = qmdp.solve(epsilon=1e-6, max_iterations=100)
    results["QMDP"] = result_qmdp
    print(f"  ✓ QMDP: {result_qmdp.convergence_time:.4f}s, {result_qmdp.iterations} iterations")
    
    print("  Running PBVI on 3x3 model...")
    pbvi = PBVISolver(model_3x3)
    result_pbvi = pbvi.solve(epsilon=1e-6, max_iterations=100)
    results["PBVI"] = result_pbvi
    print(f"  ✓ PBVI: {result_pbvi.convergence_time:.4f}s, {result_pbvi.iterations} iterations")
    
    # ===== Step 3: Comparison Analysis =====
    print("\n[Step 3] Comparative Analysis...")
    print("-" * 70)
    
    from pomdp_lib.analysis.comparator import MultiSolverComparator
    
    comparator = MultiSolverComparator(results)
    
    output_dir = "./results"
    Path(output_dir).mkdir(exist_ok=True)
    
    report = comparator.generate_report(output_dir)
    
    comparator.print_summary()
    
    # ===== Step 4: Visualization =====
    print("\n[Step 4] Generating Visualizations...")
    print("-" * 70)
    
    from pomdp_lib.vis.visualizer import Visualizer
    
    visualizer = Visualizer(results)
    visualizer.plot_all(output_dir)
    
    # ===== Final Summary =====
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"Results saved to: {output_dir}/")
    print("\n✓ All analysis complete!")


if __name__ == "__main__":
    main()
