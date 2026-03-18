# Usage Examples

## Example 1: Quick Start with 3x3 Model

```python
import sys
sys.path.insert(0, 'src')

from pomdp_lib.models import model_3x3
from pomdp_lib.solvers.qmdp import QMDPSolver

model = {
    'S': model_3x3.S, 'A': model_3x3.A, 'O': model_3x3.O,
    'T': model_3x3.T, 'Z': model_3x3.Z, 'R': model_3x3.R,
    'gamma': model_3x3.gamma, 'b0': model_3x3.b0,
    'meta': {'width': 3, 'height': 3, 'environment_id': '3x3'}
}

result = QMDPSolver(model).solve()
print(f"Converged in {result.iterations} iterations")
print(f"Policy: {result.policy}")
print(f"J(b0) = {result.convergence_trace[-1]:.4f}")
```

## Example 2: Compare QMDP and PBVI

```python
from pomdp_lib.solvers.qmdp import QMDPSolver
from pomdp_lib.solvers.pbvi import PBVISolver
from pomdp_lib.analysis.comparator import MultiSolverComparator

results = {
    'QMDP': QMDPSolver(model).solve(),
    'PBVI': PBVISolver(model).solve(num_belief_points=20),
}

comparator = MultiSolverComparator(results)
comparator.print_summary()
```

## Example 3: Visualization

```python
from pomdp_lib.vis.visualizer import Visualizer

viz = Visualizer(results)
viz.plot_all('./output')
```

## Example 4: Simulate a Policy

```python
from pomdp_lib.vis.simulator import BeliefSimulator

sim = BeliefSimulator(model)
episode = sim.run_policy(results['QMDP'].policy, max_steps=20)
print(f"Total reward: {episode['total_reward']:.2f}")
```

## Example 5: Generate Custom Model

```python
from pomdp_lib.models.generator import FoggyForestGenerator

gen = FoggyForestGenerator(
    width=4, height=4,
    exit_cell=(4, 4),
    traps=[(1, 1), (3, 2)],
    trees=[(2, 2)],
    robot_start=(1, 4),
    initial_belief_type='deterministic'
)
model_data = gen.generate()
```

## Example 6: Export to .pomdp Format

```python
from pomdp_lib.utils.model_exporter import ModelExporter

ModelExporter.export(model, './data/models/model_3x3.pomdp')
```

## Example 7: Save and Load Results

```python
from pomdp_lib.utils.io import save_results, load_results

# Save
save_results({'QMDP': result.to_dict()}, './data/results.json')

# Load
data = load_results('./data/results.json')
```
