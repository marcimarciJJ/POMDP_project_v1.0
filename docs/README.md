# A Tool for Analyzing POMDPs

A comprehensive tool for analyzing Partially Observable Markov Decision Processes (POMDPs) with both exact and approximate solution algorithms.

## Overview

This project provides a unified framework for analyzing POMDP models through:
- **Approximate Algorithms**: QMDP and Point-Based Value Iteration (PBVI) in Python
- **Exact Algorithms**: Witness and Incremental Pruning (IP) via C-based pomdp-solve (optional)
- **Analysis Tools**: Multi-solver comparison, performance metrics, and visualization

## Experiment Domain

The tool is demonstrated on the **Foggy Forest** environment — a 2D grid-based navigation task where an agent must reach an exit while avoiding traps and obstacles under partial observability (sensor uncertainty).

## Mathematical Formulation

The environment is formalized as a POMDP 7-tuple: $(S, A, T, R, \Omega, O, \gamma)$

- $S$: State space (grid cells)
- $A$: Action space {N, S, W, E, Stay}
- $T$: Transition probabilities $P(s'|s,a)$
- $R$: Reward function $R(a,s)$
- $\Omega$: Observation space (sensor readings)
- $O$: Observation probabilities $P(o|s',a)$
- $\gamma$: Discount factor (0.95)

## Architecture

### Dual-Track Solving Strategy

```
POMDP Model
    ├─→ Approximate Algorithms (Python)
    │   ├─ QMDP: Quasi-optimal MDP approach
    │   └─ PBVI: Point-Based Value Iteration
    │
    └─→ Exact Algorithms (C via pomdp-solve, optional)
        ├─ Witness Algorithm
        └─ Incremental Pruning (IP)
```

## Project Structure

```
POMDP_project_v1.0/
├── src/pomdp_lib/          # Core library
│   ├── models/             # POMDP model definitions
│   ├── solvers/            # Solver implementations
│   ├── analysis/           # Analysis and comparison tools
│   ├── vis/                # Visualization
│   └── utils/              # Utilities (I/O, model export, parsing)
├── experiments/            # Experiment scripts
├── external/               # Interface to pomdp-solve C library
├── scripts/                # Convenience scripts
├── config/                 # Configuration files
├── data/                   # Results and outputs
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── requirements.txt        # Python dependencies
```

## Quick Start

```python
import sys
sys.path.insert(0, 'src')

from pomdp_lib.models import model_3x3
from pomdp_lib.solvers.qmdp import QMDPSolver
from pomdp_lib.solvers.pbvi import PBVISolver
from pomdp_lib.analysis.comparator import MultiSolverComparator

# Build model dictionary
model = {
    'S': model_3x3.S, 'A': model_3x3.A, 'O': model_3x3.O,
    'T': model_3x3.T, 'Z': model_3x3.Z, 'R': model_3x3.R,
    'gamma': model_3x3.gamma, 'b0': model_3x3.b0,
    'meta': {'width': 3, 'height': 3, 'environment_id': '3x3'}
}

# Solve
qmdp_result = QMDPSolver(model).solve()
pbvi_result = PBVISolver(model).solve()

# Compare
comparator = MultiSolverComparator({'QMDP': qmdp_result, 'PBVI': pbvi_result})
comparator.print_summary()
```

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Experiments

```bash
# Full experiment with visualization
python experiments/run_benchmark.py

# Quick solver comparison
python experiments/compare_solvers.py

# Performance benchmark
python experiments/benchmark.py
```

## Documentation

- `docs/API.md` - API Reference
- `docs/THEORY.md` - Theoretical Background
- `docs/EXAMPLES.md` - Usage Examples
- `docs/ARCHITECTURE.md` - System Architecture

## Authors

Research Team - POMDP Analysis Project
