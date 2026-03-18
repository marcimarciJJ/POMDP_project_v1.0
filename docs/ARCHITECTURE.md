# System Architecture

## Overview

The POMDP Analysis Tool follows a layered architecture with clear separation between model definition, solving, analysis, and visualization.

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Scripts                        │
│   run_benchmark.py  compare_solvers.py  benchmark.py        │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    Core Library (pomdp_lib)                  │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ models/ │  │ solvers/ │  │analysis/ │  │    vis/     │  │
│  │         │  │          │  │          │  │             │  │
│  │model_3x3│  │ QMDP     │  │Comparator│  │ Visualizer  │  │
│  │model_5x5│  │ PBVI     │  │ Metrics  │  │ Charts      │  │
│  │generator│  │ Interface│  │          │  │ Simulator   │  │
│  └─────────┘  └──────────┘  └──────────┘  └─────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    utils/                            │   │
│  │  helpers.py  io.py  model_exporter.py  result_parser │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│               External C Solvers (optional)                  │
│               pomdp-solve (Witness, IP)                      │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Model Module (model_3x3.py)
    │
    ▼  to_dict()
Model Dictionary {S, A, O, T, Z, R, gamma, b0, meta}
    │
    ├──▶ QMDPSolver.solve() ──▶ SolverResult
    │
    ├──▶ PBVISolver.solve() ──▶ SolverResult
    │
    └──▶ ExactSolver.solve() ──▶ SolverResult (optional)
              │
              ▼ {QMDP: result, PBVI: result, ...}
         MultiSolverComparator ──▶ Report / JSON
              │
              ▼
         Visualizer ──▶ PNG plots
```

## Key Design Decisions

1. **Unified SolverResult format**: All solvers return the same dataclass, enabling drop-in comparison.
2. **Model as dictionary**: Keeps models simple and compatible with any solver.
3. **Separated analysis**: Comparator and metrics are independent of solvers.
4. **Optional C integration**: External solver is wrapped gracefully with fallback.
