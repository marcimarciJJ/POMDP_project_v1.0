# A Tool for Analyzing POMDPs

A comprehensive research tool for analyzing Partially Observable Markov Decision Processes (POMDPs) with both approximate (Python) and exact (C, optional) solution algorithms.

## Quick Start

```bash
pip install -r requirements.txt

# Run the full experiment
python experiments/run_benchmark.py

# Compare solvers
python experiments/compare_solvers.py

# Run tests
pytest tests/ -v
```

## Documentation

See the `docs/` directory for full documentation:

- `docs/API.md` - API Reference
- `docs/THEORY.md` - Theoretical Background
- `docs/EXAMPLES.md` - Usage Examples
- `docs/ARCHITECTURE.md` - System Architecture
