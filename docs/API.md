# API Reference

## Core Library: `pomdp_lib`

### Solvers

#### `QMDPSolver(model_data)`
Approximate solver using standard MDP value iteration.

**Parameters:**
- `model_data` (dict): POMDP model with keys `S, A, O, T, Z, R, gamma, b0, meta`

**Methods:**
- `solve(epsilon=1e-6, max_iterations=100) -> SolverResult`

---

#### `PBVISolver(model_data)`
Point-Based Value Iteration solver with alpha vector representation.

**Parameters:**
- `model_data` (dict): POMDP model dictionary

**Methods:**
- `solve(epsilon=1e-6, max_iterations=100, num_belief_points=None) -> SolverResult`

---

#### `SolverResult` (dataclass)
Standardized result format returned by all solvers.

**Fields:**
| Field | Type | Description |
|---|---|---|
| `solver_name` | str | Solver identifier |
| `policy` | np.ndarray | Shape (nS,), optimal action per state |
| `value_function` | np.ndarray | Shape (nS,), optimal value per state |
| `convergence_trace` | List[float] | J(b0) per iteration |
| `convergence_time` | float | Solving time in seconds |
| `iterations` | int | Number of iterations |
| `final_bellman_error` | float | Bellman residual at convergence |
| `is_converged` | bool | Whether convergence criterion was met |
| `environment_id` | str | Environment identifier |
| `num_states` | int | |
| `num_actions` | int | |
| `num_observations` | int | |
| `gamma` | float | Discount factor |

**Methods:**
- `to_dict() -> Dict` — serialize to dictionary
- `save_json(filepath)` — save to JSON file
- `save_pickle(filepath)` — save to pickle file
- `load_json(filepath)` — static method, load from JSON
- `load_pickle(filepath)` — static method, load from pickle

---

### Analysis

#### `MultiSolverComparator(results)`
Compare results from multiple solvers.

**Parameters:**
- `results` (dict): mapping solver name -> SolverResult

**Methods:**
- `compare_policies() -> Dict` — pairwise policy agreement
- `compare_values() -> Dict` — pairwise value function distances
- `compare_efficiency() -> Dict` — per-solver timing/convergence
- `generate_report(output_dir) -> Dict` — full report + JSON save
- `print_summary()` — print human-readable comparison

---

#### Metrics (`pomdp_lib.analysis.metrics`)
- `compute_policy_agreement(p1, p2) -> float`
- `compute_value_l2(v1, v2) -> float`
- `compute_value_linf(v1, v2) -> float`
- `compute_solver_metrics(result) -> Dict`
- `compute_pairwise_metrics(results) -> Dict`
- `summarize_all(results) -> Dict`

---

### Visualization

#### `Visualizer(results, style)`
Generate publication-quality figures.

**Methods:**
- `plot_value_functions(output_dir)`
- `plot_convergence_traces(output_dir)`
- `plot_efficiency_comparison(output_dir)`
- `plot_policy_agreement_matrix(output_dir)`
- `plot_all(output_dir)` — generate all plots

---

#### `BeliefSimulator(model)`
Simulate policy execution with belief tracking.

**Methods:**
- `reset() -> np.ndarray` — reset to initial belief
- `step(action) -> (belief, reward, obs)` — one step
- `run_policy(policy, max_steps) -> Dict` — full episode

---

### Utils

#### `ModelExporter`
- `export(model_data, output_file)` — write `.pomdp` file
- `export_with_labels(model_data, output_file)` — with human-readable labels

#### `POMDPResultParser`
- `parse_alpha_file(alpha_file, nS) -> (matrix, actions)`
- `parse_pg_file(pg_file) -> Dict`
- `extract_policy_from_alphas(alpha_matrix, actions, nS) -> np.ndarray`
- `extract_value_function_from_alphas(alpha_matrix, b0) -> np.ndarray`

#### Helpers (`pomdp_lib.utils.helpers`)
- `normalize_belief(b) -> np.ndarray`
- `compute_belief_update(b, a, o, T, Z) -> np.ndarray`
- `sample_from_distribution(probs) -> int`
- `compute_j_b0(b0, value_function) -> float`
- `bellman_error(V_old, V_new) -> float`

#### I/O (`pomdp_lib.utils.io`)
- `save_results(results, output_path, format='json')`
- `load_results(input_path) -> Dict`
