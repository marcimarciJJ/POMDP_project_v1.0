"""
Input/Output utilities for saving and loading POMDP results.
Supports JSON and pickle formats for solver result persistence.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np


def save_results(results: Dict[str, Any], output_path: str,
                 format: str = "json") -> None:
    """
    Save solver results to disk.

    Args:
        results: dictionary of solver results
        output_path: file path to save to
        format: 'json' or 'pickle'
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        serializable = _make_serializable(results)
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
    elif format == "pickle":
        with open(path, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'pickle'.")

    print(f"Results saved to {output_path}")


def load_results(input_path: str) -> Dict[str, Any]:
    """
    Load solver results from disk.

    Args:
        input_path: file path to load from

    Returns:
        Dictionary of results
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if path.suffix == ".json":
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    elif path.suffix in (".pkl", ".pickle"):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        raise ValueError(f"Unknown file extension: {path.suffix}")


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    else:
        return obj
