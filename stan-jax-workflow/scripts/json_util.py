"""JSON helpers shared by diagnostic scripts."""

from __future__ import annotations

from typing import Any

import numpy as np


def json_default(obj: Any) -> Any:
    """Coerce numpy scalars/arrays so ``json.dumps`` never emits invalid NaN tokens."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
