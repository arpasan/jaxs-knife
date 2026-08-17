"""Put skill scripts on sys.path for L0 tests."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "jaxs-knife" / "scripts"
sys.path.insert(0, str(SCRIPTS))
