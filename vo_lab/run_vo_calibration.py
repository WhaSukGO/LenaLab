"""Reproduction-first calibration demo (offline, no Docker/GPU/API):

  python -m vo_lab.run_vo_calibration

Proves the evaluator is "nearly perfect" on a KNOWN answer before any autonomy is allowed
— the gate opens only when the honest ORB-VO VERIFIES and the degenerate control is
REJECTED. This is the prerequisite for Track A's autonomous committee loop."""
from __future__ import annotations

import sys

from .selftest import main

if __name__ == "__main__":
    sys.exit(main(root="./_vo_calibration_run"))
