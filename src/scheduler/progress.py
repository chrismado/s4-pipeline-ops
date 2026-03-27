"""
Job progress parsing — extracts completion percentage from stdout.

Supports common patterns from ML training and render pipelines:
  - [50/100]          — step counters
  - 50%               — percentage markers
  - Epoch 5/10        — epoch counters
  - Step 500/1000     — step counters
  - Progress: 0.75    — decimal fractions
"""

import re

# Patterns ordered by specificity — first match wins
_PATTERNS = [
    # [50/100] or [50 / 100]
    re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]"),
    # Epoch 5/10 or Step 500/1000
    re.compile(r"(?:epoch|step|batch|iteration|frame)\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE),
    # 75% or 75.5%
    re.compile(r"(\d+(?:\.\d+)?)\s*%"),
    # Progress: 0.75 or progress=0.75
    re.compile(r"progress[\s:=]+(\d+\.\d+)", re.IGNORECASE),
]


def parse_progress(line: str) -> float | None:
    """
    Extract a progress percentage (0-100) from a line of output.

    Returns None if no progress pattern is found.
    """
    for pattern in _PATTERNS:
        m = pattern.search(line)
        if not m:
            continue

        groups = m.groups()
        if len(groups) == 2:
            # Fraction pattern: current/total
            current, total = float(groups[0]), float(groups[1])
            if total > 0:
                return min((current / total) * 100, 100.0)
        elif len(groups) == 1:
            val = float(groups[0])
            # If the pattern was the % one, val is already a percentage
            if "%" in pattern.pattern:
                return min(val, 100.0)
            # Decimal fraction (0.0 - 1.0)
            if 0.0 <= val <= 1.0:
                return val * 100.0
            # Could be a percentage already
            if val <= 100.0:
                return val

    return None
