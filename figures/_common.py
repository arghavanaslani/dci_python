"""Shared setup for figure scripts."""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless

# make the dci package importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

OUTPUT_DIR = os.path.join(_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    return path
