"""
LEGACY — use src/physics/common/significance_plot.py instead.

    python3 src/physics/common/significance_plot.py --analysis Fiducial [options]

All logic has been migrated. This file is kept for reference only.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

rprint(
    "[yellow][DEPRECATED][/yellow] fiducialization_plot.py is legacy. "
    "Use [bold]src/physics/common/significance_plot.py --analysis Fiducial[/bold] instead."
)
sys.exit(1)
