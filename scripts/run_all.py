"""Re-run every experiment of this repository and regenerate all figures.

    uv run python scripts/run_all.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = ["run_cylinder.py"]

for script in EXPERIMENTS:
    print(f"\n##### {script} #####", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, script)], check=True)
