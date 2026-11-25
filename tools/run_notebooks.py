"""Executes notebooks in place (outputs saved). Usage: python tools/run_notebooks.py 03 04 05"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
for prefix in sys.argv[1:]:
    path = next((ROOT / "notebooks").glob(f"{prefix}_*.ipynb"))
    nb = nbformat.read(path, as_version=4)
    t = time.time()
    NotebookClient(nb, timeout=3600, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    nbformat.write(nb, path)
    print(f"executed {path.name} in {time.time() - t:.0f}s")
