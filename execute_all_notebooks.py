"""
Execute all notebooks (05 through 12) sequentially and record execution time and status.
"""

import sys
import os
import time
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS = [
    '05_knn.ipynb',
    '06_decision_tree.ipynb',
    '07_random_forest.ipynb',
    '08_xgboost.ipynb',
    '09_lightgbm.ipynb',
    '10_mlp.ipynb',
    '11_feature_selection_and_pca.ipynb',
    '12_final_model_comparison.ipynb'
]

def execute_notebook(nb_filename):
    nb_path = os.path.join('notebooks', nb_filename)
    print(f"\n{'='*70}", flush=True)
    print(f"EXECUTING: {nb_filename}", flush=True)
    print(f"{'='*70}", flush=True)
    
    t0 = time.time()
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=900, kernel_name='hotel_venv')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': 'notebooks'}})
        elapsed = time.time() - t0
        print(f"SUCCESS: {nb_filename} executed cleanly in {elapsed:.2f} seconds!", flush=True)
        
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        return True, elapsed, None
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERROR executing {nb_filename} after {elapsed:.2f} seconds:", flush=True)
        print(e, flush=True)
        return False, elapsed, str(e)

def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else NOTEBOOKS
    summary = []
    
    for nb_name in targets:
        success, duration, error = execute_notebook(nb_name)
        summary.append((nb_name, success, duration, error))
        if not success:
            print(f"\nExecution aborted due to error in {nb_name}!")
            break
            
    print("\n" + "="*70)
    print("EXECUTION SUMMARY:")
    print("="*70)
    all_ok = True
    for name, success, duration, error in summary:
        status = "PASSED [OK]" if success else f"FAILED [ERR: {error[:30]}]"
        print(f"{name:<35} | {status:<25} | {duration:6.2f}s")
        if not success:
            all_ok = False
            
    if all_ok:
        print("\nALL NOTEBOOKS EXECUTED WITH ZERO ERRORS!")
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
