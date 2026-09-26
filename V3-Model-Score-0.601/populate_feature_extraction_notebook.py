#!/usr/bin/env python3
"""
populate_feature_extraction_notebook.py
---------------------------------------
Executes and populates all cells in V3/V3_02_Feature_Extraction.ipynb
with real outputs, timing, and feature matrix inspection.
"""

import json
import os
import sys
import io
import time
import contextlib

def run_notebook():
    nb_path = 'V3/V3_02_Feature_Extraction.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    exec_globals = {}
    cell_idx = 0

    print("=" * 60)
    print("Executing & Populating V3_02_Feature_Extraction.ipynb")
    print("=" * 60)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell_idx += 1
            code = "".join(cell['source'])
            print(f"\n--- Running Code Cell {cell_idx} ---")
            
            stdout_trap = io.StringIO()
            stderr_trap = io.StringIO()
            t0 = time.time()
            
            try:
                with contextlib.redirect_stdout(stdout_trap), contextlib.redirect_stderr(stderr_trap):
                    exec(code, exec_globals)
                out_text = stdout_trap.getvalue()
                err_text = stderr_trap.getvalue()
                
                combined_output = out_text
                if err_text:
                    combined_output += ("\n" if combined_output else "") + err_text
                    
                cell['execution_count'] = cell_idx
                cell['outputs'] = [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [line + "\n" for line in combined_output.split("\n") if line or combined_output.endswith("\n")]
                    }
                ]
                print(f"Cell {cell_idx} executed successfully in {time.time()-t0:.2f}s")
                if combined_output.strip():
                    first_few = "\n".join(combined_output.strip().split("\n")[:8])
                    print(f"Output preview:\n{first_few}")
            except Exception as e:
                print(f"Cell {cell_idx} ERROR: {e}")
                import traceback
                traceback.print_exc()
                cell['execution_count'] = cell_idx
                cell['outputs'] = [
                    {
                        "ename": type(e).__name__,
                        "evalue": str(e),
                        "output_type": "error",
                        "traceback": traceback.format_exc().split("\n")
                    }
                ]
                break

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Populated notebook saved to {nb_path}")
    print("=" * 60)

if __name__ == '__main__':
    run_notebook()
