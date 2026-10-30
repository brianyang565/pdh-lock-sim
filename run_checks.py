"""Run every model check and report the result of each.

Open this file in Spyder and press F5 (Run file). No extra packages are
needed. Each check prints PASS or FAIL together with a one-line
description; when a check fails, the error message follows it.

The checks themselves are the functions whose names begin with ``test_``
in the files of the ``tests`` folder. A new check is added by writing a
new ``test_`` function there; this script finds it automatically.
"""

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Test modules to run, in order. Add later modules (for example
# "tests.test_pdh") to this list as the project grows.
MODULES = ["tests.test_cavity", "tests.test_pdh", "tests.test_loop", "tests.test_step4"]


def run_module(name):
    """Run every ``test_`` function in one module; return (passed, failed)."""
    module = importlib.import_module(name)
    importlib.reload(module)  # pick up edits when re-run in the same Spyder session
    passed = failed = 0
    print(f"\n{name}")
    for attr in sorted(dir(module)):
        if not attr.startswith("test_"):
            continue
        func = getattr(module, attr)
        doc = (func.__doc__ or "").strip().splitlines()
        summary = doc[0] if doc else attr
        try:
            func()
        except Exception:
            failed += 1
            print(f"  FAIL  {attr}: {summary}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"  PASS  {attr}: {summary}")
    return passed, failed


def main():
    # Reload the model itself too, so that edits to pdh_sim are used on re-runs.
    for mod in [m for m in sys.modules if m == "pdh_sim" or m.startswith("pdh_sim.")]:
        importlib.reload(sys.modules[mod])

    total_pass = total_fail = 0
    for name in MODULES:
        p, f = run_module(name)
        total_pass += p
        total_fail += f
    print(f"\n{total_pass} passed, {total_fail} failed")
    return total_fail


if __name__ == "__main__":
    main()
