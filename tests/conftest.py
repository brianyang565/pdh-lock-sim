import sys
from pathlib import Path

# Allow ``import pdh_sim`` when pytest is run from the repository root
# without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
