# src/config.py
from pathlib import Path

# ---------- Project paths ----------
# ROOT_DIR = project root folder (the parent of src/)
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data folders
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results folder
RESULTS_DIR = ROOT_DIR / "results"

# Make sure these folders exist
for d in (DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Project constants ----------
# Country code used in World Bank / PovcalNet etc.
COUNTRY_CODE = "PHL"

# Min / max years for the macro panel
YEAR_MIN = 2004
YEAR_MAX = 2024
