import pandas as pd

from .config import DATA_DIR
from .clean_data import build_macro_panel
from .clean_disasters import clean_emdat_phl

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)


def build_full_panel():
    # Panel macro (poverty + GDP, déjà filtré PHL + 2004–2024)
    macro = build_macro_panel()          # colonnes : year, poverty_rate, gdp_growth

    # Panel catastrophes annuelles
    disasters = clean_emdat_phl()        # colonnes : year, event_count, total_deaths, ...

    # Merge sur l'année
    full = macro.merge(disasters, on="year", how="left")

    # Si certaines années n’ont pas de désastre : mettre 0
    disaster_cols = [c for c in full.columns if c not in ["year", "poverty_rate", "gdp_growth"]]
    full[disaster_cols] = full[disaster_cols].fillna(0)

    out_path = PROCESSED_DIR / "phl_full_panel_2004_2024.csv"
    full.to_csv(out_path, index=False)

    return full


if __name__ == "__main__":
    panel = build_full_panel()
    print(panel.head())
    print("Shape:", panel.shape)
    print("Years:", panel["year"].min(), "-", panel["year"].max())
