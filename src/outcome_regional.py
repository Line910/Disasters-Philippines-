import pandas as pd

from .config import DATA_DIR
from .build_regional_panel import build_regional_full_panel

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)


def build_regional_outcome_dataset(target_group: str = "children") -> pd.DataFrame:
    """
    Build the regional outcome dataset.

    Uses the regional poverty + disaster panel (women + children) and keeps
    only the chosen target group, with:

      - poverty_t0: poverty at the survey year
      - d_poverty_next: change in poverty to the *next* survey
      - plus macro + disaster + coastal features.
    """

    # 1) Load full regional panel (already built in build_regional_panel.py)
    panel = build_regional_full_panel()

    # 2) Choose which poverty columns to use
    if target_group == "children":
        pov_t0_col = "poverty_children"
        d_pov_col = "d_pov_children_next"
    elif target_group == "women":
        pov_t0_col = "poverty_women"
        d_pov_col = "d_pov_women_next"
    else:
        raise ValueError("target_group must be 'children' or 'women'")

    # 3) Select columns we want to keep
    cols = [
        "region",
        "year",
        pov_t0_col,
        d_pov_col,
        "gdp_growth",
        "unemployment_rate",
        "typhoon_count",
        "flood_count",
        "earthquake_count",
        "total_deaths",
        "total_affected",
        "total_damages",
        "severe_event",
        "haiyan_dummy",
        "coastal",
    ]

    # Keep only columns that actually exist (safety)
    cols = [c for c in cols if c in panel.columns]

    outcome = panel[cols].rename(
        columns={
            pov_t0_col: "poverty_t0",
            d_pov_col: "d_poverty_next",
        }
    )

    # 4) Save to CSV
    out_path = PROCESSED_DIR / "outcome_regional_dataset.csv"
    outcome.to_csv(out_path, index=False)

    return outcome


if __name__ == "__main__":
    # Simple standalone run for checking
    outcome = build_regional_outcome_dataset(target_group="children")
    print(outcome.head(), "\n")
    print("Shape:", outcome.shape)
    print("Years:", outcome["year"].min(), "-", outcome["year"].max())
    print("Target group:", "children")
