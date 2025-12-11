import pandas as pd

from .config import DATA_DIR
from .clean_disasters_regional import build_regional_disaster_panel
from .load_data import (
    load_regional_poverty_women,
    load_regional_poverty_children,
)

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)


def build_regional_full_panel() -> pd.DataFrame:
    """
    Regional panel combining:
      - poverty among women & children
      - regional disaster exposure
    and computing *change in poverty* between survey waves.
    """

    # 1) Load regional poverty (already cleaned from the PovStat tables)
    #    Expected columns: region, year, poverty_women / poverty_children
    women = load_regional_poverty_women()
    children = load_regional_poverty_children()

    # Merge women + children poverty into one table
    pov = (
        women.merge(
            children,
            on=["region", "year"],
            how="outer",
            validate="one_to_one",
        )
        .sort_values(["region", "year"])
        .reset_index(drop=True)
    )

    # 2) Compute change in poverty to the *next* survey for each region
    #    (2018 → 2021, 2021 → 2023). Last year per region has NaN by design.
    g = pov.groupby("region", group_keys=False)

    pov["next_year"] = g["year"].shift(-1)
    pov["years_to_next"] = pov["next_year"] - pov["year"]

    pov["poverty_women_next"] = g["poverty_women"].shift(-1)
    pov["poverty_children_next"] = g["poverty_children"].shift(-1)

    pov["d_pov_women_next"] = pov["poverty_women_next"] - pov["poverty_women"]
    pov["d_pov_children_next"] = (
        pov["poverty_children_next"] - pov["poverty_children"]
    )

    # 3) Load regional disaster exposure (EM-DAT)
    #    Expected columns: region, year, typhoon_count, flood_count,
    #    earthquake_count, total_deaths, total_affected, total_damages,
    #    severe_event, haiyan_dummy, ...
    disasters = build_regional_disaster_panel()

    # 4) Merge poverty + disasters on (region, year)
    panel = pov.merge(
        disasters,
        on=["region", "year"],
        how="left",
        validate="one_to_one",
    )

    # Fill missing disaster exposure with 0 (no events that year/region)
    keep_cols = {
        "region",
        "year",
        "poverty_women",
        "poverty_children",
        "next_year",
        "years_to_next",
        "poverty_women_next",
        "poverty_children_next",
        "d_pov_women_next",
        "d_pov_children_next",
    }

    disaster_cols = [c for c in panel.columns if c not in keep_cols]
    panel[disaster_cols] = panel[disaster_cols].fillna(0)

    # 5) Save
    out_path = PROCESSED_DIR / "regional_poverty_disaster_panel.csv"
    panel.to_csv(out_path, index=False)

    return panel


if __name__ == "__main__":
    # build the full regional panel and save it
    panel = build_regional_full_panel()
    print(panel.head())
    print("Shape:", panel.shape)
    print("Years:", panel["year"].min(), "-", panel["year"].max())
    print("Columns:", list(panel.columns))
