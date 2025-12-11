import pandas as pd
from .config import DATA_DIR

# Where all processed CSVs live
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)


def build_regional_ml_dataset(
    target_group: str = "children",
) -> pd.DataFrame:
    """
    Build a regional panel ready for ML.

    Each row = (region, year) corresponding to the *base* survey year
    (e.g. 2018 -> change up to 2021).

    Columns (aim):
      - region, year
      - poverty_change_3y (target)
      - poverty_t0 (pre-disaster poverty)
      - gdp_growth, unemployment_rate (national macro)
      - typhoon_count, flood_count, earthquake_count
      - total_deaths, total_affected, total_damages
      - severe_event, haiyan_dummy
      - coastal (dummy)
    """

    # ---------- 1) Load regional poverty + disaster panel ----------
    reg_path = PROCESSED_DIR / "regional_poverty_disaster_panel.csv"
    df = pd.read_csv(reg_path)

    # This panel already has:
    #  region, year, poverty_women, poverty_children,
    #  next_year, years_to_next,
    #  poverty_women_next, poverty_children_next,
    #  d_pov_women_next, d_pov_children_next,
    #  typhoon_count, flood_count, earthquake_count,
    #  total_deaths, total_affected, total_damages,
    #  severe_event, haiyan_dummy

    # ---------- 2) Add national macro variables ----------
    macro_path = PROCESSED_DIR / "phl_full_panel_2004_2024.csv"
    macro = pd.read_csv(macro_path)[["year", "gdp_growth", "unemployment_rate"]]

    # merge on year (national macro = same for all regions that year)
    df = df.merge(macro, on="year", how="left", validate="many_to_one")

    # ---------- 3) Choose which group we model (women vs children) ----------
    if target_group.lower() == "children":
        pov_col = "poverty_children"
        d_pov_col = "d_pov_children_next"
    else:
        pov_col = "poverty_women"
        d_pov_col = "d_pov_women_next"

    df = df.copy()
    df.rename(
        columns={
            pov_col: "poverty_t0",           # pre-disaster poverty
            d_pov_col: "poverty_change_3y",  # Δpoverty over ~3 years
        },
        inplace=True,
    )

    # keep only rows where we actually observe the next survey
    df = df[df["poverty_change_3y"].notna()].reset_index(drop=True)

    # ---------- 4) Add coastal dummy ----------
    # Very simple rule: CAR is inland, all others coastal.
    INLAND_REGIONS = {"Cordillera Administrative Region (CAR)"}
    df["coastal"] = df["region"].apply(
        lambda r: 0 if r in INLAND_REGIONS else 1
    ).astype(int)

    # ---------- 5) Keep only columns useful for ML ----------
    cols = [
        "region", "year",
        "poverty_change_3y", "poverty_t0",
        "gdp_growth", "unemployment_rate",
        "typhoon_count", "flood_count", "earthquake_count",
        "total_deaths", "total_affected", "total_damages",
        "severe_event", "haiyan_dummy",
        "coastal",
    ]
    cols = [c for c in cols if c in df.columns]

    df_final = (
        df[cols]
        .sort_values(["region", "year"])
        .reset_index(drop=True)
    )

    # ---------- 6) Save ----------
    out_path = PROCESSED_DIR / "regional_ml_dataset.csv"
    df_final.to_csv(out_path, index=False)

    return df_final


if __name__ == "__main__":
    panel = build_regional_ml_dataset(target_group="children")
    print(panel.head())
    print("Shape:", panel.shape)
    print("Years:", panel["year"].min(), "-", panel["year"].max())
