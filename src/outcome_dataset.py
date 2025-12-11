import pandas as pd
from .config import DATA_DIR

PROCESSED_DIR = DATA_DIR / "processed"


def build_national_outcome_panel(horizon: int = 3) -> pd.DataFrame:
    """
    Build a dataset where each row is a year t and the target is the
    change in poverty_rate over the next `horizon` years.

    Target: d_poverty = poverty_rate_{t+horizon} - poverty_rate_t
    Features: disaster intensity + pre-disaster macro conditions.
    """
    path = PROCESSED_DIR / "phl_full_panel_2004_2024.csv"
    df = pd.read_csv(path).sort_values("year").reset_index(drop=True)

    # ---- Target: future poverty change ----
    df["poverty_future"] = df["poverty_rate"].shift(-horizon)
    df["d_poverty"] = df["poverty_future"] - df["poverty_rate"]

    # Drop years too close to the end (no future poverty)
    out = df.dropna(subset=["poverty_future"]).copy()

    # ---- Pre-disaster vulnerability (at year t) ----
    out["poverty_pre"] = out["poverty_rate"]
    out["unemp_pre"] = out["unemployment_rate"]
    out["gdp_growth_pre"] = out["gdp_growth"]

    # ---- Disaster intensity (year t) ----
    # You can later add logs, squared terms, etc.
    out["events"] = out["event_count"]
    out["deaths"] = out["total_deaths"]
    out["affected"] = out["total_affected"]
    out["damages"] = out["total_damages"]

    # Big-disaster dummy: top 25% of years by deaths
    death_threshold = out["deaths"].quantile(0.75)
    out["big_disaster"] = (out["deaths"] >= death_threshold).astype(int)

    # Keep only useful columns
    cols = [
        "year",
        "d_poverty",
        "poverty_pre",
        "unemp_pre",
        "gdp_growth_pre",
        "events",
        "deaths",
        "affected",
        "damages",
        "big_disaster",
    ]
    out = out[cols]

    # Save
    out_path = PROCESSED_DIR / "phl_outcome_panel_national.csv"
    out.to_csv(out_path, index=False)

    return out


if __name__ == "__main__":
    panel = build_national_outcome_panel(horizon=3)
    print(panel.head())
    print("Shape:", panel.shape)
    print("Years:", panel["year"].min(), "-", panel["year"].max())
