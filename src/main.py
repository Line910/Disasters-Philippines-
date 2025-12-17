# src/main.py

import pandas as pd

from .config import PROCESSED_DATA_DIR
from .build_dataset import build_full_panel
from .outcome_dataset import build_national_outcome_panel
from .outcome_regional import build_regional_outcome_dataset
from .build_regional_dataset import build_regional_ml_dataset
from .modeling_regional import run_regional_models
from .plots import run_all_plots



# ---------------------------------------------------------------------
# STEP 1 – National macro + disaster panel
# ---------------------------------------------------------------------
def run_step_1_national_panel() -> pd.DataFrame:
    print("STEP 1 – National macro-disaster panel (Philippines)")
    print("-" * 70)

    nat = build_full_panel()          # writes phl_full_panel_2004_2024.csv

    print(nat.head(), "\n")
    print("Shape:", nat.shape)
    print("Years:", nat["year"].min(), "-", nat["year"].max())

    # simple correlations with poverty_rate
    corr_cols = [
        "gdp_growth",
        "unemployment_rate",
        "event_count",
        "total_deaths",
        "total_affected",
        "total_damages",
    ]
    corr_cols = [c for c in corr_cols if c in nat.columns]
    corr = nat[["poverty_rate"] + corr_cols].corr()["poverty_rate"]
    print("\n=== Correlations with poverty_rate ===")
    print(corr)
    print()

    return nat


# ---------------------------------------------------------------------
# STEP 2 – Regional poverty + disaster panel
# ---------------------------------------------------------------------
def run_step_2_regional_panel() -> pd.DataFrame:
    print("STEP 2 – Regional poverty + disaster panel")
    print("-" * 70)

    path = PROCESSED_DATA_DIR / "regional_poverty_disaster_panel.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.build_regional_panel` once "
            "to create the regional_poverty_disaster_panel.csv file."
        )

    reg = pd.read_csv(path)

    print(reg.head(), "\n")
    print("Shape:", reg.shape)
    print("Years:", reg["year"].min(), "-", reg["year"].max())
    print("Regions:", reg["region"].nunique())
    print()

    return reg



# ---------------------------------------------------------------------
# STEP 3 – National outcome dataset (Δ poverty over 3 years)
# ---------------------------------------------------------------------
def run_step_3_national_outcome(horizon: int = 3) -> pd.DataFrame:
    print(f"STEP 3 – National outcome dataset (Δ poverty over {horizon} years)")
    print("-" * 70)

    nat_out = build_national_outcome_panel(horizon=horizon)  # writes national_outcome_dataset.csv

    print(nat_out.head(), "\n")
    print("Shape:", nat_out.shape)
    print("Years:", nat_out["year"].min(), "-", nat_out["year"].max())
    print()

    return nat_out


# ---------------------------------------------------------------------
# STEP 4 – Regional outcome dataset (Δ poverty over 3 years)
# ---------------------------------------------------------------------
def run_step_4_regional_outcome(target_group: str = "children") -> pd.DataFrame:
    print(f"STEP 4 – Regional outcome dataset (Δ poverty over 3 years, group={target_group})")
    print("-" * 70)

    reg_out = build_regional_outcome_dataset(target_group=target_group)
    # this writes regional_outcome_dataset_<group>.csv

    print(reg_out.head(), "\n")
    print("Shape:", reg_out.shape)
    print("Years:", reg_out["year"].min(), "-", reg_out["year"].max())
    print("Regions:", reg_out["region"].nunique())
    print()

    return reg_out


# ---------------------------------------------------------------------
# STEP 5 – Regional ML dataset + models
# ---------------------------------------------------------------------
def run_step_5_regional_ml_and_models() -> None:
    print("STEP 5 – Regional ML datasets & models")
    print("-" * 70)

    # Build ML-ready datasets for both groups (files: regional_ml_dataset_<group>.csv)
    for group in ["women", "children"]:
        print(f"\n--- Building regional ML dataset for {group} ---")
        ml_df = build_regional_ml_dataset(target_group=group)
        print(ml_df.head(), "\n")
        print("Shape:", ml_df.shape)
        print("Years:", ml_df["year"].min(), "-", ml_df["year"].max())

        # Run models (this saves feature-importance plots for each group)
        print(f"\n--- Running models for {group} ---")
        run_regional_models(target_group=group)

    print()


# ---------------------------------------------------------------------
# STEP 6 – Visualisations (national + regional plots)
# ---------------------------------------------------------------------
def run_step_6_visualisations() -> None:
    print("STEP 6 – Visualisations")
    print("-" * 70)
    run_all_plots()
    print()


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def main() -> None:
    # 1–4 mostly descriptive
    run_step_1_national_panel()
    run_step_2_regional_panel()
    run_step_3_national_outcome(horizon=3)
    run_step_4_regional_outcome(target_group="children")

    # 5 – ML + feature importances
    run_step_5_regional_ml_and_models()

    # 6 – final plots for the report
    run_step_6_visualisations()

    print("=== DONE: full pipeline executed ===")


if __name__ == "__main__":
    main()
