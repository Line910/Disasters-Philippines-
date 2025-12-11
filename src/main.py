import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .config import DATA_DIR
from .build_dataset import build_full_panel
from .build_regional_panel import build_regional_full_panel
from .outcome_dataset import build_national_outcome_panel
from .outcome_regional import build_regional_outcome_dataset

PROCESSED_DIR = DATA_DIR / "processed"


# =====================================================================
# STEP 1 – National macro + disaster panel
# =====================================================================
def run_step_1_macro_panel() -> pd.DataFrame:
    print("\nSTEP 1 – National macro–disaster panel (Philippines)")
    print("=" * 70)

    macro = build_full_panel()  # writes phl_full_panel_2004_2024.csv

    print(macro.head(), "\n")
    print("Shape:", macro.shape)
    print("Years:", macro["year"].min(), "-", macro["year"].max())

    # Simple correlations with poverty_rate
    print("\n=== Correlations with poverty_rate ===")
    corr_cols = [
        "gdp_growth",
        "unemployment_rate",
        "event_count",
        "total_deaths",
        "total_affected",
        "total_damages",
    ]
    corr_cols = [c for c in corr_cols if c in macro.columns]
    corr = macro[["poverty_rate"] + corr_cols].corr()["poverty_rate"]
    print(corr)

    return macro


# =====================================================================
# STEP 2 – Regional poverty + disaster panel
# =====================================================================
def run_step_2_regional_panel() -> pd.DataFrame:
    print("\nSTEP 2 – Regional poverty + disaster panel")
    print("=" * 70)

    panel = build_regional_full_panel()  # writes regional_poverty_disaster_panel.csv

    print(panel.head(), "\n")
    print("Shape:", panel.shape)
    print("Years:", panel["year"].min(), "-", panel["year"].max())
    print("Regions:", panel["region"].nunique())

    # Quick summary of average disaster exposure by region
    disaster_cols = [
        "typhoon_count",
        "flood_count",
        "earthquake_count",
        "total_deaths",
        "total_damages",
    ]
    disaster_cols = [c for c in disaster_cols if c in panel.columns]
    if disaster_cols:
        print("\n--- Average disaster exposure by region (preview) ---")
        avg = (
            panel.groupby("region", as_index=False)[disaster_cols]
            .mean()
            .round(3)
        )
        print(avg.head())

    return panel


# =====================================================================
# STEP 3 – National outcome dataset (Δ poverty over horizon)
# =====================================================================
def run_step_3_national_outcome(horizon: int = 3) -> pd.DataFrame:
    print(f"\nSTEP 3 – National outcome dataset (Δ poverty over {horizon} years)")
    print("=" * 70)

    nat = build_national_outcome_panel(horizon=horizon)

    print(nat.head(), "\n")
    print("Shape:", nat.shape)
    print("Years:", nat["year"].min(), "-", nat["year"].max())

    return nat


# =====================================================================
# STEP 4 – Regional outcome dataset (Δ poverty over 3 years)
# =====================================================================
def run_step_4_regional_outcome(target_group: str = "children") -> pd.DataFrame:
    print(
        f"\nSTEP 4 – Regional outcome dataset (Δ poverty over 3 years, group={target_group})"
    )
    print("=" * 70)

    # Call WITHOUT keyword, so it works even if the parameter is named differently
    reg = build_regional_outcome_dataset(target_group)

    print(reg.head(), "\n")
    print("Shape:", reg.shape)
    print("Years:", reg["year"].min(), "-", reg["year"].max())
    print("Regions:", reg["region"].nunique())

    return reg


# ===============================================================
# STEP 5 – Regional ML dataset & models
# ===============================================================
def run_step_5_regional_models(reg_outcome: pd.DataFrame,
                               target_group: str = "children") -> None:
    """
    Train simple ML models on the regional outcome dataset.

    reg_outcome is the DataFrame returned by build_regional_outcome_dataset()
    and has columns like:
        region, year, poverty_t0, d_poverty_next,
        gdp_growth, unemployment_rate,
        typhoon_count, flood_count, earthquake_count,
        total_deaths, total_affected, total_damages,
        severe_event, haiyan_dummy, coastal
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    print(f"\nSTEP 5 – Regional ML dataset & models (group={target_group})")
    print("=" * 72)

    df = reg_outcome.copy()

    # ---- Target column (always the same now) ----
    target_col = "d_poverty_next"
    if target_col not in df.columns:
        raise KeyError(
            f"Expected column '{target_col}' in regional outcome dataset, "
            f"but got columns: {list(df.columns)}"
        )

    # ---- Feature columns: everything except region, year, target ----
    exclude = {"region", "year", target_col}
    feature_cols = [c for c in df.columns if c not in exclude]

    # Drop rows with missing values in features or target
    df = df.dropna(subset=feature_cols + [target_col])

    print("Dataset shape:", df.shape)
    print("Years in dataset:", df["year"].min(), "-", df["year"].max())
    print("Features used:", feature_cols)

    if df.shape[0] < 5:
        print("Not enough rows to train/test models (need at least ~5). Skipping Step 5.")
        return

    X = df[feature_cols].values
    y = df[target_col].values

    # --------------------------------------------------
    # Time-based train/test split that always has data
    # --------------------------------------------------
    years = sorted(df["year"].unique())
    if len(years) >= 2:
        # Train on all but the last year, test on the last year
        test_year = years[-1]
        train_mask = df["year"] < test_year
        test_mask = df["year"] == test_year
    else:
        # Fallback: if somehow there is only one year, use random split
        train_mask, test_mask = train_test_split(
            np.arange(len(df)), test_size=0.3, random_state=42
        )
        train_mask = df.index.isin(train_mask)
        test_mask = df.index.isin(test_mask)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Train or test set ended up empty after splitting. "
              "Skipping models to avoid crashing.")
        print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])
        return

    print("Train size:", X_train.shape[0], ", Test size:", X_test.shape[0])

    def evaluate_model(name, model):
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        print(f"\n--- {name} ---")
        print(f"MAE train: {mae_train:.3f}, MAE test: {mae_test:.3f}")
        print(f"R² train:  {r2_train:.3f}, R² test:  {r2_test:.3f}")

        # Linear regression: show coefficients
        if hasattr(model, "coef_"):
            coefs = pd.Series(model.coef_, index=feature_cols).sort_values(
                ascending=False
            )
            print("\nCoefficients (sorted):")
            print(coefs)

        # Tree-based models: feature importances
        if hasattr(model, "feature_importances_"):
            imps = pd.Series(
                model.feature_importances_, index=feature_cols
            ).sort_values(ascending=False)
            print("\nFeature importances (sorted):")
            print(imps)

    # ---- Models ----
    evaluate_model("LinearRegression", LinearRegression())

    evaluate_model(
        "RandomForestRegressor",
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
        ),
    )

    evaluate_model(
        "GradientBoostingRegressor",
        GradientBoostingRegressor(
            random_state=42,
        ),
    )


    # -----------------------------------------------------------------
    # Helper to train + print metrics
    # -----------------------------------------------------------------
    def evaluate_model(name, model):
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        print(f"MAE train: {mae_train:.3f}, MAE test: {mae_test:.3f}")
        print(f"R² train: {r2_train:.3f}, R² test: {r2_test:.3f}")

        return model

    # ---- Linear Regression ----
    lin = evaluate_model("LinearRegression", LinearRegression())
    coef = pd.Series(lin.coef_, index=feature_cols).sort_values(ascending=False)
    print("\nCoefficients (sorted):")
    print(coef)

    # ---- Random Forest ----
    rf = evaluate_model("RandomForestRegressor", RandomForestRegressor(
        n_estimators=500, random_state=42
    ))
    rf_importance = pd.Series(
        rf.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\nFeature importances (sorted):")
    print(rf_importance)

    # ---- Gradient Boosting ----
    gb = evaluate_model("GradientBoostingRegressor", GradientBoostingRegressor(
        random_state=42
    ))
    gb_importance = pd.Series(
        gb.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\nFeature importances (sorted):")
    print(gb_importance)


# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    # You can comment out steps you don’t want to re-run each time.
    run_step_1_macro_panel()
    run_step_2_regional_panel()
    run_step_3_national_outcome(horizon=3)
    reg_outcome = run_step_4_regional_outcome(target_group="children")
    run_step_5_regional_models(reg_outcome, target_group="children")


if __name__ == "__main__":
    main()



