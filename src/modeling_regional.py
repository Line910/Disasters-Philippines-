# src/modeling_regional.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import PROCESSED_DATA_DIR, RESULTS_DIR
from .build_regional_dataset import build_regional_ml_dataset

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

PROCESSED_DIR = PROCESSED_DATA_DIR


def _save_feature_importances(
    importances,
    feature_cols,
    model_name: str,
    target_group: str,
) -> None:
    """
    Save a horizontal bar plot of feature importances / coefficients
    into results/.
    """
    s = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    s.plot(kind="barh")
    plt.xlabel("Importance")
    plt.title(f"{model_name} – {target_group}")
    plt.tight_layout()

    out_path = RESULTS_DIR / f"feature_importances_{target_group}_{model_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


def load_or_build_dataset(target_group: str = "children") -> pd.DataFrame:
    """
    Load regional_ml_dataset.csv si dispo, sinon le reconstruit.
    """
    path = PROCESSED_DIR / "regional_ml_dataset.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = build_regional_ml_dataset(target_group=target_group)
    return df


def run_regional_models(target_group: str = "children") -> None:
    print(f"=== Regional modeling (target group: {target_group}) ===")

    df = load_or_build_dataset(target_group=target_group)

    print("Dataset shape:", df.shape)
    print("Years in dataset:", sorted(df["year"].unique()))
    print(df.head(), "\n")

    # -------- Target & features --------
    y_col = "poverty_change_3y"

    feature_cols = [
        # pré-vulnérabilité
        "poverty_t0",
        "gdp_growth",
        "unemployment_rate",
        # intensité des désastres
        "typhoon_count",
        "flood_count",
        "earthquake_count",
        "total_deaths",
        "total_affected",
        "total_damages",
        "severe_event",
        "haiyan_dummy",
        # géographie
        "coastal",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[y_col]

    # -------- Train / test split (2018 = train, 2021 = test) --------
    train_years = [2018]
    test_years = [2021]

    train_mask = df["year"].isin(train_years)
    test_mask = df["year"].isin(test_years)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")

    # -------- Helper to evaluate & print results --------
    def evaluate_model(name, model):
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        mae_train = mean_absolute_error(y_train, pred_train)
        mae_test = mean_absolute_error(y_test, pred_test)
        r2_train = r2_score(y_train, pred_train)
        r2_test = r2_score(y_test, pred_test)

        print(f"--- {name} ---")
        print(f"MAE train: {mae_train:.3f}, MAE test: {mae_test:.3f}")
        print(f"R^2  train: {r2_train:.3f}, R^2  test: {r2_test:.3f}")

        # Coefficients / importances
        if hasattr(model, "coef_"):
            coef = pd.Series(model.coef_, index=feature_cols)
            print("\nCoefficients (sorted):")
            print(coef.sort_values(ascending=False))
            # on sauvegarde le graphe avec les |coefficients|
            _save_feature_importances(
                np.abs(model.coef_),
                feature_cols,
                f"{name}_abscoef",
                target_group,
            )
        elif hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_cols)
            print("\nFeature importances (sorted):")
            print(imp.sort_values(ascending=False))
            _save_feature_importances(
                model.feature_importances_,
                feature_cols,
                name,
                target_group,
            )
        print()

    # -------- 3 models --------
    lin = LinearRegression()
    rf = RandomForestRegressor(
        n_estimators=200, random_state=0, min_samples_leaf=2
    )
    gb = GradientBoostingRegressor(random_state=0)

    evaluate_model("LinearRegression", lin)
    evaluate_model("RandomForestRegressor", rf)
    evaluate_model("GradientBoostingRegressor", gb)


if __name__ == "__main__":
    run_regional_models(target_group="children")
