# src/plots.py

import pandas as pd
import matplotlib.pyplot as plt

from .config import PROCESSED_DATA_DIR, RESULTS_DIR
from .build_regional_dataset import build_regional_ml_dataset

PROCESSED_DIR = PROCESSED_DATA_DIR


# ---------- Helpers ----------

def _load_national_panel() -> pd.DataFrame:
    """
    Load the national macro-poverty panel created in STEP 1.
    File: phl_full_panel_2004_2024.csv
    """
    path = PROCESSED_DIR / "phl_full_panel_2004_2024.csv"
    df = pd.read_csv(path)
    return df.sort_values("year")


def _load_regional_ml(group: str) -> pd.DataFrame:
    """
    Load regional ML dataset for a group (women / children).
    If the CSV doesn't exist yet, rebuild it.
    """
    tg = group.lower()
    path = PROCESSED_DIR / f"regional_ml_dataset_{tg}.csv"
    if path.exists():
        return pd.read_csv(path)
    # fallback: rebuild
    return build_regional_ml_dataset(target_group=group)


# ---------- Plot 1: national poverty trend ----------

def plot_national_poverty_trend() -> None:
    df = _load_national_panel()

    plt.figure(figsize=(8, 4))
    plt.plot(df["year"], df["poverty_rate"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Poverty rate (%)")
    plt.title("National poverty rate over time – Philippines")
    plt.grid(True, axis="y", alpha=0.3)

    out_path = RESULTS_DIR / "national_poverty_trend.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


# ---------- Plot 2: scatter – damages vs poverty change ----------

def plot_regional_scatter(group: str) -> None:
    """
    For a given group (women / children):
      x-axis: total_damages
      y-axis: poverty_change_3y
      color: coastal vs inland
    """
    df = _load_regional_ml(group)

    # safety: drop rows without needed columns
    df = df.dropna(subset=["total_damages", "poverty_change_3y", "coastal"])

    plt.figure(figsize=(6, 5))
    for coastal_value, label in [(1, "Coastal"), (0, "Inland")]:
        m = df["coastal"] == coastal_value
        plt.scatter(
            df.loc[m, "total_damages"],
            df.loc[m, "poverty_change_3y"],
            label=label,
            alpha=0.8,
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Total damages (000 USD)")
    plt.ylabel("Poverty change over 3 years (percentage points)")
    plt.title(f"Disaster damages vs poverty change – {group}")
    plt.legend()

    out_path = RESULTS_DIR / f"scatter_{group}_total_damages_vs_poverty_change_3y.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


# ---------- Plot 3: bar chart – poverty change by region ----------

def plot_regional_bar(group: str) -> None:
    """
    Bar chart by region:
      x: region
      y: poverty_change_3y
      color: coastal vs inland
    """
    df = _load_regional_ml(group)

    df = df.sort_values("poverty_change_3y")
    colors = df["coastal"].map({1: "tab:blue", 0: "tab:orange"})

    plt.figure(figsize=(10, 4))
    plt.bar(df["region"], df["poverty_change_3y"], color=colors)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xticks(rotation=90)
    plt.ylabel("Poverty change over 3 years (percentage points)")
    plt.title(f"Poverty change by region – {group}")

    out_path = RESULTS_DIR / f"bar_poverty_change_by_region_{group}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


# ---------- Master function called from main.py ----------

def run_all_plots() -> None:
    print("=== PLOTS: national trend ===")
    plot_national_poverty_trend()

    print("\n=== PLOTS: regional (women) ===")
    plot_regional_scatter("women")
    plot_regional_bar("women")

    print("\n=== PLOTS: regional (children) ===")
    plot_regional_scatter("children")
    plot_regional_bar("children")


if __name__ == "__main__":
    run_all_plots()
