import pandas as pd

from .config import DATA_DIR
from .load_data import load_emdat

# Where we save cleaned files
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)


# ------------------ helpers ------------------ #

# Map any geographic text to a Philippines region name
REGION_KEYWORDS = {
    "NATIONAL CAPITAL": "National Capital Region (NCR)",
    "CORDILLERA": "Cordillera Administrative Region (CAR)",
    "ILOCOS": "Region I (Ilocos Region)",
    "CAGAYAN VALLEY": "Region II (Cagayan Valley)",
    "CENTRAL LUZON": "Region III (Central Luzon)",
    "CALABARZON": "Region IV-A (CALABARZON)",
    "MIMAROPA": "MIMAROPA Region",
    "BICOL": "Region V (Bicol Region)",
    "WESTERN VISAYAS": "Region VI (Western Visayas)",
    "CENTRAL VISAYAS": "Region VII (Central Visayas)",
    "EASTERN VISAYAS": "Region VIII Eastern Visayas",
    "ZAMBOANGA PENINSULA": "Region IX (Zamboanga Peninsula)",
    "NORTHERN MINDANAO": "Region X (Northern Mindanao)",
    "DAVAO": "Region XI (Davao Region)",
    "SOCCSKSARGEN": "Region XII (SOCCSKSARGEN)",
    "CARAGA": "Region XIII (Caraga)",
    "BANGSAMORO": "Bangsamoro Autonomous Region in Muslim Mindanao (BARMM)",
}


def _assign_region(emdat: pd.DataFrame) -> pd.Series:
    """
    Use Region / Location / Admin Units text to guess the PH administrative region.
    Very simple rule-based mapping using keyword search.
    """
    # Combine geographic text into one big string per row
    geo = (
        emdat[["Region", "Location", "Admin Units"]]
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.upper()
    )

    # Start with NA
    region = pd.Series(pd.NA, index=emdat.index, dtype="object")

    for key, reg_name in REGION_KEYWORDS.items():
        mask = geo.str.contains(key, na=False)
        region[mask] = reg_name

    return region


# ------------------ main builder ------------------ #

def build_regional_disaster_panel() -> pd.DataFrame:
    """
    Build panel of disaster exposure by (region, year) for the Philippines.

    Output columns:
        region, year,
        typhoon_count, flood_count, earthquake_count,
        total_deaths, total_affected, total_damages,
        severe_event, haiyan_dummy
    """
    emdat = load_emdat()

    # Philippines only, 2004–2024
    emdat = emdat[emdat["Country"] == "Philippines"].copy()
    emdat = emdat[(emdat["Start Year"] >= 2004) & (emdat["Start Year"] <= 2024)].copy()
    emdat["year"] = emdat["Start Year"].astype(int)

    # Assign regions
    emdat["region"] = _assign_region(emdat)
    emdat = emdat.dropna(subset=["region"]).copy()

    # Column names in your EM-DAT file
    death_col = "Total Deaths"
    affected_col = "Total Affected"
    damage_col = "Total Damage ('000 US$)"

    # Make sure numeric columns are really numbers
    for col in [death_col, affected_col, damage_col]:
        emdat[col] = (
            emdat[col]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        emdat[col] = pd.to_numeric(emdat[col], errors="coerce").fillna(0)

    # Hazard-type counts
    emdat["typhoon_flag"] = (emdat["Disaster Type"] == "Storm").astype(int)
    emdat["flood_flag"] = (emdat["Disaster Type"] == "Flood").astype(int)
    emdat["earthquake_flag"] = (emdat["Disaster Type"] == "Earthquake").astype(int)

    # Severe event: big deaths or big damages
    emdat["severe_event"] = (
        (emdat[death_col] >= 500) | (emdat[damage_col] >= 500_000)
    ).astype(int)

    # Haiyan dummy (2013 Typhoon Haiyan / Yolanda)
    emdat["haiyan_event"] = (
        emdat["Event Name"].str.contains("HAIYAN", case=False, na=False)
    ).astype(int)

    # Aggregate by region-year
    grouped = (
        emdat
        .groupby(["region", "year"], as_index=False)
        .agg(
            typhoon_count=("typhoon_flag", "sum"),
            flood_count=("flood_flag", "sum"),
            earthquake_count=("earthquake_flag", "sum"),
            total_deaths=(death_col, "sum"),
            total_affected=(affected_col, "sum"),
            total_damages=(damage_col, "sum"),
            severe_event=("severe_event", "max"),
            haiyan_dummy=("haiyan_event", "max"),
        )
    )

    grouped = grouped.sort_values(["region", "year"])

    # Save
    out_path = PROCESSED_DIR / "regional_disasters_phl.csv"
    grouped.to_csv(out_path, index=False)

    return grouped


if __name__ == "__main__":
    panel = build_regional_disaster_panel()
    print("=== Regional disaster panel (preview) ===")
    print(panel.head())
    print("Shape:", panel.shape)
    print("Years:", panel["year"].min(), "-", panel["year"].max())
    print("Regions:", sorted(panel["region"].unique()))
