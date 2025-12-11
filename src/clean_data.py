import pandas as pd
from .config import DATA_DIR, COUNTRY_CODE, YEAR_MIN, YEAR_MAX
from .load_data import load_gdp_worldbank, load_poverty, load_unemployment


PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# ----------GDP World Bank----------
def clean_gdp_worldbank():
    df = load_gdp_worldbank()

    # garder seulement Philippines
    df = df[df["Country Code"] == COUNTRY_CODE]

    year_cols = [c for c in df.columns if str(c).isdigit()]

    tidy = df.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="year",
        value_name="gdp_growth",
    )

    tidy["year"] = tidy["year"].astype(int)
    tidy["gdp_growth"] = pd.to_numeric(tidy["gdp_growth"], errors="coerce")
    tidy = tidy.dropna(subset=["gdp_growth"])

    # filtrer années
    tidy = tidy[(tidy["year"] >= YEAR_MIN) & (tidy["year"] <= YEAR_MAX)]

    tidy.to_csv(PROCESSED_DIR / "gdp_growth_phl_2004_2024.csv", index=False)
    return tidy

#----------Poverty WB----------
def clean_poverty_wb():
    df = load_poverty()

    # garder seulement Philippines
    df = df[df["Country Code"] == COUNTRY_CODE]

    year_cols = [c for c in df.columns if str(c).isdigit()]

    tidy = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name"],
        value_vars=year_cols,
        var_name="year",
        value_name="poverty_rate",
    )

    tidy["year"] = tidy["year"].astype(int)
    tidy["poverty_rate"] = pd.to_numeric(tidy["poverty_rate"], errors="coerce")
    tidy = tidy.dropna(subset=["poverty_rate"])

    # filtrer années
    tidy = tidy[(tidy["year"] >= YEAR_MIN) & (tidy["year"] <= YEAR_MAX)]

    tidy.to_csv(PROCESSED_DIR / "poverty_rate_phl_2004_2024.csv", index=False)
    return tidy

#----------Unemployment WB----------
def clean_unemployment_wb():
    df = load_unemployment()

    # 1) ne garder que Philippines
    df = df[df["Country Code"] == COUNTRY_CODE]

    # 2) colonnes d'années
    year_cols = [c for c in df.columns if str(c).isdigit()]

    tidy = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name"],
        value_vars=year_cols,
        var_name="year",
        value_name="unemployment_rate",
    )

    tidy["year"] = tidy["year"].astype(int)
    tidy["unemployment_rate"] = pd.to_numeric(
        tidy["unemployment_rate"], errors="coerce"
    )
    tidy = tidy.dropna(subset=["unemployment_rate"])

    # 3) filtrer sur 2004–2024 (config.py)
    tidy = tidy[(tidy["year"] >= YEAR_MIN) & (tidy["year"] <= YEAR_MAX)]

    tidy.to_csv(PROCESSED_DIR / "unemployment_rate_phl_2004_2024.csv", index=False)
    return tidy

#----------Build Macro Panel----------
def build_macro_panel():
    gdp = clean_gdp_worldbank()[["year", "gdp_growth"]]
    pov = clean_poverty_wb()[["year", "poverty_rate"]]
    unemp = clean_unemployment_wb()[["year", "unemployment_rate"]]

    panel = pov.merge(gdp, on="year", how="inner")
    panel = panel.merge(unemp, on="year", how="left")

    panel.to_csv(PROCESSED_DIR / "phl_macro_panel_2004_2024.csv", index=False)
    return panel


if __name__ == "__main__":
    panel = build_macro_panel()
    print(panel.head())
    print(panel["year"].min(), panel["year"].max())
