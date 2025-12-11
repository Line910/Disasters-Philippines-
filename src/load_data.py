from io import StringIO
import pandas as pd
from .config import DATA_DIR

PROCESSED_DIR = DATA_DIR / "processed"


def read_csv_any(filename: str) -> pd.DataFrame:
    """
    Generic CSV reader:
    - lets pandas guess the separator
    - skips bad lines instead of crashing
    """
    path = DATA_DIR / filename
    return pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")


# ---------- RAW FILE LOADERS ----------


def load_emdat() -> pd.DataFrame:
    """
    Load the EM-DAT file (semicolon-separated, with BOM).
    """
    path = DATA_DIR / "emdat_2004_2025.csv"
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    return df


def load_gdp_worldbank() -> pd.DataFrame:
    """
    Load World Bank GDP growth file (standard WDI CSV).
    """
    path = DATA_DIR / "API_NY.GDP.MKTP.KD.ZG_DS2_fr_csv_v2_6072.csv"
    # WDI CSVs usually have 4 metadata rows before the header
    return pd.read_csv(path, skiprows=4)


def load_gdp_philippines() -> pd.DataFrame:
    """
    Load your custom Philippines GDP file (converted from Numbers).
    """
    return read_csv_any("gdp_philippines.csv")


# ---------- HELPER FOR WORLD BANK ';' FORMAT ----------


def _load_worldbank_semicolon_csv(filename: str) -> pd.DataFrame:
    """
    Load a World Bank CSV exported with ';' as separator and a metadata block
    at the top (poverty, unemployment, etc.).
    """
    path = DATA_DIR / filename

    # Read raw text
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the first line that starts with "Country Name;"
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Country Name;") or line.startswith('"Country Name";'):
            header_idx = i
            break

    # Fallback: if not found, let pandas guess
    if header_idx is None:
        return pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")

    # Keep only the data part and parse as a normal ';'-separated CSV
    csv_text = "".join(lines[header_idx:])
    df = pd.read_csv(StringIO(csv_text), sep=";")

    return df


def load_poverty() -> pd.DataFrame:
    """
    Load World Bank poverty rate file for all countries.
    """
    return _load_worldbank_semicolon_csv("poverty_rate_philippines.csv")


def load_unemployment() -> pd.DataFrame:
    """
    Load World Bank unemployment rate file for all countries.
    """
    return _load_worldbank_semicolon_csv("unemployment_rate.csv")


def load_sae() -> pd.DataFrame:
    """
    Load SAE 2018–2021 file (regional poverty).
    """
    return read_csv_any("2018 - 2021 SAE_with PSGC_noHUC.csv")

def load_regional_poverty_women() -> pd.DataFrame:
    path = PROCESSED_DIR / "regional_poverty_women.csv"
    df = pd.read_csv(path)

    # standardise columns
    df = df.rename(columns={
        "poverty_rate": "poverty_women",     # or whatever the column is called
    })
    return df[["region", "year", "poverty_women"]]


def load_regional_poverty_children() -> pd.DataFrame:
    path = PROCESSED_DIR / "regional_poverty_children.csv"
    df = pd.read_csv(path)

    df = df.rename(columns={
        "poverty_rate": "poverty_children",  # or the real column name
    })
    return df[["region", "year", "poverty_children"]]
def load_regional_poverty_panel() -> pd.DataFrame:
    """
    Merge women + children into one regional panel:
    region, year, poverty_women, poverty_children
    """
    women = load_regional_poverty_women()
    children = load_regional_poverty_children()

    panel = pd.merge(
        women,
        children,
        on=["region", "year"],
        how="outer",
    ).sort_values(["region", "year"])

    return panel


# ---------- QUICK MANUAL TEST ----------

if __name__ == "__main__":
    print("=== EMDAT ===")
    print(load_emdat().head(), "\n")

    print("=== GDP World Bank ===")
    print(load_gdp_worldbank().head(), "\n")

    print("=== GDP Philippines (custom) ===")
    print(load_gdp_philippines().head(), "\n")

    print("=== Poverty (World Bank) ===")
    print(load_poverty().head(), "\n")

    print("=== Unemployment (World Bank) ===")
    print(load_unemployment().head(), "\n")

    print("=== SAE 2018–2021 ===")
    print(load_sae().head(), "\n")

    print("=== Regional poverty panel (women & children) ===")
    print(load_regional_poverty_panel().head(), "\n")
