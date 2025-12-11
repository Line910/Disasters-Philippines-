from pathlib import Path
import re

import pandas as pd

from .config import DATA_DIR, YEAR_MIN, YEAR_MAX

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)


def _read_regional_semicolon_table(filename: str, value_name: str) -> pd.DataFrame:
    """
    Read a regional poverty CSV that looks like:

        ;;;;;;;;;;
        ;;;;;;;;;;
        Region;2018;2021;2023;;;;;;;
        PHILIPPINES;23,9;26,3;23,4;...
        National Capital Region (NCR);4,3;6,3;3,6;...
        ...

    and return a long DataFrame: region, year, <value_name>.
    """
    path = DATA_DIR / filename

    # read all lines
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    # find the header line starting with "Region;"
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Region;"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"No 'Region;' header found in {filename}")

    header = lines[header_idx].split(";")

    # find year columns in the header (2018, 2021, 2023, ...)
    year_positions: list[int] = []
    year_values: list[int] = []

    for j, token in enumerate(header):
        token = token.strip()
        if re.fullmatch(r"(19|20)\d{2}", token):
            year_positions.append(j)
            year_values.append(int(token))

    if not year_positions:
        raise ValueError(f"No year columns found in header of {filename}: {header}")

    rows = []
    data_start = header_idx + 1

    for line in lines[data_start:]:
        # stop on empty line / only semicolons
        if not line.strip("; "):
            break

        parts = line.split(";")
        if len(parts) == 0:
            continue

        region = parts[0].strip()
        if not region:
            continue

        # skip national total if you only want regions
        # (enlève ce if si tu veux garder "PHILIPPINES")
        if region.upper().startswith("PHILIPPINES"):
            continue

        for pos, year in zip(year_positions, year_values):
            if pos >= len(parts):
                continue
            value = parts[pos].strip()
            if not value:
                continue

            # convert "23,4" -> 23.4
            value = value.replace(",", ".")
            try:
                val_float = float(value)
            except ValueError:
                continue

            rows.append(
                {
                    "region": region,
                    "year": year,
                    value_name: val_float,
                }
            )

    df = pd.DataFrame(rows)
    df["year"] = df["year"].astype(int)
    return df


def build_regional_poverty_panel() -> pd.DataFrame:
    """
    Build regional poverty panel combining:
    - poverty among women
    - poverty among children

    Output columns:
        region, year, poverty_women, poverty_children
    """

    # adapte ces noms si tes fichiers s'appellent un peu différemment
    women = _read_regional_semicolon_table(
        "regional_poverty_women.csv", "poverty_women"
    )
    children = _read_regional_semicolon_table(
        "regional_poverty_childran.csv", "poverty_children"
    )

    panel = pd.merge(
        women,
        children,
        on=["region", "year"],
        how="outer",
    ).sort_values(["region", "year"])

    # garde seulement les années dans le range global
    panel = panel[(panel["year"] >= YEAR_MIN) & (panel["year"] <= YEAR_MAX)]

    out_path = PROCESSED_DIR / "regional_poverty_phl.csv"
    panel.to_csv(out_path, index=False)

    return panel


if __name__ == "__main__":
    panel = build_regional_poverty_panel()

    # split and save
    women = panel[["region", "year", "poverty_women"]]
    children = panel[["region", "year", "poverty_children"]]

    out_w = DATA_DIR / "processed" / "regional_poverty_women.csv"
    out_c = DATA_DIR / "processed" / "regional_poverty_children.csv"

    women.to_csv(out_w, index=False)
    children.to_csv(out_c, index=False)
