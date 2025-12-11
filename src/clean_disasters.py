import pandas as pd

from .config import DATA_DIR, COUNTRY_CODE, YEAR_MIN, YEAR_MAX
from .load_data import load_emdat

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# DAPTE ces noms de colonnes si besoin en regardant emdat_2004_2025.csv
YEAR_COL = "Start Year"              
COUNTRY_COL = "Country"        
DEATHS_COL = "Total Deaths"    
AFFECTED_COL = "Total Affected"  
DAMAGES_COL = "Total Damage ('000 US$)"  

#----------EMDAT----------

def clean_emdat_phl():
    df = load_emdat()

    # Vérification rapide des colonnes
    print("Colonnes EMDAT:", list(df.columns))

    # 1) garder seulement Philippines
    if COUNTRY_COL not in df.columns:
        raise ValueError(f"Colonne pays '{COUNTRY_COL}' introuvable dans EMDAT.")

    df = df[df[COUNTRY_COL] == "Philippines"]

    # 2) gérer l'année
    if YEAR_COL not in df.columns:
        raise ValueError(f"Colonne année '{YEAR_COL}' introuvable dans EMDAT.")

    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df.dropna(subset=[YEAR_COL])
    df[YEAR_COL] = df[YEAR_COL].astype(int)

    # 3) filtrer la période 2004–2024 (config)
    df = df[(df[YEAR_COL] >= YEAR_MIN) & (df[YEAR_COL] <= YEAR_MAX)]

    # 4) convertir les colonnes numériques si elles existent
    for col in [DEATHS_COL, AFFECTED_COL, DAMAGES_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5) agrégation annuelle
    agg_dict = {"event_count": (YEAR_COL, "size")}

    if DEATHS_COL in df.columns:
        agg_dict["total_deaths"] = (DEATHS_COL, "sum")
    if AFFECTED_COL in df.columns:
        agg_dict["total_affected"] = (AFFECTED_COL, "sum")
    if DAMAGES_COL in df.columns:
        agg_dict["total_damages"] = (DAMAGES_COL, "sum")

    yearly = df.groupby(YEAR_COL).agg(**agg_dict).reset_index()
    yearly = yearly.rename(columns={YEAR_COL: "year"})

    # 6) sauvegarde
    out_path = PROCESSED_DIR / "disasters_phl_2004_2024.csv"
    yearly.to_csv(out_path, index=False)

    return yearly


if __name__ == "__main__":
    yearly = clean_emdat_phl()
    print(yearly.head())
    print(yearly["year"].min(), yearly["year"].max())
