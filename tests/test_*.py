import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------
# 1. Load raw data
# -------------------------

# Paths – adapte le chemin si besoin
emdat_path = "data EMDAT 2004.2025.xlsx"
wdi_macro_path = "P_Data_Extract_From_World_Development_Indicators-2.xlsx"
wdi_pov_path = "P_Data_Extract_From_World_Development_Indicators-3.xlsx"

emdat = pd.read_excel(emdat_path)
wdi_macro = pd.read_excel(wdi_macro_path)
wdi_pov = pd.read_excel(wdi_pov_path)

# -------------------------
# 2. Filter EM-DAT for Philippines 2000–2021
# (on s’arrête à 2021 pour avoir +3 ans de recul)
# -------------------------

emdat_ph = (
    emdat
    .loc[
        (emdat["ISO"] == "PHL") &
        (emdat["Start Year"].between(2000, 2021))
    ]
    .copy()
)

# Garder quelques colonnes pertinentes
emdat_ph = emdat_ph[[
    "DisNo.",
    "Event Name",
    "Disaster Group",
    "Disaster Subgroup",
    "Disaster Type",
    "Disaster Subtype",
    "ISO",
    "Country",
    "Start Year",
    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "No. Homeless",
    "Total Affected",
    "Total Damage ('000 US$)",
    "Magnitude",
    "Magnitude Scale"
]]

emdat_ph.rename(columns={"Start Year": "year_event"}, inplace=True)

# -------------------------
# 3. Reshape WDI macro data (GDP & GDP growth)
# -------------------------

def extract_wdi_series(df, series_code, new_name):
    """Extract one WDI series as a long df: year, value."""
    row = df.loc[df["Series Code"] == series_code]
    if row.empty:
        raise ValueError(f"Series code {series_code} not found in WDI file.")
    row = row.iloc[0]
    # Keep only year columns
    year_cols = [c for c in df.columns if "[YR" in c]
    out = row[year_cols].to_frame(name=new_name).reset_index()
    out["year"] = out["index"].str.extract(r"(\d{4})").astype(int)
    out.drop(columns=["index"], inplace=True)
    return out

# GDP current US$
gdp = extract_wdi_series(wdi_macro, "NY.GDP.MKTP.CD", "gdp_current_usd")
# GDP growth %
gdp_growth = extract_wdi_series(wdi_macro, "NY.GDP.MKTP.KD.ZG", "gdp_growth")

macro = (
    gdp.merge(gdp_growth, on="year", how="outer")
)

# -------------------------
# 4. Reshape WDI poverty series
# (national poverty headcount – attention à l’échelle, à vérifier)
# -------------------------

pov = extract_wdi_series(
    wdi_pov,
    "SI.POV.NAHC",    # Poverty headcount ratio at national poverty lines
    "poverty_national_raw"
)

# Fusion macro + poverty
macro = macro.merge(pov, on="year", how="outer")

# -------------------------
# 5. Build pre- and post-disaster macro indicators
#    pre = year_event - 1
#    post_3y = year_event + 3
# -------------------------

# Préparation des tables pré et post
macro_pre = macro.copy()
macro_pre.rename(columns={
    "year": "year_pre",
    "gdp_current_usd": "gdp_pre",
    "gdp_growth": "gdp_growth_pre",
    "poverty_national_raw": "poverty_pre"
}, inplace=True)

macro_post3 = macro.copy()
macro_post3.rename(columns={
    "year": "year_post3",
    "gdp_current_usd": "gdp_post3",
    "gdp_growth": "gdp_growth_post3",
    "poverty_national_raw": "poverty_post3"
}, inplace=True)

# Ajouter les années de référence aux désastres
emdat_ph["year_pre"] = emdat_ph["year_event"] - 1
emdat_ph["year_post3"] = emdat_ph["year_event"] + 3

# Merge avec macro pré et post
df = (
    emdat_ph
    .merge(macro_pre, on="year_pre", how="left")
    .merge(macro_post3, on="year_post3", how="left")
)

# -------------------------
# 6. Construire la target: Δ pauvreté à 3 ans
# -------------------------

df["poverty_change_3y"] = df["poverty_post3"] - df["poverty_pre"]

# On enlève les lignes où on n’a pas de target ou pas de pré
df_model = df.dropna(subset=["poverty_pre", "poverty_post3", "poverty_change_3y"]).copy()

print("Nombre d’événements utilisables :", len(df_model))

# -------------------------
# 7. Préparation X / y pour le modèle
# -------------------------

target = "poverty_change_3y"

numeric_features = [
    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "No. Homeless",
    "Total Affected",
    "Total Damage ('000 US$)",
    "Magnitude",
    "gdp_pre",
    "gdp_growth_pre",
    "poverty_pre"
]

categorical_features = [
    "Disaster Group",
    "Disaster Subgroup",
    "Disaster Type",
    "Disaster Subtype"
]

# On peut avoir des colonnes manquantes, on nettoie un minimum
df_model[numeric_features] = df_model[numeric_features].fillna(0)

X = df_model[numeric_features + categorical_features]
y = df_model[target]

# -------------------------
# 8. Pipeline scikit-learn
# -------------------------

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train / test split (par défaut aléatoire – tu pourras plus tard faire un split temporel)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("R² :", r2_score(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))

