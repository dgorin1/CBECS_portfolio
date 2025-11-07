# pipeline/02_build_dataset.py
from __future__ import annotations
import pathlib, yaml
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import json
from transformers import (
    TopCodeClipper, MassRecoder, MassDrop, OneHotEncoder, DropMissing
)

# ---------- Paths & config ----------
PROJ = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(PROJ/"conf"/"config.yaml"))
TYPES = yaml.safe_load(open(PROJ/"conf"/"variable_types.yaml"))
RULES = yaml.safe_load(open(PROJ/"conf"/"cleaning_rules.yaml"))

RAW_DIR = PROJ/CFG["data"]["raw_dir"]
PROC_DIR = PROJ/CFG["data"]["processed_dir"]
PROC_DIR.mkdir(parents=True, exist_ok=True)

year = CFG["datasets"]["year"]
target_col = CFG["build"]["target"]  # 'MFBTU' expected

print(f"Loading raw: {year}")
df = pd.read_csv(PROJ / "data" / "raw" / f"cbecs_{year}_microdata.csv")

# ---------- Column groups from config ----------
cat_cols = [c for c in TYPES["categorical_variables"] if c in df.columns]
num_cols = [c for c in TYPES["numeric_variables"] if c in df.columns]

# ---------- Target engineering ( ----------
df["LOG_MFBTU"] = np.log(df[target_col])
df.drop(columns=[target_col], inplace=True)

# ---------- Pre-clean block (deterministic ops only) ----------
preclean = Pipeline(steps=[
    ("drop_missing_cols", DropMissing(RULES.get("missing_thresh"), "LOG_MFBTU")),
    ("mass_drop", MassDrop(RULES.get("drop_columns"), RULES.get("regex_for_delete"))),
    ("mass_recodes", MassRecoder(RULES.get("recode_rules"))),
    ("top_codes", TopCodeClipper(RULES.get("top_codes"))),
])

df_clean = preclean.fit_transform(df)



# ---------- Persist canonical dataset (parquet) ----------
clean_path = PROC_DIR / f"cbecs_{year}_clean.parquet"
df_clean.to_parquet(clean_path, index=False)
print(f"Saved cleaned dataset (parquet): {clean_path}")

# ---------- Persist column schema (JSON) ----------
schema_path = PROC_DIR / f"cbecs_{year}_schema.json"
schema = {
    "columns": list(df_clean.columns),
    "dtypes": {col: str(dtype) for col, dtype in df_clean.dtypes.items()}
}
with open(schema_path, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)
print(f"Saved schema: {schema_path}")




