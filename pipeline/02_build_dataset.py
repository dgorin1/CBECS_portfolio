# pipeline/02_build_dataset.py
from __future__ import annotations
import pathlib, yaml
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import json
from transformers import (
    TopCodeClipper, MassRecoder, MassDrop, OneHotEncoder, DropMissing, MedianImputer, CategorizeCols, AutoTransform
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
    ("categorize_cols", CategorizeCols(TYPES.get("categorical_variables"))),
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


# ############# Testing
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# # Assign X, y 
# X = df.drop('LOG_MFBTU', axis=1)
# y = df['LOG_MFBTU']

# # Split the data into validation, train, and test sets
# # e.g., 80/10/10 split
# test_size = 0.10
# val_size  = 0.10
# rand      = 1

# # 1) Hold out test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=test_size, random_state=rand, shuffle=True
# )

# # 2) From the remainder, carve out validation
# val_rel = val_size / (1.0 - test_size)
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=val_rel, random_state=rand, shuffle=True
#     )

# # Check and make sure the splits worked as expected...
# df.shape
# check_shape = X_train.shape[0]+X_test.shape[0]+X_val.shape[0] == df.shape[0]

# print(f"Split correctly: {check_shape}")


# scale_numeric_only = ColumnTransformer(
#     transformers=[("scale_num", StandardScaler(), num_cols)],
#     remainder="passthrough"
# )

# preproc = Pipeline(steps=[
#     ("median_imputer", MedianImputer(RULES["impute_rules"], num_cols)),
#     ("auto", AutoTransform(scoring="r2", cv=5, n_jobs=-1, num_cols=num_cols, suffix_identity=False)),
#     ("ohe", OneHotEncoder()),
    
# ])


# X_arr = preproc.fit_transform(X_train, y_train)

# breakpoint()