# pipeline/02_build_dataset.py
from __future__ import annotations
import pathlib, yaml
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import json
from transformers import (
    TopCodeClipper, MassRecoder, MassDrop, DropMissing,  CategorizeCols, AutoTransform
)

from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV, ElasticNetCV
import numpy as np
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.base import clone



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



# ---- at the very top of the file ----
import logging, warnings, numpy as np
from sklearn import set_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
# Keep arrays as DataFrames after transforms so we don't lose column names
set_config(transform_output="pandas")

# Raise on nasty numeric surprises so you get a traceback to the exact step
np.seterr(all="raise")
warnings.filterwarnings("error", category=RuntimeWarning)  # make overflow warnings crash
# print(f"Loading raw: {year}")
# df = pd.read_csv(PROJ / "data" / "raw" / f"cbecs_{year}_microdata.csv")

# # ---------- Column groups from config ----------
# cat_cols = [c for c in TYPES["categorical_variables"] if c in df.columns]
# num_cols = [c for c in TYPES["numeric_variables"] if c in df.columns]

# # ---------- Target engineering ( ----------
# df["LOG_MFBTU"] = np.log(df[target_col])
# df.drop(columns=[target_col], inplace=True)

# # ---------- Pre-clean block (deterministic ops only) ----------
# preclean = Pipeline(steps=[
#     ("categorize_cols", CategorizeCols(TYPES.get("categorical_variables"))),
#     ("drop_missing_cols", DropMissing(RULES.get("missing_thresh"), "LOG_MFBTU")),
#     ("mass_drop", MassDrop(RULES.get("drop_columns"), RULES.get("regex_for_delete"))),
#     ("mass_recodes", MassRecoder(RULES.get("recode_rules"))),
#     ("top_codes", TopCodeClipper(RULES.get("top_codes"))),
# ])

# df_clean = preclean.fit_transform(df)

# # ---------- Persist canonical dataset (parquet) ----------
# clean_path = PROC_DIR / f"cbecs_{year}_clean.parquet"
# df_clean.to_parquet(clean_path, index=False)
# print(f"Saved cleaned dataset (parquet): {clean_path}")

# # ---------- Persist column schema (JSON) ----------
# schema_path = PROC_DIR / f"cbecs_{year}_schema.json"
# schema = {
#     "columns": list(df_clean.columns),
#     "dtypes": {col: str(dtype) for col, dtype in df_clean.dtypes.items()}
# }
# with open(schema_path, "w", encoding="utf-8") as f:
#     json.dump(schema, f, indent=2)
# print(f"Saved schema: {schema_path}")


### Testing time.


# First read in the file...
df = pd.read_parquet(PROC_DIR / "cbecs_2018_clean.parquet")
cat_cols = [c for c in TYPES["categorical_variables"] if c in df.columns]
num_cols = [c for c in TYPES["numeric_variables"] if c in df.columns]

# Coerce all non-numeric columns to be astype object. Many are actually integers that refer to codes.
df[cat_cols] = df[cat_cols].astype(object)


# Assign X, y 
X = df.drop('LOG_MFBTU', axis=1)
y = df['LOG_MFBTU']

# Split the data into validation, train, and test sets
# e.g., 80/10/10 split
test_size = 0.10
val_size  = 0.10
rand      = 1

# 1) Hold out test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=rand, shuffle=True
)

# 2) From the remainder, carve out validation
val_rel = val_size / (1.0 - test_size)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=val_rel, random_state=rand, shuffle=True
    )

# Check and make sure the splits worked as expected...
df.shape
check_shape = X_train.shape[0]+X_test.shape[0]+X_val.shape[0] == df.shape[0]

print(f"Split correctly: {check_shape}")



import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np, pandas as pd


# num_pipe = Pipeline([
# ("median_imputer", MedianImputer(RULES["impute_rules"])),
# ("auto", AutoTransform(scoring="r2", n_jobs=-1, suffix_identity=False)),  # slow step; runs once below
# ("scaler", StandardScaler())
# ])
# cat_pipe = Pipeline([
#     ("ohe", OneHotEncoder(handle_unknown="ignore"))
# ])

# preproc = ColumnTransformer(
# transformers=[
#     ("num", num_pipe, selector(dtype_include=np.number)),
#     ("cat", cat_pipe, selector(dtype_include=["object", "category", "bool"])),
# ],
# remainder="drop"
# )


# lasso = Pipeline([
#     ("preproc", clone(preproc)),
#     ("model", LassoCV(cv=5, n_jobs=-1, max_iter=100_000, tol=1e-4))
# ])

# enet = Pipeline([
#     ("preproc", clone(preproc)),
#     ("model", ElasticNetCV(cv=5, n_jobs=-1,
#                            l1_ratio=[0.4,0.5,0.6,0.7,0.8],
#                            max_iter=100_000, tol=1e-4, selection="cyclic",
#                            random_state=42))
# ])

# s_lasso = cross_val_score(lasso, X_train, y_train, cv=rkf, scoring="r2", n_jobs=-1)
# s_enet  = cross_val_score(enet,  X_train, y_train, cv=rkf, scoring="r2", n_jobs=-1)

# print(f"Lasso  mean={s_lasso.mean():.3f} ± {s_lasso.std():.3f}")
# print(f"ENet   mean={s_enet.mean():.3f}  ± {s_enet.std():.3f}")
# print(f"Δ(EN-Lasso) = {(s_enet - s_lasso).mean():.4f}")

# Numeric pipeline: impute -> Yeo-Johnson (per-feature, handles zeros/negatives) -> (standardize inside PT)


class Winsorize(BaseEstimator, TransformerMixin):
    def __init__(self, low=0.001, high=0.999): self.low, self.high = low, high
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.q_low  = np.nanquantile(X, self.low,  axis=0)
        self.q_high = np.nanquantile(X, self.high, axis=0)
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.q_low, self.q_high)



num_pipe = Pipeline([
    ("median_imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("winsor",  Winsorize(0.001, 0.999)),           # <-- optional but helpful
    ("yeojohnson", PowerTransformer(method="box-cox", standardize=True)),  # or standardize=False + StandardScaler()
])

# Categorical pipeline
cat_pipe = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])



# Route by dtype
preproc = ColumnTransformer(
    transformers=[
        ("num", num_pipe, selector(dtype_include=np.number)),
        ("cat", cat_pipe, selector(dtype_include=["object", "category", "bool"])),
    ],
    remainder="drop"
)

# Full pipeline: preprocessing + model
pipe = Pipeline([
    ("preproc", preproc),
    ("model", Ridge())  # swap for other estimators if you want
])

# ----- Search space -----
# With Yeo-Johnson in place, just tune Ridge alpha
param_grid = {
    "model__alpha": np.logspace(-6, 6, 25),
}

cv = KFold(n_splits=5, shuffle=True, random_state=1)
search = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="r2",
    cv=cv,
    n_jobs=-1,
    refit=True,  # keep best model fitted on full train
)

# ----- Fit + evaluate -----
search.fit(X_train, y_train)
print("Best R² (CV):", f"{search.best_score_:.3f}")
print("Picked alpha:", search.best_params_["model__alpha"])

# Predict on test
y_pred = search.predict(X_test)
breakpoint()