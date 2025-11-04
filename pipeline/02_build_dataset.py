# pipeline/02_build_dataset.py
from __future__ import annotations
import pathlib, yaml
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from transformers import (
    TopCodeClipper, MassRecoder, MassDrop, MedianImputer)

# ---------- Paths & config ----------
PROJ = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(PROJ/"conf"/"config.yaml"))
TYPES = yaml.safe_load(open(PROJ/"conf"/"variable_types.yaml"))
RULES = yaml.safe_load(open(PROJ/"conf"/"cleaning_rules.yaml"))

RAW_DIR = PROJ/CFG["data"]["raw_dir"]
PROC_DIR = PROJ/CFG["data"]["processed_dir"]
PROC_DIR.mkdir(parents=True, exist_ok=True)

year = CFG["datasets"]["year"]
target_col = CFG["build"]["target"]

print(f"Loading raw: {year}")
df = pd.read_csv(PROJ / "data" / "raw" /  f"cbecs_{year}_microdata.csv")

# ---------- Column groups ----------
cat_cols = [c for c in TYPES["categorical_variables"] if c in df.columns]
num_cols = [c for c in TYPES["numeric_variables"] if c in df.columns]



# ---------- Pre-clean block  ----------
preclean = Pipeline(steps=[
    ("mass_drop", MassDrop(RULES.get("drop_columns"), RULES.get("regex_for_delete"))),
    ("mass_recodes", MassRecoder(RULES.get("recode_rules"))),
    ("top_codes", TopCodeClipper(RULES.get("top_codes"))),
    ("median_imputer", MedianImputer(RULES.get("impute_rules"))),
    
    
    ])

df_clean = preclean.fit_transform(df)






