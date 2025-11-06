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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel

from transformers import (
    TopCodeClipper, MassRecoder, MassDrop, MedianImputer, OneHotEncoder, DropMissing,
    AutoTransform)

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

# Take log of the target variable and delete the original column.

df['LOG_MFBTU'] = np.log(df['MFBTU'])
df = df.drop(columns=['MFBTU'])

# ---------- Pre-clean block  ----------
preclean = Pipeline(steps=[
    ("drop_missing", DropMissing(RULES.get("missing_thresh"), "LOG_MFBTU")),
    ("mass_drop", MassDrop(RULES.get("drop_columns"), RULES.get("regex_for_delete"))), # Drop obviously redundant, useless, or columsn with obvious data leakage risks
    ("mass_recodes", MassRecoder(RULES.get("recode_rules"))), # Recode missing variables to 0, 1, or 2 when obvious. i.e. "# elevators missing is 0"
    ("top_codes", TopCodeClipper(RULES.get("top_codes"))), # Top code values which were top coded
    ("One_hot_encoder", OneHotEncoder(cat_cols, num_cols)), # One hot encode categorical variables
    ("median_imputer", MedianImputer(RULES.get("impute_rules"))) # Impute median where necessary and add imputation flags because missing has meaning in this dataset
    ])

df_clean = preclean.fit_transform(df)
nan_columns = df_clean.columns[df_clean.isna().any()].tolist()



# Split Data
X = df_clean.drop("LOG_MFBTU", axis=1)
y = df_clean.LOG_MFBTU

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# ---------- Feature engineering block  ----------
linear_feat_engineer = Pipeline(steps = [
    ('Auto_transform', AutoTransform(
        scoring='r2',
        cv=5,
        n_jobs=-1,
        num_cols = num_cols,
        suffix_identity=False
    )),
    ('scaler', StandardScaler()),
    ('select', SelectFromModel(LassoCV(cv=10, n_jobs=-1), prefit=False)),  # fast feature cut
    ('model', LinearRegression())

])

# Fit the full pipeline (model included)  <<< changed from fit_transform
linear_feat_engineer.fit(x_train, y_train)

