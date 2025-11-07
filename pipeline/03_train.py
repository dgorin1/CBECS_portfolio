# pipeline/02_build_dataset.py
from __future__ import annotations
import pathlib, yaml
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from transformers import (
    TopCodeClipper, MassRecoder, MassDrop, OneHotEncoder, DropMissing
)
# Split Data
X = df_clean.drop("LOG_MFBTU", axis=1)
y = df_clean.LOG_MFBTU

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# ---------- Feature engineering block  ----------
linear_feat_engineer = Pipeline(steps = [

    ("median_imputer", MedianImputer(RULES.get("impute_rules"))), # Impute median where necessary and add imputation flags because missing has meaning in this dataset
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