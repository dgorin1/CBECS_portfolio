# pipeline/transformers.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from typing import Dict, List, Any
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold




class TopCodeClipper(BaseEstimator, TransformerMixin):
    """Clip top-coded numeric columns to their cap and ."""
    def __init__(self, top_codes: Dict):
        self.top_codes = top_codes
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        for label, rule in self.top_codes.items():
            if label in X.columns:
                # Change the cap to the actual number
                mask =  X[label] == rule['threshold']
                X.loc[mask,label] = rule['cap_to']
                # Add flag
                X[f"{label}_topcoded"] = mask.astype(int)
        return X
    
class MassRecoder(BaseEstimator, TransformerMixin):
    """Replace NaNs with 0, 1, or 2 depending on codebook"""
    def __init__(self, rules):
        self.rules = rules

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        for rule in self.rules:
            columns = [c for c in rule['columns'] if c in X.columns]
            if rule['when'] == "NA":
                X.loc[:, columns] = X.loc[:, columns].fillna(rule["set_to"])
            else:
                mask = X[columns] == int(rule['when'])
                X[columns] = X[columns].mask(X[columns] == int(rule["when"]), int(rule["set_to"]))
        return X
    
class MassDrop(BaseEstimator, TransformerMixin):
    """Drops columns from list provided in yaml. This is for columns that are obviously useless"""
    def __init__(self, cols_to_drop, regex_for_delete):
        self.cols_to_drop = cols_to_drop
        self.regex_for_delete = regex_for_delete

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        self.cols_to_drop = [c for c in self.cols_to_drop if c in X.columns]
        # Cols with specific pattern indicating flags
        flags_to_drop = X.filter(regex=self.regex_for_delete, axis=1).columns.tolist()
        
        # Cols with pattern indicating they're survey weights
        weight_cols_to_drop = [c for c in X.columns if self._is_0_1_9_only(X[c])]

        # Drop cols
        X = X.drop(self.cols_to_drop + flags_to_drop + weight_cols_to_drop, axis=1)
        return X
    
    def _is_0_1_9_only(self,X) -> bool:
        """
        Returns True if (ignoring NaN) the unique numeric values in the column
        are a subset of {0,1,9}.
        """
        vals = X.dropna().unique()
        return len(vals) > 0 and set(np.unique(vals)).issubset({0, 1, 9})
    
class MedianImputer(BaseEstimator, TransformerMixin):
    """Impute Median and add flag column"""

    def __init__(self, impute_rules):
        self.impute_rules = impute_rules
        self.columns_to_inpute = impute_rules.keys()

    def fit(self, X, y=None):
        self.impute_cols_manual = [c for c in self.columns_to_inpute if c in X.columns]
        self.impute_cols_auto = X.columns[X.isna().any()].tolist()
        self.medians_manual = X[self.impute_cols_manual].median().to_dict()
        self.medians_auto = X[self.impute_cols_auto].median().to_dict()
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        for col in self.impute_cols_manual:
            mask = X[col] == self.impute_rules[col]['value']
            X[col] = X[col].mask(mask, self.medians_manual[col])

            # Create flag column
            X[f"{col}_median_imputed"] = mask.astype(int)
            
        for col in self.impute_cols_auto:
            # Fill Median
            mask = X[col].isna()
            X[col] = X[col].fillna(self.medians_auto[col])

            # Create flag
            X[f"{col}_median_imputed"] = mask.astype(int)
        return X
        
    
class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, num_cols):
        self.cat_cols = cat_cols
        self.num_cols = num_cols

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        self.cat_cols = [c for c in self.cat_cols if c in X.columns]
        self.num_cols = [c for c in self.num_cols if c in X.columns]

        # Fill NA with Missing so it can be its own category
        X[self.cat_cols] = X[self.cat_cols].fillna("missing")
        
        # Perform One Hot Encoding (OHE) on categorical columns
        X_ohe = pd.get_dummies(X[self.cat_cols], drop_first=True)  # drop_first=False if you want to keep all categories
        X = pd.concat([X.drop(self.cat_cols, axis=1), X_ohe], axis=1)
        return X


class DropMissing(BaseEstimator, TransformerMixin):
    
    def __init__(self, thresh, target):
        self.thresh = thresh
        self.target = target

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        missing_frac = X.isna().mean()
        to_drop = missing_frac[missing_frac >= self.thresh].index
        X = X.dropna(subset = self.target)
        X = X.drop(columns=to_drop)
        return X


class AutoTransform(BaseEstimator, TransformerMixin):
    """
    Select per-column transform from {'identity','log1p','sqrt'} via CV on a base estimator.
    - Only columns in `num_cols` are considered numeric/transformable.
    - Non-numeric (i.e., not in num_cols) are passed through unchanged.
    - Output keeps original order; numeric columns are renamed to '{col}_{transform}'.
    """

    def __init__(self,
                 estimator=None,
                 scoring: str = 'neg_mean_squared_error',
                 cv: int = 5,
                 n_jobs = None,
                 random_state: int = 42,
                 candidates  = None,
                 num_cols  = None,   # <- pass your numeric columns here
                 return_df: bool = True,
                 suffix_identity: bool = True):
        self.estimator = estimator if estimator is not None else LinearRegression()
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.candidates = candidates if candidates is not None else ['identity', 'log1p', 'sqrt']
        self.num_cols = num_cols
        self.return_df = return_df
        self.suffix_identity = suffix_identity

        # fitted attrs
        self.best_per_feature_: Dict[str, str] = {}           # {num_col: chosen_transform}
        self.cv_scores_: Dict[str, Dict[str, float]] = {}     # {num_col: {transform: score}}
        self._cols_: List[str] = []
        self._num_cols_: List[str] = []
        self._non_num_cols_: List[str] = []
        self._out_cols_: List[str] = []

    # ---- helpers ----
    def _valid(self, name: str, col: np.ndarray) -> bool:
        m = np.nanmin(col)
        if name == 'sqrt':   return m >= 0
        if name == 'log1p':  return m >= -1
        return True  # identity

    def _apply_once(self, name: str, col: np.ndarray) -> np.ndarray:
        if name == 'identity': return col
        if name == 'log1p':    return np.log1p(col)
        if name == 'sqrt':     return np.sqrt(col)
        return col

    # ---- sklearn API ----
    def fit(self, X, y):
        # Coerce to DataFrame for labeled ops
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self._cols_ = list(X_df.columns)

        # Use provided num_cols; validate and preserve order as in X
        if self.num_cols is None:
            # fallback: treat all columns as numeric if none provided
            self._num_cols_ = self._cols_
        else:
            missing = [c for c in self.num_cols if c not in self._cols_]
            if missing:
                raise ValueError(f"num_cols contains columns not in X: {missing}")
            # keep only columns that exist and preserve dataset order
            self._num_cols_ = [c for c in self._cols_ if c in set(self.num_cols)]

        self._non_num_cols_ = [c for c in self._cols_ if c not in self._num_cols_]

        # numeric matrix for evaluation
        X_num = X_df[self._num_cols_].to_numpy(dtype=float)
        y = np.asarray(y)

        cv = self.cv if not isinstance(self.cv, int) else KFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        self.best_per_feature_.clear()
        self.cv_scores_.clear()

        # Per-column transform selection
        for j, col_name in enumerate(self._num_cols_):
            col = X_num[:, j]
            scores_for_col: Dict[str, float] = {}
            best_score = -np.inf
            best_name = 'identity'

            for name in self.candidates:
                if not self._valid(name, col):
                    continue
                X_tmp = X_num.copy()
                X_tmp[:, j] = self._apply_once(name, col)

                est = clone(self.estimator)
                scores = cross_val_score(est, X_tmp, y, scoring=self.scoring,
                                         cv=cv, n_jobs=self.n_jobs)
                mean_score = float(np.mean(scores))
                scores_for_col[name] = mean_score
                if mean_score > best_score:
                    best_score, best_name = mean_score, name

            self.best_per_feature_[col_name] = best_name
            self.cv_scores_[col_name] = scores_for_col

        # Build output column names in original order
        self._out_cols_.clear()
        for c in self._cols_:
            if c in self._num_cols_:
                suffix = self.best_per_feature_[c]
                if suffix == 'identity' and not self.suffix_identity:
                    self._out_cols_.append(c)
                else:
                    self._out_cols_.append(f"{c}_{suffix}")
            else:
                self._out_cols_.append(c)
        return self

    def transform(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self._cols_)
        out_parts = []
        for c in self._cols_:
            if c in self._num_cols_:
                tname = self.best_per_feature_[c]
                arr = self._apply_once(tname, X_df[c].to_numpy(dtype=float))
                col_name = c if (tname == 'identity' and not self.suffix_identity) else f"{c}_{tname}"
                out_parts.append(pd.Series(arr, index=X_df.index, name=col_name))
            else:
                out_parts.append(X_df[c])

        X_out = pd.concat(out_parts, axis=1)
        X_out = X_out[self._out_cols_]  # enforce final order
        return X_out if self.return_df else X_out.to_numpy()