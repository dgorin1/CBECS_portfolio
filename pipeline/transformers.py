# pipeline/transformers.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from typing import Dict, List, Any
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import time
import sys

import pandas as pd



class CategorizeCols(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols

    def fit(self, X ,y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X[self.cat_cols] = X[self.cat_cols].astype(object)
        return X


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
                X[f"{label}_topcoded"] = mask.astype(int).astype(object)
        return X
    
class MassRecoder(BaseEstimator, TransformerMixin):
    """Replace NaNs with 0, 1, or 2 depending on codebook"""
    def __init__(self, rules):
        self.rules = rules

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        pd.set_option('future.no_silent_downcasting', True)
        for rule in self.rules:
            columns = [c for c in rule['columns'] if c in X.columns]
            if rule['when'] == "NA":
                X.loc[:, columns] = X.loc[:, columns].fillna(rule["set_to"])
            else:
                mask = X[columns] == int(rule['when'])
                X[columns] = X[columns].mask(X[columns] == int(rule["when"]), int(rule["set_to"]))
        return X
    
class MassDrop(BaseEstimator, TransformerMixin):
    """Drops columns from list provided in yaml, regex matches, and any columns starting with 'Z'."""
    def __init__(self, cols_to_drop, regex_for_delete):
        self.cols_to_drop = cols_to_drop
        self.regex_for_delete = regex_for_delete

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        # Keep only cols that exist
        explicit_drops = [c for c in self.cols_to_drop if c in X.columns]

        # Regex-based drops
        flags_to_drop = X.filter(regex=self.regex_for_delete, axis=1).columns.tolist()

        # Any column whose name starts with 'Z'
        z_prefix_cols = [c for c in X.columns if str(c).startswith('Z')]

        # Drop them all
        X = X.drop(explicit_drops + flags_to_drop + z_prefix_cols, axis=1, errors='ignore')
        return X
    
class MedianImputer(BaseEstimator, TransformerMixin):
    """Impute per-column median and add '{col}_median_imputed' flag.

    Behavior
    --------
    - Manual rules: for specified columns, replace a trigger value (and NaNs) then fill with column median.
    - Automatic: for *other* numeric columns with any NaNs, fill with their medians.
    - Numeric columns are auto-detected from dtypes unless `num_cols` is provided.
    """

    def __init__(self, impute_rules: Dict, num_cols: Optional[List[str]] = None):
        self.impute_rules = impute_rules or {}
        self.columns_to_impute = list(self.impute_rules.keys())
        # Backwards-compat alias for old code that referenced the misspelled attribute
        self.columns_to_inpute = self.columns_to_impute
        self.num_cols = num_cols  # optional override

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        # infer numeric columns unless the user provided an explicit list
        if self.num_cols is None:
            inferred_num = list(X.select_dtypes(include='number').columns)
            # preserve order as in X
            self._num_cols_ = [c for c in X.columns if c in inferred_num]
        else:
            allowed = set(self.num_cols)
            self._num_cols_ = [c for c in X.columns if c in allowed]

        # MANUAL: apply only to columns that are present *and* numeric
        self.impute_cols_manual = [
            c for c in self.columns_to_impute if (c in X.columns and c in self._num_cols_)
        ]

        # medians for manual step
        self.medians_manual = X[self.impute_cols_manual].median().to_dict()

        # AUTOMATIC: numeric columns with NaNs, excluding those covered by manual rules
        auto_candidates = [c for c in self._num_cols_ if c not in self.impute_cols_manual]
        if auto_candidates:
            needs_auto = X[auto_candidates].isna().any()
            self.cols_to_auto_impute = needs_auto.index[needs_auto].tolist()
            self.medians_auto = X[self.cols_to_auto_impute].median().to_dict()
        else:
            self.cols_to_auto_impute = []
            self.medians_auto = {}

        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()

        # ---------- MANUAL ----------
        cols = [c for c in self.impute_cols_manual if c in X.columns]
        if cols:
            Xc = X[cols]

            # trigger values per column (e.g., -1, 999, etc.)
            trigger_values = pd.Series({c: self.impute_rules[c]['value'] for c in cols}).reindex(cols)

            # where to flag (trigger OR NaN)
            mask_trigger = Xc.eq(trigger_values)
            mask_nan = Xc.isna()
            flag = (mask_trigger | mask_nan).astype(int)

            # compute medians BEFORE modifying values
            med_before = Xc.median(skipna=True)
            med_manual = pd.Series(self.medians_manual).reindex(cols)

            # replace trigger with manual medians, then fill residual NaNs with pre-change medians
            X.loc[:, cols] = Xc.mask(mask_trigger, med_manual, axis=1).fillna(med_before)

            # add indicator columns
            X[[f"{c}_median_imputed" for c in cols]] = flag.astype(int).astype(object)

        # ---------- AUTOMATIC ----------
        cols_auto = [c for c in self.cols_to_auto_impute if c in X.columns]
        if cols_auto:
            Xc = X[cols_auto]
            mask = Xc.isna()
            flag_auto = mask.astype(int)

            med = pd.Series(self.medians_auto).reindex(cols_auto)
            X.loc[:, cols_auto] = Xc.fillna(med)

            X[[f"{c}_median_imputed" for c in cols_auto]] = flag_auto.astype(int).astype(object)

        return X
        
    
class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    DataFrame-friendly one-hot encoder with numeric passthrough.

    - If `cat_cols` is None, infer categoricals as non-numeric dtypes (e.g., object/category).
    - If `num_cols` is None, infer numerics with select_dtypes(include='number').
    - Adds a 'missing' category for cats and fills NaNs with 'missing' before encoding.
    - Aligns columns at transform time (drops unseen, adds missing with 0).
    """

    def __init__(self,
                 cat_cols: Optional[List[str]] = None,
                 num_cols: Optional[List[str]] = None,
                 drop_first: bool = True,
                 dtype=np.uint8):
        self.cat_cols = None if cat_cols is None else list(cat_cols)
        self.num_cols = None if num_cols is None else list(num_cols)
        self.drop_first = drop_first
        self.dtype = dtype

    # internal: cast given categorical columns, add 'missing'
    def _prepare_cats(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        if not cols:
            return X
        X = X.copy()
        # Ensure categorical dtype, then add 'missing' and fill
        X[cols] = X[cols].astype("category")
        for c in cols:
            X[c] = X[c].cat.add_categories(["missing"]).fillna("missing")
        return X

    def fit(self, X: pd.DataFrame, y=None):
        # --- Infer columns if not provided ---
        if self.num_cols is None:
            inferred_num = list(X.select_dtypes(include='number').columns)
            self.num_cols_ = [c for c in X.columns if c in inferred_num]  # preserve order
        else:
            allow = set(self.num_cols)
            self.num_cols_ = [c for c in X.columns if c in allow]

        if self.cat_cols is None:
            inferred_cat = list(X.select_dtypes(exclude='number').columns)  # object/category/etc.
            self.cat_cols_ = [c for c in X.columns if c in inferred_cat]
        else:
            allow = set(self.cat_cols)
            self.cat_cols_ = [c for c in X.columns if c in allow]

        # Prepare cats on a copy for fitting
        Xc = self._prepare_cats(X, self.cat_cols_)

        # Build training dummies and remember their columns
        if self.cat_cols_:
            X_ohe = pd.get_dummies(
                Xc[self.cat_cols_], drop_first=self.drop_first, dtype=self.dtype
            )
            self.ohe_cols_ = list(X_ohe.columns)
        else:
            self.ohe_cols_ = []

        # Final order at transform: [numeric passthrough, one-hot columns]
        self.all_cols_ = self.num_cols_ + self.ohe_cols_
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        # Numeric passthrough (only those learned in fit that are present now)
        num_cols_now = [c for c in self.num_cols_ if c in X.columns]
        X_num = X[num_cols_now] if num_cols_now else pd.DataFrame(index=X.index)

        # Prepare and encode categoricals
        cat_cols_now = [c for c in self.cat_cols_ if c in X.columns]
        if cat_cols_now:
            Xc = self._prepare_cats(X, cat_cols_now)
            X_ohe = pd.get_dummies(
                Xc[cat_cols_now], drop_first=self.drop_first, dtype=self.dtype
            )
        else:
            X_ohe = pd.DataFrame(index=X.index)

        # Align to training OHE columns: add missing with 0, drop unseen
        X_ohe = X_ohe.reindex(columns=self.ohe_cols_, fill_value=0)

        # Concatenate and enforce final training order (adding any missing with 0)
        X_out = pd.concat([X_num, X_ohe], axis=1)
        X_out = X_out.reindex(columns=self.all_cols_, fill_value=0)

        # Ensure index alignment with input
        X_out.index = X.index
        return X_out

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

    Behavior:
    - Numeric columns are auto-detected with pandas dtypes (include='number').
      You can still override with `num_cols` if desired.
    - Non-numeric columns are passed through unchanged.
    - Output keeps original order; numeric columns are renamed to '{col}_{transform}',
      unless `suffix_identity=False` (then identity keeps the original name).
    """
    def __init__(self,
                 estimator=None,
                 scoring: str = 'neg_mean_squared_error',
                 cv: int = 5,
                 n_jobs = None,
                 random_state: int = 1,
                 candidates = None,
                 num_cols: Optional[List[str]] = None,  # optional override
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
    def _apply_once(self, name: str, col: np.ndarray) -> np.ndarray:
        if name == 'identity': return col
        if name == 'log1p':    return np.log1p(col)
        if name == 'sqrt':     return np.sqrt(col)
        return col

    def _print(self, msg: str):
        # simple hook for optional logging
        print(msg)

    # ---- sklearn API ----
    def fit(self, X, y):
        t0 = time.time()

        # Coerce to DataFrame for labeled ops
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self._cols_ = list(X_df.columns)

        # Infer numeric columns unless an explicit override is provided
        if self.num_cols is None:
            inferred_num = list(X_df.select_dtypes(include='number').columns)
            self._num_cols_ = [c for c in self._cols_ if c in inferred_num]  # preserve order
        else:
            # Use provided list, preserving order as in X
            allow = set(self.num_cols)
            self._num_cols_ = [c for c in self._cols_ if c in allow]

        self._non_num_cols_ = [c for c in self._cols_ if c not in self._num_cols_]

        # numeric matrix for evaluation (may be empty if no numeric cols)
        X_num = X_df[self._num_cols_].to_numpy(dtype=float, copy=False)
        y = np.asarray(y)

        # Prebuild CV splitter once (only if there are numeric cols)
        if len(self._num_cols_) > 0:
            kf = self.cv if not isinstance(self.cv, int) else KFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
            splits = list(kf.split(X_num, y))  # reuse splits
        else:
            splits = []

        self.best_per_feature_.clear()
        self.cv_scores_.clear()

        total = len(self._num_cols_)

        # ---------- precompute candidate columns for ALL features (vectorized) ----------
        cand_mat = {'identity': X_num}

        if total > 0:
            # validity masks
            min_vals = np.nanmin(X_num, axis=0)  # safe because total > 0
            valid_log = (min_vals >= -1)
            valid_sqrt = (min_vals >= 0)

            if 'log1p' in self.candidates:
                X_log = np.empty_like(X_num)
                X_log[:] = np.nan
                X_log[:, valid_log] = np.log1p(X_num[:, valid_log])
                cand_mat['log1p'] = X_log

            if 'sqrt' in self.candidates:
                X_sqrt = np.empty_like(X_num)
                X_sqrt[:] = np.nan
                X_sqrt[:, valid_sqrt] = np.sqrt(X_num[:, valid_sqrt])
                cand_mat['sqrt'] = X_sqrt

        # ---------- feature loop  ----------
        for j, col_name in enumerate(self._num_cols_):
            scores_for_col = {}
            best_score = -np.inf
            best_name = 'identity'

            # collect candidates valid for this column (check NaN marker)
            valid_names = []
            for name, M in cand_mat.items():
                if name == 'identity' or not np.isnan(M[0, j]):
                    valid_names.append(name)

            # Score each candidate by CV, reusing folds
            for name in valid_names:
                mean_scores = []

                for tr_idx, te_idx in splits:
                    Xtr = cand_mat['identity'][tr_idx].copy()  # one copy per fold
                    Xtr[:, j] = cand_mat[name][tr_idx, j]
                    Xte = cand_mat['identity'][te_idx].copy()
                    Xte[:, j] = cand_mat[name][te_idx, j]

                    est = clone(self.estimator)
                    est.fit(Xtr, y[tr_idx])

                    if self.scoring == 'r2':
                        yhat = est.predict(Xte)
                        ss_res = np.sum((y[te_idx] - yhat) ** 2)
                        ss_tot = np.sum((y[te_idx] - y[te_idx].mean()) ** 2)
                        score = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                    else:
                        from sklearn.metrics import get_scorer
                        scorer = get_scorer(self.scoring)
                        score = scorer(est, Xte, y[te_idx])

                    mean_scores.append(score)

                mean_score = float(np.mean(mean_scores)) if len(mean_scores) else -np.inf
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
                self._out_cols_.append(c if (suffix == 'identity' and not self.suffix_identity)
                                    else f"{c}_{suffix}")
            else:
                self._out_cols_.append(c)

        # done
        _ = time.time() - t0
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