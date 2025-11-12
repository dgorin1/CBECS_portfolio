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







class OneOffRecodeToMissing(BaseEstimator, TransformerMixin):
    def __init__(self, rules):
        self.rules = rules

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for label, rule in self.rules.items():
            X[label] = X[label].replace(rule["value"], np.nan)
            

        return X




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
    
# --- Drop-all-NaN variant of your MedianImputer ---

from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# class MedianImputer(BaseEstimator, TransformerMixin):
#     """
#     Impute numeric columns by median and add '{col}_median_imputed' flags.
#     If a numeric column is ALL NaN in the current fit split (no median can be learned),
#     it is DROPPED for that fit (optionally flagged via '{col}_all_missing_this_split').

#     Parameters
#     ----------
#     impute_rules : Dict
#         Rule dict for "manual" columns: {col: {"value": <trigger_value>}, ...}
#         For these columns, values equal to 'value' are treated like missing before imputation.
#     num_cols : Optional[List[str]]
#         Optional explicit list of numeric columns. If None, inferred from dtypes (include='number').

#     Attributes after fit
#     --------------------
#     _num_cols_ : List[str]
#         Numeric columns considered at fit (order follows X).
#     impute_cols_manual : List[str]
#         Subset of numeric columns covered by impute_rules and present in X.
#     medians_manual : Dict[str, float]
#         Learned medians for manual columns.
#     cols_to_auto_impute : List[str]
#         Other numeric columns with any NaNs at fit time.
#     medians_auto : Dict[str, float]
#         Learned medians for auto-impute columns.
#     all_nan_cols_ : List[str]
#         Numeric columns that were all-NaN in the fit split (dropped during transform).
#     """

#     def __init__(self, impute_rules: Dict, num_cols: Optional[List[str]] = None):
#         self.impute_rules = impute_rules or {}
#         self.columns_to_impute = list(self.impute_rules.keys())
#         # Back-compat alias from your original code
#         self.columns_to_inpute = self.columns_to_impute
#         self.num_cols = num_cols

#         # fitted attrs
#         self._num_cols_: List[str] = []
#         self.impute_cols_manual: List[str] = []
#         self.medians_manual: Dict[str, float] = {}
#         self.cols_to_auto_impute: List[str] = []
#         self.medians_auto: Dict[str, float] = {}
#         self.all_nan_cols_: List[str] = []

#     def fit(self, X: pd.DataFrame, y=None):
#         X = X.copy()

#         # Infer numeric columns (preserve original order in X)
#         if self.num_cols is None:
#             inferred = list(X.select_dtypes(include='number').columns)
#             self._num_cols_ = [c for c in X.columns if c in inferred]
#         else:
#             allow = set(self.num_cols)
#             self._num_cols_ = [c for c in X.columns if c in allow]

#         # Manual: numeric + listed in rules + present
#         self.impute_cols_manual = [
#             c for c in self.columns_to_impute if (c in X.columns and c in self._num_cols_)
#         ]

#         # Learn medians for manual columns (normal case)
#         self.medians_manual = {}
#         if self.impute_cols_manual:
#             self.medians_manual = X[self.impute_cols_manual].median().to_dict()

#         # Auto: other numeric cols that have NaNs at fit time
#         auto_candidates = [c for c in self._num_cols_ if c not in self.impute_cols_manual]
#         self.cols_to_auto_impute = []
#         self.medians_auto = {}
#         if auto_candidates:
#             needs_auto = X[auto_candidates].isna().any()
#             self.cols_to_auto_impute = needs_auto.index[needs_auto].tolist()
#             if self.cols_to_auto_impute:
#                 self.medians_auto = X[self.cols_to_auto_impute].median().to_dict()

#         # Detect columns that are ALL NaN (no finite value → no median learnable)
#         self.all_nan_cols_ = []
#         for c in self.impute_cols_manual + self.cols_to_auto_impute:
#             # We only look at numeric columns; safe to cast
#             col = pd.to_numeric(X[c], errors="coerce")
#             # "all missing" if there are no finite values
#             if not np.isfinite(col.to_numpy(dtype=float)).any():
#                 self.all_nan_cols_.append(c)

#         return self

#     def transform(self, X: pd.DataFrame, y=None):
#         X = X.copy()

#         # ---------- MANUAL IMPUTE (skip all-NaN columns) ----------
#         manual_cols = [c for c in self.impute_cols_manual
#                        if c in X.columns and c not in self.all_nan_cols_]
#         if manual_cols:
#             Xc = X[manual_cols]

#             # Treat trigger values as missing (per-rule)
#             trigger = pd.Series({c: self.impute_rules[c]['value'] for c in manual_cols}).reindex(manual_cols)
#             mask_trigger = Xc.eq(trigger)
#             mask_nan = Xc.isna()
#             flag = (mask_trigger | mask_nan).astype(int)

#             # Use medians learned at fit; fill any residual NaNs with fold-local medians
#             med_fit = pd.Series(self.medians_manual).reindex(manual_cols)
#             med_local = Xc.median(skipna=True)
#             # Replace trigger with fit medians first, then fill residual NaNs by local fold median
#             X.loc[:, manual_cols] = Xc.mask(mask_trigger, med_fit, axis=1).fillna(med_local)

#             # Add indicator columns
#             X[[f"{c}_median_imputed" for c in manual_cols]] = flag.astype(int)

#         # ---------- AUTO IMPUTE (skip all-NaN columns) ----------
#         auto_cols = [c for c in self.cols_to_auto_impute
#                      if c in X.columns and c not in self.all_nan_cols_]
#         if auto_cols:
#             Xc = X[auto_cols]
#             mask = Xc.isna()
#             flag_auto = mask.astype(int)

#             med_fit = pd.Series(self.medians_auto).reindex(auto_cols)
#             # If any med_fit is NaN (shouldn't happen unless column changed), fall back to fold-local
#             med_local = Xc.median(skipna=True).fillna(med_fit)
#             X.loc[:, auto_cols] = Xc.fillna(med_local)

#             X[[f"{c}_median_imputed" for c in auto_cols]] = flag_auto.astype(int)

#         # ---------- DROP all-NaN columns (for this fit split only) ----------
#         to_drop = [c for c in self.all_nan_cols_ if c in X.columns]
#         if to_drop:
#             # Optional: emit a per-split flag so you can audit removals
#             for c in to_drop:
#                 X[f"{c}_all_missing_this_split"] = 1
#             X[[f"{c}_all_missing_this_split" for c in to_drop]] = (
#                 X[[f"{c}_all_missing_this_split" for c in to_drop]].astype("uint8").astype(object)
#             )
#             X = X.drop(columns=to_drop, errors="ignore")

#         return X
        
# class OneHotEncoder(BaseEstimator, TransformerMixin):
#     """
#     DataFrame-friendly one-hot encoder with numeric passthrough.

#     - If `cat_cols` is None, infer categoricals as non-numeric dtypes (e.g., object/category).
#     - If `num_cols` is None, infer numerics with select_dtypes(include='number').
#     - Adds a 'missing' category for cats and fills NaNs with 'missing' before encoding.
#     - Aligns columns at transform time (drops unseen, adds missing with 0).
#     """

#     def __init__(self,
#                  cat_cols: Optional[List[str]] = None,
#                  num_cols: Optional[List[str]] = None,
#                  drop_first: bool = True,
#                  dtype=np.uint8):
#         self.cat_cols = None if cat_cols is None else list(cat_cols)
#         self.num_cols = None if num_cols is None else list(num_cols)
#         self.drop_first = drop_first
#         self.dtype = dtype

#     # internal: cast given categorical columns, add 'missing'
#     def _prepare_cats(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
#         if not cols:
#             return X
#         X = X.copy()
#         # Ensure categorical dtype, then add 'missing' and fill
#         X[cols] = X[cols].astype("category")
#         for c in cols:
#             X[c] = X[c].cat.add_categories(["missing"]).fillna("missing")
#         return X

#     def fit(self, X: pd.DataFrame, y=None):
#         # --- Infer columns if not provided ---
#         if self.num_cols is None:
#             inferred_num = list(X.select_dtypes(include='number').columns)
#             self.num_cols_ = [c for c in X.columns if c in inferred_num]  # preserve order
#         else:
#             allow = set(self.num_cols)
#             self.num_cols_ = [c for c in X.columns if c in allow]

#         if self.cat_cols is None:
#             inferred_cat = list(X.select_dtypes(exclude='number').columns)  # object/category/etc.
#             self.cat_cols_ = [c for c in X.columns if c in inferred_cat]
#         else:
#             allow = set(self.cat_cols)
#             self.cat_cols_ = [c for c in X.columns if c in allow]

#         # Prepare cats on a copy for fitting
#         Xc = self._prepare_cats(X, self.cat_cols_)

#         # Build training dummies and remember their columns
#         if self.cat_cols_:
#             X_ohe = pd.get_dummies(
#                 Xc[self.cat_cols_], drop_first=self.drop_first, dtype=self.dtype
#             )
#             self.ohe_cols_ = list(X_ohe.columns)
#         else:
#             self.ohe_cols_ = []

#         # Final order at transform: [numeric passthrough, one-hot columns]
#         self.all_cols_ = self.num_cols_ + self.ohe_cols_
#         return self

#     def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
#         # Numeric passthrough (only those learned in fit that are present now)
#         num_cols_now = [c for c in self.num_cols_ if c in X.columns]
#         X_num = X[num_cols_now] if num_cols_now else pd.DataFrame(index=X.index)

#         # Prepare and encode categoricals
#         cat_cols_now = [c for c in self.cat_cols_ if c in X.columns]
#         if cat_cols_now:
#             Xc = self._prepare_cats(X, cat_cols_now)
#             X_ohe = pd.get_dummies(
#                 Xc[cat_cols_now], drop_first=self.drop_first, dtype=self.dtype
#             )
#         else:
#             X_ohe = pd.DataFrame(index=X.index)

#         # Align to training OHE columns: add missing with 0, drop unseen
#         X_ohe = X_ohe.reindex(columns=self.ohe_cols_, fill_value=0)

#         # Concatenate and enforce final training order (adding any missing with 0)
#         X_out = pd.concat([X_num, X_ohe], axis=1)
#         X_out = X_out.reindex(columns=self.all_cols_, fill_value=0)

#         # Ensure index alignment with input
#         X_out.index = X.index
#         return X_out

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

class DropSparse(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X=X.copy()
        percent = X.isna().mean()
        cols_to_drop = percent.index[percent>self.threshold]
        return X.drop(columns=cols_to_drop)



import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# class AutoTransform(BaseEstimator, TransformerMixin):
#     """

#     """
#     def __init__(self, candidates=("none", "log1p", "sqrt")):
#         self.candidates = candidates
   
#     def fit(self, X, y):
#         # Get numeric columns
#         self.numeric_cols = self.get_numeric(X)

#         fit_val = []
#         for 
#         for candidate in self.candidates:
#             lr =  LinearRegression()
#             if candidate == "none":
#                 lr()

#             breakpoint()
            

#         return self

#     def transform(self, X):
       

#        return X
    
#     def get_numeric(self, X):
#         dtypes = ['number']
#         num = X.select_dtypes(include=dtypes)
#         is_binary = ((num == 0) | (num == 1) | num.isna()).all(axis=0)
#         out = num.columns[~is_binary].tolist()
        
#         return out

from typing import Dict, List
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from sklearn.linear_model import LinearRegression


from typing import Dict, List
import numpy as np
import pandas as pd
import time

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from sklearn.linear_model import LinearRegression


from typing import Dict, List
import numpy as np
import pandas as pd
import time

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from sklearn.linear_model import LinearRegression


from typing import Dict, List
import numpy as np
import pandas as pd
import time

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from sklearn.linear_model import LinearRegression


from typing import Dict, List
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from sklearn.linear_model import LinearRegression


from typing import Dict, List
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from sklearn.linear_model import LinearRegression


from typing import Dict, List
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from sklearn.linear_model import LinearRegression


from typing import Dict, List
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from sklearn.linear_model import LinearRegression
from pandas.api.types import is_numeric_dtype


class AutoTransform(BaseEstimator, TransformerMixin):
    """
    Select per-column transform from {'identity','log1p','sqrt'} via CV on a base estimator.

    Behavior:
    - Only **numeric** columns are considered transformable (object/bool are excluded).
    - Columns that are effectively one-hot/binary (non-NaN values ⊆ {0,1}) are **ignored** even if numeric.
    - Non-transformable columns pass through unchanged and keep original names.
    - If the chosen transform is 'identity', the column keeps its ORIGINAL name (no suffix).

    Scoring/selection:
    - During CV we use ONLY the active column (univariate) and drop rows
      if that column or y is NaN/inf. Other columns' NaNs are ignored for scoring.
    - We compute mean CV R^2 for each candidate transform.
    - We keep 'identity' unless the best alternative beats identity by ≥ 0.005 R^2.
    """

    def __init__(self,
                 estimator=None,
                 scoring: str = 'r2',   # selection/prints use R^2
                 cv: int = 5,
                 n_jobs = None,         # unused; kept for API compatibility
                 random_state: int = 42,
                 candidates  = None,
                 num_cols  = None,      # optional allowlist; will still be filtered to numeric & non-binary
                 return_df: bool = True):
        self.estimator = estimator if estimator is not None else LinearRegression()
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.candidates = candidates if candidates is not None else ['identity', 'log1p', 'sqrt']
        self.num_cols = num_cols
        self.return_df = return_df

        # fitted attrs
        self.best_per_feature_: Dict[str, str] = {}           # {num_col: chosen_transform}
        self.cv_scores_: Dict[str, Dict[str, float]] = {}     # {num_col: {transform: mean_r2}}
        self._cols_: List[str] = []
        self._num_cols_: List[str] = []        # numeric & not binary
        self._non_num_cols_: List[str] = []    # everything else (incl. binary)
        self._binary_cols_: List[str] = []     # subset of _non_num_cols_ that are binary numeric
        self._out_cols_: List[str] = []

        # threshold to beat identity
        self._improve_thresh_ = 0.005

    # ---- helpers ----
    @staticmethod
    def _is_binary_series(s: pd.Series) -> bool:
        """Return True if non-NaN unique values are subset of {0,1}."""
        if s.dtype == bool:
            return True
        vals = pd.unique(s.dropna())
        if len(vals) == 0:
            return False
        try:
            # handle float/integer zeros/ones
            return set(np.unique(vals)).issubset({0, 1, 0.0, 1.0})
        except TypeError:
            return False

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

    def _cv_mean_r2_on_single_column(self, est, xcol: np.ndarray, y, cv) -> float:
        """
        Mean CV R^2 using ONLY the provided 1D column `xcol`.
        Rows are dropped per split only if that column or y is non-finite.
        """
        scorer = get_scorer('r2')
        xcol = np.asarray(xcol, dtype=float).reshape(-1)
        y = np.asarray(y)

        scores = []
        for train_idx, test_idx in cv.split(xcol.reshape(-1, 1), y):
            xtr = xcol[train_idx]; xte = xcol[test_idx]
            ytr = y[train_idx];     yte = y[test_idx]

            tr_mask = np.isfinite(xtr) & np.isfinite(ytr)
            te_mask = np.isfinite(xte) & np.isfinite(yte)
            if tr_mask.sum() == 0 or te_mask.sum() == 0:
                continue

            Xtr_1 = xtr[tr_mask].reshape(-1, 1)
            Xte_1 = xte[te_mask].reshape(-1, 1)
            ytr_1 = ytr[tr_mask]
            yte_1 = yte[te_mask]

            est_ = clone(est)
            est_.fit(Xtr_1, ytr_1)
            scores.append(scorer(est_, Xte_1, yte_1))

        return float(np.mean(scores)) if scores else float('-inf')

    # ---- sklearn API ----
    def fit(self, X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self._cols_ = list(X_df.columns)

        # Candidate numeric columns
        if self.num_cols is None:
            numeric_candidates = list(X_df.select_dtypes(include=[np.number]).columns)
        else:
            # keep order from X_df but restrict to provided list
            allow = set(self.num_cols)
            numeric_candidates = [c for c in X_df.columns if c in allow and is_numeric_dtype(X_df[c])]
            skipped_non_numeric = [c for c in self.num_cols if c in X_df.columns and not is_numeric_dtype(X_df[c])]
            missing = [c for c in self.num_cols if c not in X_df.columns]
            if missing:
                print(f"[AutoTransform] Skipping missing columns (not in X): {missing}", flush=True)
            if skipped_non_numeric:
                print(f"[AutoTransform] Skipping non-numeric columns from num_cols: {skipped_non_numeric}", flush=True)

        # Filter out binary (one-hot) columns among numeric
        self._binary_cols_ = [c for c in numeric_candidates if self._is_binary_series(X_df[c])]
        self._num_cols_ = [c for c in numeric_candidates if c not in self._binary_cols_]
        self._non_num_cols_ = [c for c in self._cols_ if c not in self._num_cols_]

        print(f"[AutoTransform] Numeric transformable columns: {len(self._num_cols_)}", flush=True)
        if self._binary_cols_:
            print(f"[AutoTransform] Ignoring {len(self._binary_cols_)} binary columns (0/1): {self._binary_cols_}", flush=True)
        non_numeric = [c for c in self._cols_ if c not in numeric_candidates]
        if non_numeric:
            print(f"[AutoTransform] Passing through {len(non_numeric)} non-numeric columns unchanged.", flush=True)

        # numeric matrix for evaluation
        X_num = X_df[self._num_cols_].to_numpy(dtype=float) if self._num_cols_ else np.empty((len(X_df), 0))
        y = np.asarray(y)

        cv = self.cv if not isinstance(self.cv, int) else KFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        self.best_per_feature_.clear()
        self.cv_scores_.clear()

        # Per-column transform selection
        for j, col_name in enumerate(self._num_cols_):
            col = X_num[:, j]
            r2_for_col: Dict[str, float] = {}
            print(f"[AutoTransform] Column '{col_name}':", flush=True)

            # Evaluate every candidate
            for name in self.candidates:
                if not self._valid(name, col):
                    r2 = float('-inf')
                else:
                    xcol_t = self._apply_once(name, col)
                    est = clone(self.estimator)
                    r2 = self._cv_mean_r2_on_single_column(est, xcol_t, y, cv)

                r2_for_col[name] = r2
                print(f"  - {name:<8} R^2 = {r2:.6f}", flush=True)

            # Decision: keep identity unless best beats it by >= threshold
            r2_identity = r2_for_col.get('identity', float('-inf'))
            best_name = max(r2_for_col, key=r2_for_col.get)
            best_r2 = r2_for_col[best_name]

            chosen_name = 'identity'
            if best_name != 'identity' and (best_r2 - r2_identity) >= self._improve_thresh_:
                chosen_name = best_name

            # Print summary
            others = {k: v for k, v in r2_for_col.items() if k != chosen_name}
            others_str = ", ".join([f"{k}: {v:.6f}" for k, v in others.items()])
            if chosen_name == 'identity':
                delta = best_r2 - r2_identity
                print(f"  -> kept: {col_name} (identity) | best alt ΔR^2 = {delta:.6f} < {self._improve_thresh_:.3f}; "
                      f"others: {others_str}", flush=True)
            else:
                delta = best_r2 - r2_identity
                print(f"  -> picked: {chosen_name} (R^2 = {best_r2:.6f}, +{delta:.6f} over identity); "
                      f"others: {others_str}", flush=True)

            self.best_per_feature_[col_name] = chosen_name
            self.cv_scores_[col_name] = r2_for_col

        # Build output column names in original order
        self._out_cols_.clear()
        for c in self._cols_:
            if c in self._num_cols_:
                suffix = self.best_per_feature_[c]
                # If identity, keep original name; otherwise add suffix
                self._out_cols_.append(c if suffix == 'identity' else f"{c}_{suffix}")
            else:
                self._out_cols_.append(c)

        return self

    def transform(self, X):
        print("[AutoTransform] Transforming...", flush=True)
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self._cols_)
        out_parts = []
        for c in self._cols_:
            if c in self._num_cols_:
                tname = self.best_per_feature_[c]
                arr = self._apply_once(tname, X_df[c].to_numpy(dtype=float))
                # If identity, keep original name; otherwise add suffix
                col_name = c if (tname == 'identity') else f"{c}_{tname}"
                out_parts.append(pd.Series(arr, index=X_df.index, name=col_name))
            else:
                # passthrough (includes non-numeric and binary columns)
                out_parts.append(X_df[c])

        X_out = pd.concat(out_parts, axis=1)
        X_out = X_out[self._out_cols_]  # enforce final order
        print("[AutoTransform] Transform complete.", flush=True)
        return X_out if self.return_df else X_out.to_numpy()
    
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Clip each numeric feature to [lower_quantile, upper_quantile] learned on train.
    Example: lower=0.01, upper=0.99  (i.e., 1%/99% winsorization)
    """
    def __init__(self, lower=0.01, upper=0.99):
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError("Require 0 <= lower < upper <= 1")
        self.lower = lower
        self.upper = upper
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        # nan-safe quantiles (SimpleImputer already removed NaNs, but guard anyway)
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)

        # handle degenerate cases where quantiles collapse
        tie = self.upper_ < self.lower_
        if np.any(tie):
            # swap where needed
            lo, hi = self.lower_.copy(), self.upper_.copy()
            self.lower_[tie], self.upper_[tie] = hi[tie], lo[tie]
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.clip(X, self.lower_, self.upper_)