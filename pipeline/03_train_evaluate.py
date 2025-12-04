from __future__ import annotations

import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import optuna  # type: ignore
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None  # type: ignore[assignment]

PerfWarn = pd.errors.PerformanceWarning
warnings.filterwarnings("ignore", category=PerfWarn)

# ---------------------------------------------------------------------------
# Project / paths
# ---------------------------------------------------------------------------

# This file lives at: <project_root>/src/pipeline/03_train_evaluate.py
PROJ = Path(__file__).resolve().parents[1]

print(f"[INFO] Project root resolved to: {PROJ}")

CFG = yaml.safe_load(open(PROJ / "conf" / "config.yaml"))
VAR_TYPES = yaml.safe_load(open(PROJ / "conf" / "variable_types.yaml"))

RAW_DIR = PROJ / CFG["data"]["raw_dir"]
PROC_DIR = PROJ / CFG["data"]["processed_dir"]

ARTIFACTS = PROJ / "notebooks" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Artifacts directory: {ARTIFACTS}")

TARGET_COL = "LOG_MFBTU"
CLEAN_FILE = "cbecs_2018_clean.parquet"

SCORING = "r2"

# ---------------------------------------------------------------------------
# Dataclasses for metrics
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    model: str
    split: str
    r2: float
    mse: float
    sse: float
    n: int

    def as_dict(self) -> Dict[str, float]:
        d = asdict(self)
        # ensure JSON-serialisable primitives
        for k, v in list(d.items()):
            if isinstance(v, (np.generic,)):
                d[k] = float(v)
        return d


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------


def load_dataset(
    proc_dir: Path = PROC_DIR,
    filename: str = CLEAN_FILE,
    target: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the cleaned parquet produced by the earlier pipeline stage."""
    path = proc_dir / filename
    print(f"[STEP] Loading processed dataset from: {path}")
    df = pd.read_parquet(path)
    print(f"[INFO] Loaded dataset with shape {df.shape}")

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in data.")
    print(f"[INFO] Using target column: '{target}'")

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Mirror the 80 / 10 / 10 split used in the notebook."""
    print("[STEP] Creating train/val/test split (80/10/10)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # Validation size is relative to remaining pool
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_rel,
        random_state=random_state,
        shuffle=True,
    )

    check_shape = len(X_train) + len(X_val) + len(X_test) == len(X)
    if not check_shape:
        raise RuntimeError("Train/val/test split sizes do not add up to full dataset.")

    print(
        f"[INFO] Split sizes -> "
        f"train: {len(X_train)} rows, val: {len(X_val)} rows, test: {len(X_test)} rows"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer used in the notebook:

    * numeric: SimpleImputer(median, add_indicator=True) + StandardScaler
    * categorical: OneHotEncoder(handle_unknown="ignore")
    * dtype-based selectors, no hard-coded column names
    """
    print("[STEP] Building preprocessing pipeline (numeric + categorical)...")
    num_pipe = Pipeline(
        steps=[
            ("median_imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, selector(dtype_include=np.number)),
            ("cat", cat_pipe, selector(dtype_include=["object", "category", "bool"])),
        ],
        remainder="drop",
    )
    print("[INFO] Preprocessor built.")
    return preproc


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _baseline_mean(y_train: np.ndarray, y_eval: np.ndarray, split: str) -> Metrics:
    yhat = np.full_like(y_eval, fill_value=float(np.mean(y_train)), dtype=float)
    resid = y_eval - yhat
    sse = float(np.sum(resid**2))
    mse = float(sse / len(y_eval))
    r2 = float(r2_score(y_eval, yhat))
    return Metrics(model="baseline_mean", split=split, r2=r2, mse=mse, sse=sse, n=len(y_eval))


def _col2d(X: pd.DataFrame, feature: str) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        if feature not in X.columns:
            raise KeyError(f"Column '{feature}' not found in X.")
        return X[[feature]].to_numpy()
    X = np.asarray(X)
    raise TypeError("For ndarray X, this helper expects a pandas DataFrame input.")


def baseline_with_sqft(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    feature: str = "SQFT_log1p",
    split: str = "val",
) -> Tuple[Metrics, Metrics]:
    """
    Simple interpretable baselines:

    * baseline_mean: predict mean(y_train)
    * LR[feature]: single-feature LinearRegression on `feature`
    """
    print(f"[STEP] Computing baselines on {split} split (mean + LR[{feature}])...")
    # mean baseline
    mean_metrics = _baseline_mean(np.asarray(y_train), np.asarray(y_eval), split=split)

    # single-feature LR
    Xtr_col = _col2d(X_train, feature)
    Xev_col = _col2d(X_eval, feature)

    lr = LinearRegression().fit(Xtr_col, y_train)
    yhat = lr.predict(Xev_col)

    resid = y_eval - yhat
    sse = float(np.sum(resid**2))
    mse = float(sse / len(y_eval))
    r2 = float(r2_score(y_eval, yhat))

    lr_metrics = Metrics(
        model=f"LR[{feature}]",
        split=split,
        r2=r2,
        mse=mse,
        sse=sse,
        n=len(y_eval),
    )

    print(
        f"[INFO] Baseline metrics ({split}) -> "
        f"mean R²={mean_metrics.r2:.4f}, LR[{feature}] R²={lr_metrics.r2:.4f}"
    )

    return mean_metrics, lr_metrics


# ---------------------------------------------------------------------------
# Lasso – nested CV for alpha, final refit
# ---------------------------------------------------------------------------


def fit_final_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preproc: ColumnTransformer,
    random_state: int = 1,
) -> Pipeline:
    """
    Replicates the nested CV strategy from the notebook:

    * outer CV: RepeatedKFold
    * inner CV: KFold inside LassoCV
    * collect alpha_ from each outer estimator, take median
    * refit a fresh Lasso pipeline on full train using that alpha
    """
    print("[STEP] Starting nested CV for Lasso alpha selection...")
    # Outer CV (repeated for stability)
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)

    # Inner CV for alpha
    inner = KFold(n_splits=5, shuffle=True, random_state=random_state)

    lasso_nested = Pipeline(
        steps=[
            ("preproc", clone(preproc)),
            ("model", LassoCV(cv=inner, n_jobs=-1, max_iter=200_000, tol=1e-4)),
        ]
    )

    out = cross_validate(
        lasso_nested,
        X_train,
        y_train,
        cv=rkf,
        scoring=SCORING,
        n_jobs=-1,
        return_estimator=True,
    )

    alphas = [est.named_steps["model"].alpha_ for est in out["estimator"]]
    alpha_star = float(np.median(alphas))
    print(f"[INFO] Nested CV complete. Selected alpha (median): {alpha_star:.6f}")

    print("[STEP] Fitting final Lasso model on full training data...")
    final_lasso = Pipeline(
        steps=[
            ("preproc", clone(preproc)),
            (
                "model",
                Lasso(alpha=alpha_star, max_iter=200_000, tol=1e-4, selection="cyclic"),
            ),
        ]
    )
    final_lasso.fit(X_train, y_train)
    print("[INFO] Final Lasso model fitted.")
    return final_lasso


# ---------------------------------------------------------------------------
# XGBoost – Optuna-tuned wrapper (optional)
# ---------------------------------------------------------------------------


def build_xgb_model(preproc: ColumnTransformer, params: Dict) -> Pipeline:
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not installed; cannot build XGB model.")

    xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=1,
        n_jobs=-1,
        **params,
    )

    return Pipeline([("preproc", clone(preproc)), ("model", xgb)])


def tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preproc: ColumnTransformer,
    n_trials: int = 40,
    random_state: int = 1,
) -> Dict:
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not installed; cannot tune XGB.")

    print(f"[STEP] Running Optuna tuning for XGB ({n_trials} trials)...")
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 600, 1400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        }

        pipe = build_xgb_model(preproc, params)
        scores = cross_val_score(pipe, X_train, y_train, cv=rkf, scoring=SCORING, n_jobs=-1)
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_trial.params
    print(f"[INFO] XGB tuning complete. Best {SCORING}={study.best_value:.4f}")
    print("[INFO] Best XGB params:")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")

    with open(ARTIFACTS / "xgb_optuna_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"[INFO] Saved XGB best params to {ARTIFACTS / 'xgb_optuna_best_params.json'}")

    return best_params


def get_xgb_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preproc: ColumnTransformer,
    run_tuning: bool,
) -> Dict:
    params_path = ARTIFACTS / "xgb_optuna_best_params.json"
    if run_tuning or not params_path.exists():
        if run_tuning:
            print("[XGB] Tuning requested via flag.")
        else:
            print("[XGB] No cached params found; tuning required.")
        print("[XGB] Running Optuna tuning...")
        return tune_xgb(X_train, y_train, preproc)
    else:
        print(f"[XGB] Loading tuned params from {params_path}")
        with open(params_path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def eval_model(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    split: str,
) -> Metrics:
    print(f"[STEP] Evaluating model '{model_name}' on {split} split...")
    yhat = model.predict(X)
    resid = y - yhat
    sse = float(np.sum(resid**2))
    mse = float(mean_squared_error(y, yhat))
    r2 = float(r2_score(y, yhat))
    print(f"[INFO] {model_name} ({split}) -> R²={r2:.4f}, MSE={mse:.4f}")
    return Metrics(model=model_name, split=split, r2=r2, mse=mse, sse=sse, n=len(y))


def summarise_metrics(metrics: Iterable[Metrics]) -> pd.DataFrame:
    print("[STEP] Summarising metrics across models and splits...")
    rows = [m.as_dict() for m in metrics]
    df = pd.DataFrame(rows).sort_values(["model", "split"]).reset_index(drop=True)
    print("[INFO] Metrics summary constructed.")
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_training(
    run_xgb_tuning: bool = False,
) -> pd.DataFrame:
    """Main entry point used by the CLI and pipeline orchestration."""
    print("=== Training pipeline started ===")
    print("Loading data...")
    X, y = load_dataset()

    print("Splitting into train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    print("Building preprocessor...")
    preproc = build_preprocessor()

    metrics: list[Metrics] = []

    # ------------------------------------------------------------------
    # Baselines (mean + single-feature LR on log-sqft)
    # ------------------------------------------------------------------
    print("=== Fitting baseline models ===")
    m_val_mean, m_val_lr = baseline_with_sqft(
        X_train,
        y_train,
        X_val,
        y_val,
        feature="SQFT_log1p",
        split="val",
    )
    m_test_mean, m_test_lr = baseline_with_sqft(
        X_train,
        y_train,
        X_test,
        y_test,
        feature="SQFT_log1p",
        split="test",
    )
    metrics.extend([m_val_mean, m_val_lr, m_test_mean, m_test_lr])

    # ------------------------------------------------------------------
    # Lasso
    # ------------------------------------------------------------------
    print("=== Fitting final Lasso pipeline (nested CV for alpha) ===")
    final_lasso = fit_final_lasso(X_train, y_train, preproc)
    lasso_path = ARTIFACTS / "model_lasso.joblib"
    joblib.dump(final_lasso, lasso_path)
    print(f"[INFO] Saved Lasso model to: {lasso_path}")

    metrics.append(eval_model("lasso_final", final_lasso, X_val, y_val, split="val"))
    metrics.append(eval_model("lasso_final", final_lasso, X_test, y_test, split="test"))

    # ------------------------------------------------------------------
    # XGBoost (optional; requires xgboost installed)
    # ------------------------------------------------------------------
    if XGBRegressor is not None:
        try:
            print("=== Preparing XGB pipeline ===")
            xgb_params = get_xgb_params(X_train, y_train, preproc, run_tuning=run_xgb_tuning)
            final_xgb = build_xgb_model(preproc, xgb_params)
            print("[STEP] Fitting XGB pipeline on training data...")
            final_xgb.fit(X_train, y_train)

            xgb_path = ARTIFACTS / "model_xgb.joblib"
            joblib.dump(final_xgb, xgb_path)
            print(f"[INFO] Saved XGB model to: {xgb_path}")

            metrics.append(eval_model("xgb_final", final_xgb, X_val, y_val, split="val"))
            metrics.append(eval_model("xgb_final", final_xgb, X_test, y_test, split="test"))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Skipping XGB training due to error: {exc!r}")
    else:
        print("[INFO] xgboost not installed – skipping XGB model.")

    # ------------------------------------------------------------------
    # Persist metrics
    # ------------------------------------------------------------------
    print("=== Saving metrics ===")
    df_metrics = summarise_metrics(metrics)

    metrics_csv = ARTIFACTS / "model_metrics.csv"
    metrics_json = ARTIFACTS / "model_metrics.json"

    df_metrics.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w") as f:
        json.dump(df_metrics.to_dict(orient="records"), f, indent=2)

    print(f"[INFO] Metrics written to:\n  - {metrics_csv}\n  - {metrics_json}")

    print("=== Training complete. Final metrics ===")
    print(df_metrics)

    return df_metrics


def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate models for the CBECS energy project.")
    parser.add_argument(
        "--run-xgb-tuning",
        action="store_true",
        help="Run Optuna tuning for XGBoost (otherwise load cached params if available).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_training(run_xgb_tuning=args.run_xgb_tuning)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())