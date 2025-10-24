# src/features.py
from __future__ import annotations
import re
import pandas as pd

class CBECSCleaner:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _drop_target_na(self, df: pd.DataFrame) -> pd.DataFrame:
        tgt = self.cfg["target"]["name"]
        if self.cfg["target"].get("drop_na", True):
            return df.dropna(subset=[tgt])
        return df

    def _add_log_target(self, df: pd.DataFrame) -> pd.DataFrame:
        tgt = self.cfg["target"]["name"]
        log = self.cfg["target"]["log_name"]
        if tgt in df:
            df[log] = (df[tgt]).apply(lambda x: pd.NA if pd.isna(x) or x <= 0 else None)
            df[log] = (df[tgt]).where(df[tgt] > 0).pipe(lambda s: s.apply(pd.to_numeric, errors="coerce")).map(lambda v: None if pd.isna(v) else __import__("math").log(v))
            df[log] = pd.to_numeric(df[log])
        return df

    def _fillna_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        for val_key, value in [("to_0", 0), ("to_1", 1), ("to_2", 2)]:
            cols = self.cfg.get("fillna", {}).get(val_key, [])
            cols = [c for c in cols if c in df.columns]
            if cols:
                df[cols] = df[cols].fillna(value)
        return df

    def _single_value_recode(self, df: pd.DataFrame) -> pd.DataFrame:
        spec = self.cfg.get("recode", {}).get("single_value", {})
        for col, rule in spec.items():
            if col in df.columns:
                sentinel = rule["sentinel"]; repl = rule["replace"]
                flag = f"{col}_TOPCODED"
                df[flag] = (df[col] == sentinel).astype("Int64")
                df.loc[df[col] == sentinel, col] = repl
        return df

    def _nfloor_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        conf = self.cfg.get("recode", {}).get("nfloor")
        if not conf or "NFLOOR" not in df.columns: 
            return df
        mid, top = conf["mid_code"], conf["top_code"]
        mid_rep, top_rep = conf["mid_replace"], conf["top_replace"]
        if conf.get("add_bins", True):
            df["NFLOOR_10_14_BIN"] = (df["NFLOOR"] == mid).astype("Int64")
            df["NFLOOR_TOP_BIN"]   = (df["NFLOOR"] == top).astype("Int64")
        df["NFLOOR"] = df["NFLOOR"].replace({mid: mid_rep, top: top_rep})
        return df

    def _one_off_fixes(self, df: pd.DataFrame) -> pd.DataFrame:
        rec = self.cfg.get("one_off_fixes", {}).get("recode_equal", {})
        for col, mapping in rec.items():
            if col in df.columns:
                df[col] = df[col].replace(mapping)
        return df

    def _impute_median_with_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.cfg.get("impute_median_with_flag", []) if c in df.columns]
        for col in cols:
            m = df[col].isna()
            df[f"{col}_IMPUTED"] = m.astype("Int64")
            med = df.loc[~m, col].median()
            df.loc[m, col] = med
        return df

    def _hcbeds_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        rule = self.cfg.get("domain_rules", {}).get("hcbeds")
        if not rule: return df
        col = rule["col"]; pba = rule["pba_col"]; vals = set(rule["applicable_values"])
        if col not in df.columns or pba not in df.columns: 
            return df
        df[col] = pd.to_numeric(df[col], errors="coerce")
        app = df[pba].isin(vals)
        sentinel = self.cfg["recode"]["single_value"]["HCBED"]["sentinel"]
        cutoff   = self.cfg["recode"]["single_value"]["HCBED"]["replace"]
        top = df[col].eq(sentinel)
        na  = df[col].isna()
        df[f"{col}_TOPCODED"] = top.astype("Int64")
        df.loc[top, col] = cutoff
        df[f"{col}_APPLICABLE"] = app.astype("Int64")
        df.loc[~app & na, col] = 0
        df[f"{col}_MISSING_IF_APPL"] = (app & na).astype("Int64")
        return df

    def _engineer_seats(self, df: pd.DataFrame) -> pd.DataFrame:
        eng = self.cfg.get("engineer", {}).get("seats")
        if not eng: return df
        cols = [c for c in eng["cols"] if c in df.columns]
        if cols:
            df[eng["out_total"]] = df[cols].sum(axis=1, min_count=1)
        flags = [f for f in eng.get("keep_top_flags", []) if f in df.columns]
        if flags:
            df[eng["out_top_any"]] = df[flags].max(axis=1)
            df.drop(columns=[c for c in cols+flags if c in df.columns], inplace=True, errors="ignore")
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drops = self.cfg.get("drops", {})
        # leakage + obvious
        to_drop = [c for c in drops.get("leakage_risks", []) + drops.get("obvious", []) if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
        # regex weights
        pat = drops.get("weight_regex")
        if pat:
            weight_cols = [c for c in df.columns if re.fullmatch(pat, c)]
            if weight_cols:
                df = df.drop(columns=weight_cols)
        return df

    def _drop_high_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        thr = self.cfg.get("post", {}).get("drop_missing_threshold")
        if thr is None: 
            return df
        keep_count = int((1 - thr) * len(df))  # e.g., thr=0.9 -> keep if >=10% non-null
        return df.dropna(axis=1, thresh=keep_count)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._drop_target_na(df)
        df = self._add_log_target(df)
        df = self._fillna_groups(df)
        df = self._single_value_recode(df)
        df = self._nfloor_rules(df)
        df = self._one_off_fixes(df)
        df = self._impute_median_with_flag(df)
        df = self._hcbeds_rule(df)
        df = self._engineer_seats(df)
        df = self._drop_columns(df)
        df = self._drop_high_missing(df)
        return df