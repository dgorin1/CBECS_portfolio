# pipeline/transformers.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Any



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
            if rule['when'] == "NA":
                X.loc[:, rule["columns"]] = X.loc[:, rule["columns"]].fillna(rule["set_to"])
               
            else:
                mask = X[rule["columns"]] == int(rule['when'])
                X[rule["columns"]] = X[rule["columns"]].mask(X[rule["columns"]] == int(rule["when"]), int(rule["set_to"]))
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
        flags_to_drop = X.filter(regex=self.regex_for_delete, axis=1).columns.tolist()
        X = X.drop(self.cols_to_drop + flags_to_drop, axis=1)
        return X

class MedianImputer(BaseEstimator, TransformerMixin):
    """Impute Median and add flag column"""

    def __init__(self, impute_rules):
        self.impute_rules = impute_rules
        self.columns_to_inpute = impute_rules.keys()

    def fit(self, X, y=None):
        self.impute_cols = [c for c in self.columns_to_inpute if c in X.columns]
        self.medians = X[self.impute_cols].median().to_dict()
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        for col in self.impute_cols:
            # If we're imputing NaNs
            if self.impute_rules[col]['value'] == 'NA':
                X[col] = X[col].fillna(self.medians[col])
            else:
                # If we're imputing another flag...
                mask = X[col] == self.impute_rules[col]['value']
                X[col] = X[col].mask(mask, self.medians[col])

            # Create flag column
            X[f"{col}_median_imputed"] = mask.astype(int)

        return X
        
    

