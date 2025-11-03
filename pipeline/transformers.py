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


