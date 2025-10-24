# pipeline/02_build_dataset.py
from __future__ import annotations
import pathlib, yaml, pandas as pd
from src.features import CBECSCleaner

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "conf" / "config.yaml"))
RAW_DIR = ROOT / CFG["data"]["raw_dir"]
PROCESSED_DIR = ROOT / CFG["data"]["processed_dir"]
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load just 2018 for now (as you decided)
year = 2018
raw_path = RAW_DIR / f"cbecs_{year}_microdata.csv"
df = pd.read_csv(raw_path)

# Apply cleaning/engineering
cleaner = CBECSCleaner(CFG)
df_proc = cleaner.transform(df)

# Save processed
out_csv = PROCESSED_DIR / f"cbecs_{year}_processed.csv"
df_proc.to_csv(out_csv, index=False)
print(f"✅ Saved: {out_csv} (rows={len(df_proc)}, cols={df_proc.shape[1]})")