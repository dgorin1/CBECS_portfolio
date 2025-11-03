# pipeline/01_download.py
import pathlib
import requests
import yaml

# --- load config ---
# 1) Where to put the file (relative to your repo root)
CFG = yaml.safe_load(open("conf/config.yaml"))

# Create a filepath
RAW_DIR = pathlib.Path(CFG["data"]["raw_dir"])

# Make directory if doesn't exist and name the output raw file for 2018
RAW_DIR.mkdir(parents=True, exist_ok=True)



year = int(CFG["datasets"]["year"])

OUT_PATH = RAW_DIR / f"cbecs_{year}_microdata.csv"

# Get CSV URL from yaml
CSV_URL = CFG["datasets"]["raw_csv_url"]
print(f"Downloading {CSV_URL} -> {OUT_PATH}...")

# download data..

dataset = requests.get(CSV_URL)
with open(OUT_PATH, 'wb') as f:
    f.write(dataset.content)

print("Done downloading data!")
