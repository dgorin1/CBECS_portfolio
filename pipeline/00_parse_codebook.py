import pathlib
import yaml
import requests
import warnings
import pandas as pd


# Download and write the file...
# Import config
CFG = yaml.safe_load(open("conf/config.yaml"))

RAW_DIR =  pathlib.Path(CFG["data"]["raw_dir"])
RAW_DIR.mkdir(parents=True, exist_ok=True)
# Download the file...
OUT_PATH_DATA = RAW_DIR / f"codebook_2018.xlsx"

# Get CSV URL from yaml
CSV_URL = CFG["datasets"]["codebook_url"]
print(f"Downloading {CSV_URL} -> {OUT_PATH_DATA}...")


code_book = requests.get(CSV_URL)

with open(OUT_PATH_DATA, 'wb') as f:
    f.write(code_book.content)


# Read in and begin to parse the dataset...
dataset = pd.read_excel(OUT_PATH_DATA, engine='openpyxl')
breakpoint()