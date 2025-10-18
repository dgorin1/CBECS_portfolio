import pathlib
import json
import pandas as pd
import yaml

CFG = yaml.safe_load(open("conf/config.yaml"))
RAW_DIR = pathlib.Path(CFG["data"]["raw_dir"])
PROC_DIR = pathlib.path(CFG["data"]["processed_dir"])


breakpoint()