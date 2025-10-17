#!/usr/bin/env bash
set -euo pipefail

python pipeline/01_download.py
python pipeline/02_build_dataset.py
python pipeline/03_train.py
python pipeline/04_evaluate.py
