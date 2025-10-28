import pathlib
import yaml
import requests
import warnings
import pandas as pd
import numpy as np

# Download and write the file...
# Import config

CFG = yaml.safe_load(open("conf/config.yaml"))

# Get raw directory
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
data = pd.read_excel(OUT_PATH_DATA, 
                        engine="openpyxl", 
                        skiprows=1,
                        skipfooter=1, 
                        names = ["variable_order",	
                                 "name",
                                 "type", 
                                 "description", 
                                 "values", 
                                 "Question_text"])


def split_numeric_categorical(data: pd.DataFrame):
    # get list of values with numeric ranges. they all start with ### – ###
    mask = data["values"].str.match("\d.* – \d*.*", na=True)
    mask = (mask) & (data.type !="Char")
    return data.loc[mask, "name"].tolist(), data.loc[~mask, "name"].tolist()



# grab the list of all numeric column names
numeric_names, categorical_names = split_numeric_categorical(data)

yaml_data = {'numeric_variables': numeric_names,
             'categorical_variables': categorical_names
             }

yaml_file_path = "conf/variable_types.yaml"

with open(yaml_file_path, 'w') as file:
    yaml.dump(yaml_data, file, default_flow_style=False)

print(f"YAML file saved at {yaml_file_path}")
breakpoint()


# Now let's clean the data...
