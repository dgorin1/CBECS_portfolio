import pathlib
import yaml
import requests
import pandas as pd

# Load the configuration from a YAML file
CFG = yaml.safe_load(open("conf/config.yaml"))

# Define the raw directory to store downloaded data
RAW_DIR = pathlib.Path(CFG["data"]["raw_dir"])
RAW_DIR.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Define the path to save the downloaded codebook data
OUT_PATH_DATA = RAW_DIR / f"codebook_2018.xlsx"

# Get the URL of the codebook from the configuration
CSV_URL = CFG["datasets"]["codebook_url"]
print(f"Downloading {CSV_URL} -> {OUT_PATH_DATA}...")

# Download the codebook from the URL and save it to the file path
code_book = requests.get(CSV_URL)
with open(OUT_PATH_DATA, 'wb') as f:
    f.write(code_book.content)

# Read the codebook into a pandas DataFrame
# Skipping the first row (header) and the last row (footer) as specified in the original file
data = pd.read_excel(
    OUT_PATH_DATA, 
    engine="openpyxl", 
    skiprows=1,         # Skip the first row (header)
    skipfooter=1,       # Skip the last row (footer)
    names=["variable_order", "name", "type", "description", "values", "Question_text"]
)

# Function to split the data into numeric and categorical variables based on the "values" column
def split_numeric_categorical(data: pd.DataFrame):
    """
    This function splits the variables into numeric and categorical based on the 'values' column.
    Numeric variables are identified by a range pattern (e.g., '10 – 20').
    
    Parameters:
    - data: DataFrame containing the codebook data
    
    Returns:
    - A tuple of lists: (numeric_variables, categorical_variables)
    """
    # Create a boolean mask for rows where 'values' matches a numeric range pattern (e.g., '10 – 20')
    mask = data["values"].str.match(r"\d.*\s*[–—-]\s*\d.*", na=True)
    
    # Further refine the mask: Exclude rows with type 'Char' since they're not numeric
    mask = mask & (data["type"] != "Char")
    
    # Return the names of variables that are numeric and categorical
    numeric_vars = data.loc[mask, "name"].tolist()  # Variables that match the numeric pattern
    categorical_vars = data.loc[~mask, "name"].tolist()  # Variables that do not match the numeric pattern
    
    return numeric_vars, categorical_vars

# Get lists of numeric and categorical variable names
numeric_names, categorical_names = split_numeric_categorical(data)

# Ensure PKLTN is treated as categorical
if "PKLTN" in categorical_names:
    categorical_names.remove("PKLTN")
if "PKLTN" not in numeric_names:
    numeric_names.append("PKLTN")

# Exclude PUBID entirely from both lists
numeric_names = [n for n in numeric_names if n != "PUBID"]
categorical_names = [n for n in categorical_names if n != "PUBID"]

# Prepare the data to be saved in YAML format
yaml_data = {
    'numeric_variables': numeric_names,
    'categorical_variables': categorical_names
}

# Define the path for the output YAML file
yaml_file_path = "conf/variable_types.yaml"

# Write the variable types to the YAML file
with open(yaml_file_path, 'w') as file:
    yaml.dump(yaml_data, file, default_flow_style=False)

print(f"YAML file saved at {yaml_file_path}")