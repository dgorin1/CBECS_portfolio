import yaml
import pandas as pd
import pathlib

# --- Read in the config file ---
# Load the configuration settings from the YAML file
CFG = yaml.safe_load(open("conf/config.yaml"))

# Extract the datasets and directories from the config
datasets = CFG["datasets"]
RAW_DIR = CFG["data"]["raw_dir"]

# Initialize an empty dictionary to store DataFrames for each year
data_frames = {}

# --- Read in datasets ---
for dataset in datasets:
    year = dataset["year"]  # Get the year for the current dataset (e.g., 2018 or 2012)
    
    # Construct the path to the raw data file based on the dataset year
    raw_file = pathlib.Path(CFG["data"]["raw_dir"]) / f"cbecs_{year}_microdata.csv"

    # Check if the file exists at the specified path
    if raw_file.exists():
        # If the file exists, read it into a pandas DataFrame and store it in the dictionary
        df = pd.read_csv(raw_file)
        data_frames[year] = df
    else:
        # If the file is not found, print a warning message
        print(f"Warning: {raw_file} not found.")

# --- Columns to keep (from the config) ---
# Extract the lists of categorical and numeric columns from the config
keep_categorical = CFG["keep_categorical"]
keep_numeric = CFG["keep_numeric"]

# Initialize an empty list to store the selected DataFrames
df_selected_list = []

# --- Select relevant columns from each dataset ---
for year, df in data_frames.items():
    # Combine the categorical and numeric columns into one list
    columns_to_keep = keep_categorical + keep_numeric

    # Ensure that only columns that exist in the current dataset are selected
    columns_to_select = [col for col in columns_to_keep if col in df.columns]

    # Select the relevant columns from the DataFrame based on the column list
    df_selected = df[columns_to_select]
    
    # Add the 'year' column to the DataFrame for tracking which dataset the row came from
    df_selected["year"] = year
    
    # Append the processed DataFrame to the list
    df_selected_list.append(df_selected)

    # Print the selected columns for verification
    print(f"Selected columns for {year}: {df_selected.columns.tolist()}")

# --- Merge all DataFrames into one large DataFrame ---
# Concatenate the DataFrames in the list into a single DataFrame (ignoring the index)
final_df = pd.concat(df_selected_list, ignore_index=True)

# Print the first few rows of the final merged DataFrame to verify everything looks correct
print(final_df.head())

