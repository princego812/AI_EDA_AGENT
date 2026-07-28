
import os
import pandas as pd


def read_uploaded_file(file_path):
    # Extract the file extension in lowercase (e.g., '.csv', '.xlsx')
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # Dictionary mapping extensions to their respective pandas read functions
    readers = {
        ".csv": pd.read_csv,
        ".tsv": lambda f: pd.read_csv(f, sep="\t"),
        ".xls": pd.read_excel,
        ".xlsx": pd.read_excel,
        ".xlsm": pd.read_excel,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
        ".feather": pd.read_feather,
        ".pkl": pd.read_pickle,
        ".pickle": pd.read_pickle,
    }

    if file_extension in readers:
        # Call the corresponding pandas function
        return readers[file_extension](file_path)
    else:
        raise ValueError(
            f"Unsupported file extension: '{file_extension}'. "
            f"Supported formats are: {list(readers.keys())}"
        )


# --- Example Usage ---
# df = read_uploaded_file("data.csv")
# print(df.head())
