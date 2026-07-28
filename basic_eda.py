
import pandas as pd


def perform_eda(df: pd.DataFrame):
    """Performs basic Exploratory Data Analysis (EDA) on a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    """
    print("=" * 60)
    print(" 📊 EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 60)

    # 1. Dataset Shape
    print("\n[1] DATASET SHAPE")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")

    # 2. Columns & Data Types
    print("\n[2] COLUMNS AND DATA TYPES")
    dtype_df = pd.DataFrame(
        {
            "Data Type": df.dtypes,
            "Non-Null Count": df.notnull().sum(),
            "Null Count": df.isnull().sum(),
        }
    )
    print(dtype_df)

    # 3. Missing Values Summary
    print("\n[3] MISSING VALUES SUMMARY")
    missing_count = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame(
        {"Missing Values": missing_count, "Percentage (%)": missing_percentage}
    )
    missing_df = missing_df[missing_df["Missing Values"] > 0].sort_values(
        by="Missing Values", ascending=False
    )

    if missing_df.empty:
        print("🎉 Great news! There are no missing values in this dataset.")
    else:
        print(missing_df)

    # 4. Duplicate Rows
    print("\n[4] DUPLICATE ROWS")
    duplicates = df.duplicated().sum()
    print(
        f"Number of duplicate rows: {duplicates} ({(duplicates / len(df)) * 100:.2f}%)"
    )

    # 5. Statistical Summary (Numerical Features)
    print("\n[5] STATISTICAL SUMMARY (Numerical Columns)")
    num_df = df.select_dtypes(include=["number"])
    if not num_df.empty:
        print(num_df.describe().T)
    else:
        print("No numerical columns found in the dataset.")

    # 6. Statistical Summary (Categorical Features)
    print("\n[6] STATISTICAL SUMMARY (Categorical Columns)")
    cat_df = df.select_dtypes(include=["object", "category"])
    if not cat_df.empty:
        print(cat_df.describe().T)
    else:
        print("No categorical columns found in the dataset.")

    print("\n" + "=" * 60)
    print(" END OF EDA REPORT")
    print("=" * 60)
