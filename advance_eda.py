
# ==========================================
# ADVANCED EDA SCRIPT: advance_eda.py
# ==========================================
# Required Libraries:
# pip install pandas numpy matplotlib seaborn statsmodels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def eda_by_ai(df):
    """
    Performs comprehensive Advanced Exploratory Data Analysis (EDA) on an already loaded pandas DataFrame.
    Covers data cleaning, descriptive stats, univariate, bivariate, time-series, and multivariate analysis.
    """
    
    # Set general plotting aesthetics
    sns.set_theme(style="whitegrid")
    
    print("--- 1. DATA PREPROCESSING & CLEANING ---")
    
    # Make a copy to avoid modifying the original dataframe directly
    df = df.copy()
    
    # Clean column names: strip whitespace, convert to lowercase, replace spaces with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print(f"Cleaned Columns: {list(df.columns)}")
    
    # Automatically detect and parse date/timestamp columns
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Try parsing as datetime if it looks like a date string
                parsed_dates = pd.to_datetime(df[col], errors='coerce')
                if parsed_dates.notnull().sum() > 0.5 * len(df): # If > 50% successfully parsed
                    df[col] = parsed_dates
                    date_cols.append(col)
                    print(f"Successfully parsed '{col}' as datetime.")
            except Exception:
                pass
                
    # Handle missing values gracefully
    missing_counts = df.isnull().sum()
    print("\nMissing values per column:\n", missing_counts[missing_counts > 0])
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Imputed missing values in numerical column '{col}' with median: {median_val}")
            else:
                mode_val = df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "Missing"
                df[col].fillna(fill_val, inplace=True)
                print(f"Imputed missing values in categorical column '{col}' with mode/constant: {fill_val}")

    print("\n--- 2. PHASE 1: DESCRIPTIVE STATISTICS & CORRELATION ---")
    
    # Comprehensive descriptive statistics for both numerical and categorical columns
    desc_stats = df.describe(include='all')
    print("\nComprehensive Descriptive Statistics:")
    print(desc_stats)
    
    # Correlation Matrix for numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        print("\nComputing Pearson Correlation Matrix...")
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Pearson Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        print("\nSkipping Correlation Matrix: Insufficient numerical columns available.")

    print("\n--- 3. PHASE 2: UNIVARIATE ANALYSIS ---")
    
    # Numerical Columns: Histogram with KDE curve and Boxplot
    for col in numeric_df.columns:
        print(f"Generating univariate plots for numerical column: '{col}'")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram + KDE
        sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'Histogram & KDE of {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Density/Count')
        
        # Boxplot
        sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
        axes[1].set_title(f'Boxplot of {col}')
        axes[1].set_xlabel(col)
        
        plt.tight_layout()
        plt.show()

    # Categorical/Object Columns: Count Plots (Limited to top 15 categories)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        print(f"Generating univariate count plot for categorical column: '{col}'")
        plt.figure(figsize=(12, 6))
        
        # Handle high cardinality by keeping top 15
        top_categories = df[col].value_counts().nlargest(15).index
        filtered_data = df[df[col].isin(top_categories)]
        
        sns.countplot(data=filtered_data, x=col, order=top_categories, palette='viridis')
        plt.title(f'Count Plot of Top Categories in {col}', fontsize=14, fontweight='bold')
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    print("\n--- 4. PHASE 3: BIVARIATE ANALYSIS & TIME SERIES ---")
    
    # Bivariate Analysis: Target primary numerical column ('sales' if present) against categorical attributes ('region', 'segment')
    target_num = 'sales' if 'sales' in numeric_df.columns else (numeric_df.columns[0] if not numeric_df.empty else None)
    target_cats = [c for c in ['region', 'segment'] if c in df.columns]
    
    if target_num and target_cats:
        for cat_col in target_cats:
            print(f"Analyzing relationship between numerical '{target_num}' and categorical '{cat_col}'")
            grouped_stats = df.groupby(cat_col)[target_num].agg(['mean', 'sum', 'median', 'count']).reset_index()
            print(f"\nGrouped Summary Statistics for '{target_num}' by '{cat_col}':")
            print(grouped_stats)
            
            plt.figure(figsize=(10, 5))
            sns.barplot(data=df, x=cat_col, y=target_num, estimator=np.mean, ci=None, palette='muted')
            plt.title(f'Mean {target_num.capitalize()} by {cat_col.capitalize()}', fontsize=14, fontweight='bold')
            plt.xlabel(cat_col, fontsize=12)
            plt.ylabel(f'Mean {target_num}', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    # Time Series Analysis (Conditional Execution)
    if date_cols and target_num:
        date_col = date_cols[0]
        print(f"\nTime Series Analysis detected using date column: '{date_col}' and metric '{target_num}'")
        
        # Create a temporal aggregation (e.g., Monthly or Yearly)
        ts_df = df.set_index(date_col).resample('M')[target_num].sum().reset_index()
        
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=ts_df, x=date_col, y=target_num, marker='o', color='b', linewidth=2)
        plt.title(f'Temporal Trend of {target_num.capitalize()} Over Time (Monthly)', fontsize=14, fontweight='bold')
        plt.xlabel('Date (Month)', fontsize=12)
        plt.ylabel(f'Total {target_num.capitalize()}', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("\nSkipping Time Series Analysis: No valid date column found.")

    print("\n--- 5. PHASE 4: MULTIVARIATE ANALYSIS (BAR PLOTS WITH HUE) ---")
    
    # Perform deep multivariate analysis incorporating 'sales', 'region', and 'segment' (or fallback variables)
    multi_num = 'sales' if 'sales' in df.columns else (numeric_df.columns[0] if not numeric_df.empty else None)
    multi_cats = [c for c in ['region', 'segment'] if c in df.columns]
    
    if multi_num and len(multi_cats) >= 2:
        cat1, cat2 = multi_cats[0], multi_cats[1]
        print(f"Generating Multivariate Bar Plot: x='{cat1}', y='{multi_num}', hue='{cat2}'")
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x=cat1, y=multi_num, hue=cat2, estimator=np.mean, ci=None, palette='Set2')
        plt.title(f'Multivariate Analysis: Mean {multi_num.capitalize()} by {cat1.capitalize()} and {cat2.capitalize()}', fontsize=14, fontweight='bold')
        plt.xlabel(cat1.capitalize(), fontsize=12)
        plt.ylabel(f'Mean {multi_num.capitalize()}', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=cat2.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    elif multi_num and len(multi_cats) == 1 and len(categorical_cols) >= 2:
        # Fallback to alternative categorical columns if exact names aren't both present
        cat1 = multi_cats[0]
        other_cats = [c for c in categorical_cols if c != cat1]
        cat2 = other_cats[0]
        
        print(f"Generating Multivariate Bar Plot (Fallback): x='{cat1}', y='{multi_num}', hue='{cat2}'")
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x=cat1, y=multi_num, hue=cat2, estimator=np.mean, ci=None, palette='Set2')
        plt.title(f'Multivariate Analysis: Mean {multi_num.capitalize()} by {cat1.capitalize()} and {cat2.capitalize()}', fontsize=14, fontweight='bold')
        plt.xlabel(cat1.capitalize(), fontsize=12)
        plt.ylabel(f'Mean {multi_num.capitalize()}', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=cat2.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    else:
        print("\nSkipping Multivariate Bar Plot: Required combination of numerical and categorical variables not found.")
        
    print("\n--- EDA PIPELINE COMPLETE ---")
