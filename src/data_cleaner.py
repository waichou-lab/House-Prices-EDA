import pandas as pd
import numpy as np

def check_missing_data(df, show_top=15):
    """
    Check data missing status
    """
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing, 
        'Missing_Percent': missing_pct.round(2)
    })
    
    print(f"🔍 Missing Values Analysis:")
    print(f"Columns with missing values: {len(missing_df)}")
    print(f"Total missing values: {df.isnull().sum().sum()}")
    
    if len(missing_df) > 0:
        print(f"\nTop {show_top} columns with most missing values:")
        display(missing_df.head(show_top))
    
    return missing_df

def remove_high_missing_columns(df, threshold=80):
    """
    Remove columns with high missing values
    """
    missing_pct = (df.isnull().sum() / len(df)) * 100
    columns_to_drop = missing_pct[missing_pct > threshold].index
    
    if len(columns_to_drop) > 0:
        print(f"🗑️ Removing {len(columns_to_drop)} columns with > {threshold}% missing:")
        for col in columns_to_drop:
            print(f"  - {col}: {missing_pct[col]:.1f}%")
        
        df_clean = df.drop(columns=columns_to_drop)
        print(f"Cleaned data shape: {df_clean.shape}")
        return df_clean
    else:
        print("✅ No high missing columns to remove")
        return df