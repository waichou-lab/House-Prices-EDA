import pandas as pd
import numpy as np

def create_new_features(df):
    """
    Create new features
    """
    df_new = df.copy()
    
    # House age
    if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
        df_new['HouseAge'] = df_new['YrSold'] - df_new['YearBuilt']
        print("✅ New feature: HouseAge")
    
    # Total area
    area_columns = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea']
    if all(col in df.columns for col in area_columns):
        df_new['TotalArea'] = df_new[area_columns].sum(axis=1)
        print("✅ New feature: TotalArea")
    
    # Has pool
    if 'PoolArea' in df.columns:
        df_new['HasPool'] = (df_new['PoolArea'] > 0).astype(int)
        print("✅ New feature: HasPool")
    
    # Total bathrooms
    bath_columns = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in df.columns for col in bath_columns):
        df_new['TotalBath'] = (df_new['FullBath'] + 
                              df_new['HalfBath'] * 0.5 + 
                              df_new['BsmtFullBath'] + 
                              df_new['BsmtHalfBath'] * 0.5)
        print("✅ New feature: TotalBath")
    
    return df_new

def apply_log_transform(df, columns):
    """
    Apply log transformation to specified columns
    """
    df_transformed = df.copy()
    
    for col in columns:
        if col in df.columns:
            # Ensure no negative values
            if (df[col] < 0).any():
                print(f"⚠️  {col} contains negative values, skipping log transform")
                continue
            
            df_transformed[f'log_{col}'] = np.log1p(df[col])
            print(f"✅ Log transform: {col} → log_{col}")
    
    return df_transformed