import pandas as pd
import numpy as np

def create_new_features(df):
    """
    建立新特徵
    
    Parameters:
    df (DataFrame): 原始資料
    
    Returns:
    DataFrame: 包含新特徵的資料
    """
    df_new = df.copy()
    
    # 房屋年齡
    if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
        df_new['HouseAge'] = df_new['YrSold'] - df_new['YearBuilt']
        print("✅ 新增特徵: HouseAge (房屋年齡)")
    
    # 總面積
    area_columns = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea']
    if all(col in df.columns for col in area_columns):
        df_new['TotalArea'] = df_new[area_columns].sum(axis=1)
        print("✅ 新增特徵: TotalArea (總面積)")
    
    # 是否有游泳池
    if 'PoolArea' in df.columns:
        df_new['HasPool'] = (df_new['PoolArea'] > 0).astype(int)
        print("✅ 新增特徵: HasPool (是否有游泳池)")
    
    # 衛浴總數
    bath_columns = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in df.columns for col in bath_columns):
        df_new['TotalBath'] = (df_new['FullBath'] + 
                              df_new['HalfBath'] * 0.5 + 
                              df_new['BsmtFullBath'] + 
                              df_new['BsmtHalfBath'] * 0.5)
        print("✅ 新增特徵: TotalBath (衛浴總數)")
    
    return df_new

def apply_log_transform(df, columns):
    """
    對指定欄位應用對數轉換
    
    Parameters:
    df (DataFrame): 原始資料
    columns (list): 要轉換的欄位列表
    
    Returns:
    DataFrame: 轉換後的資料
    """
    df_transformed = df.copy()
    
    for col in columns:
        if col in df.columns:
            # 確保沒有負值
            if (df[col] < 0).any():
                print(f"⚠️  {col} 包含負值，跳過對數轉換")
                continue
            
            df_transformed[f'log_{col}'] = np.log1p(df[col])
            print(f"✅ 對數轉換: {col} → log_{col}")
    
    return df_transformed