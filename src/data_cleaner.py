import pandas as pd
import numpy as np

def check_missing_data(df, show_top=15):
    """
    檢查資料缺失狀況
    
    Parameters:
    df (DataFrame): 要檢查的資料框
    show_top (int): 顯示前 N 個缺失最嚴重的欄位
    
    Returns:
    DataFrame: 缺失值報告
    """
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        '缺失數量': missing, 
        '缺失比例%': missing_pct.round(2)
    })
    
    print(f"🕳️ 缺失值分析:")
    print(f"有缺失值的欄位數量: {len(missing_df)}")
    print(f"總缺失值數量: {df.isnull().sum().sum()}")
    
    if len(missing_df) > 0:
        print(f"\n缺失最嚴重的 {show_top} 個欄位:")
        display(missing_df.head(show_top))
    
    return missing_df

def remove_high_missing_columns(df, threshold=80):
    """
    移除缺失值過高的欄位
    
    Parameters:
    df (DataFrame): 原始資料
    threshold (float): 缺失比例閾值 (%)
    
    Returns:
    DataFrame: 清理後的資料
    """
    missing_pct = (df.isnull().sum() / len(df)) * 100
    columns_to_drop = missing_pct[missing_pct > threshold].index
    
    if len(columns_to_drop) > 0:
        print(f"🗑️ 移除 {len(columns_to_drop)} 個缺失超過 {threshold}% 的欄位:")
        for col in columns_to_drop:
            print(f"  - {col}: {missing_pct[col]:.1f}%")
        
        df_clean = df.drop(columns=columns_to_drop)
        print(f"清理後資料形狀: {df_clean.shape}")
        return df_clean
    else:
        print("✅ 沒有需要移除的高缺失值欄位")
        return df