import pandas as pd
import os

def load_dataset(data_path="../data"):
    """
    載入訓練和測試資料集
    
    Parameters:
    data_path (str): 資料檔案路徑
    
    Returns:
    tuple: (train_df, test_df)
    """
    try:
        train = pd.read_csv(f"{data_path}/train.csv")
        test = pd.read_csv(f"{data_path}/test.csv")
        print("✅ 資料集載入成功")
        return train, test
    except FileNotFoundError as e:
        print(f"❌ 找不到資料檔案: {e}")
        return None, None

def get_data_info(train_df, test_df):
    """
    顯示資料集基本資訊
    
    Parameters:
    train_df (DataFrame): 訓練資料
    test_df (DataFrame): 測試資料
    """
    print("📊 資料集資訊:")
    print(f"訓練集形狀: {train_df.shape}")
    print(f"測試集形狀: {test_df.shape}")
    print(f"訓練集欄位: {len(train_df.columns)}")
    print(f"測試集欄位: {len(test_df.columns)}")
    
    if 'SalePrice' in train_df.columns:
        print(f"目標變數範圍: ${train_df['SalePrice'].min():,} - ${train_df['SalePrice'].max():,}")