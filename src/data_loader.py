import pandas as pd
import os

def load_dataset(data_path="../data"):
    """
    Load training and test datasets
    """
    try:
        train = pd.read_csv(f"{data_path}/train.csv")
        test = pd.read_csv(f"{data_path}/test.csv")
        print("✅ Datasets loaded successfully")
        return train, test
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return None, None

def get_data_info(train_df, test_df):
    """
    Display dataset basic information
    """
    print("📊 Dataset Information:")
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Training features: {len(train_df.columns)}")
    print(f"Test features: {len(test_df.columns)}")
    
    if 'SalePrice' in train_df.columns:
        print(f"Target variable range: ${train_df['SalePrice'].min():,} - ${train_df['SalePrice'].max():,}")