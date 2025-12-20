"""
期中 Project: 資料插補實作
House Prices 資料集插補處理
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

print("=" * 60)
print("期中 Project: 資料插補實作")
print("=" * 60)

# 1. 載入資料
print("\n1. 載入資料...")
try:
    df = pd.read_csv('data/train.csv')
    print(f"✓ 成功載入 train.csv")
    print(f"  資料形狀: {df.shape}")
    print(f"  原始缺失值: {df.isnull().sum().sum()} 個")
except FileNotFoundError:
    print("✗ 錯誤: 找不到 data/train.csv")
    print("  請確認:")
    print("  1. data 資料夾是否存在")
    print("  2. train.csv 是否在 data 資料夾內")
    exit()

# 2. 類別資料補 "None"
print("\n2. 處理類別資料 (補 'None')...")
category_cols = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
]

for col in category_cols:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna('None')
            print(f"  {col:20} 補 {missing_count:3d} 個 'None'")

# 3. 數值資料補 0
print("\n3. 處理數值資料 (補 0)...")
numeric_cols_zero = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'
]

for col in numeric_cols_zero:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(0)
            print(f"  {col:20} 補 {missing_count:3d} 個 0")

# 4. MasVnrArea 特殊處理
if 'MasVnrArea' in df.columns and 'MasVnrType' in df.columns:
    mask = (df['MasVnrType'] == 'None') & (df['MasVnrArea'].isnull())
    count = mask.sum()
    if count > 0:
        df.loc[mask, 'MasVnrArea'] = 0
        print(f"  MasVnrArea         根據類型補 {count:3d} 個 0")

# 5. Electrical 補眾數
print("\n4. Electrical 補眾數...")
if 'Electrical' in df.columns:
    missing_count = df['Electrical'].isnull().sum()
    if missing_count > 0:
        mode_value = df['Electrical'].mode()[0]
        df['Electrical'] = df['Electrical'].fillna(mode_value)
        print(f"  Electrical         補 {missing_count:3d} 個 '{mode_value}'")

# 6. KNN 機器學習插補
print("\n5. KNN 機器學習插補...")
knn_cols = ['LotFrontage', 'MasVnrArea']
knn_cols = [col for col in knn_cols if col in df.columns]

if knn_cols:
    # 選擇相關特徵
    feature_cols = [
        'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
        'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
        'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # 準備 KNN 資料
    all_knn_cols = knn_cols + feature_cols
    knn_data = df[all_knn_cols].copy()
    
    # 執行 KNN
    knn_imputer = KNNImputer(n_neighbors=5)
    knn_data_imputed = knn_imputer.fit_transform(knn_data)
    
    # 更新原始資料
    for i, col in enumerate(all_knn_cols):
        if col in knn_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = knn_data_imputed[:, i]
                print(f"  {col:20} 插補 {missing_count:3d} 個值")

# 7. 保存結果
print("\n6. 保存結果...")
output_path = 'output/train_IMP.csv'
df.to_csv(output_path, index=False)
print(f"✓ 已保存為: {output_path}")

# 8. 顯示結果報告
print("\n" + "=" * 60)
print("插補結果報告")
print("=" * 60)

# 檢查缺失值
missing_total = df.isnull().sum().sum()
print(f"總缺失值數量: {missing_total}")

if missing_total == 0:
    print("✓ 所有缺失值已處理完成!")
else:
    print("⚠ 仍有缺失值:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"  {col:20}: {missing:3d} 個")

print(f"\n資料形狀: {df.shape}")
print(f"輸出檔案: output/train_IMP.csv")
print("=" * 60)