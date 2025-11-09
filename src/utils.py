import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def setup_environment():
    """
    設定繪圖環境和全域設定
    """
    warnings.filterwarnings('ignore')
    
    # 中文字型設定
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 圖表風格
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    print("✅ 環境設定完成")

def detect_outliers_iqr(df, column):
    """
    使用 IQR 方法檢測離群值
    
    Parameters:
    df (DataFrame): 資料
    column (str): 要檢查的欄位
    
    Returns:
    DataFrame: 離群值資料
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"📊 {column} 離群值檢測:")
    print(f"  正常範圍: {lower_bound:.2f} - {upper_bound:.2f}")
    print(f"  離群值數量: {len(outliers)}")
    
    return outliers