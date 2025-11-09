import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def setup_environment():
    """
    Setup plotting environment and global settings
    """
    warnings.filterwarnings('ignore')
    
    # Font settings
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Plot style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    print("✅ Environment setup completed")

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"📊 Outlier detection for {column}:")
    print(f"  Normal range: {lower_bound:.2f} - {upper_bound:.2f}")
    print(f"  Outliers count: {len(outliers)}")
    
    return outliers