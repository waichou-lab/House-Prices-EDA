import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_price_distribution(df, price_col='SalePrice', figsize=(12, 5)):
    """
    繪製價格分布圖
    
    Parameters:
    df (DataFrame): 資料
    price_col (str): 價格欄位名稱
    figsize (tuple): 圖表大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 原始分布
    sns.histplot(df[price_col], kde=True, ax=axes[0], color='skyblue')
    axes[0].axvline(df[price_col].mean(), color='red', linestyle='--', alpha=0.7, label='平均')
    axes[0].axvline(df[price_col].median(), color='green', linestyle='--', alpha=0.7, label='中位數')
    axes[0].set_title(f'{price_col} 原始分布\n偏態: {df[price_col].skew():.3f}')
    axes[0].legend()
    
    # 對數轉換後分布
    sns.histplot(np.log1p(df[price_col]), kde=True, ax=axes[1], color='lightcoral')
    axes[1].set_title(f'{price_col} 對數轉換後分布')
    
    plt.tight_layout()
    return fig

def plot_correlation_analysis(df, target_col='SalePrice', top_n=15, figsize=(14, 10)):
    """
    相關性分析與視覺化
    
    Parameters:
    df (DataFrame): 資料
    target_col (str): 目標變數
    top_n (int): 顯示前 N 個相關特徵
    figsize (tuple): 圖表大小
    """
    # 計算相關係數
    corr_with_target = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
    
    # 建立相關性表格
    corr_df = pd.DataFrame({
        '特徵': corr_with_target.head(top_n).index,
        '相關係數': corr_with_target.head(top_n).values
    })
    
    # 繪製相關性長條圖
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 長條圖
    top_corr = corr_with_target.head(top_n).iloc[1:]  # 排除目標變數自己
    colors = ['green' if x > 0.6 else 'blue' for x in top_corr.values]
    
    axes[0].barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(top_corr)))
    axes[0].set_yticklabels(top_corr.index)
    axes[0].set_xlabel('相關係數')
    axes[0].set_title(f'與 {target_col} 最相關的 {top_n-1} 個特徵')
    
    # 添加數值標籤
    for i, v in enumerate(top_corr.values):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')
    
    # 熱力圖
    top_features = corr_with_target.head(top_n).index
    corr_matrix = df[top_features].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, ax=axes[1], cbar_kws={"shrink": .8})
    axes[1].set_title(f'前{top_n}個特徵相關性熱力圖')
    
    plt.tight_layout()
    return fig, corr_df