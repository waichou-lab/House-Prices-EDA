import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_price_distribution(df, price_col='SalePrice', figsize=(12, 5)):
    """
    Plot price distribution
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Original distribution
    sns.histplot(df[price_col], kde=True, ax=axes[0], color='skyblue')
    axes[0].axvline(df[price_col].mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
    axes[0].axvline(df[price_col].median(), color='green', linestyle='--', alpha=0.7, label='Median')
    axes[0].set_title(f'{price_col} Distribution\nSkewness: {df[price_col].skew():.3f}')
    axes[0].legend()
    
    # Log transformed distribution
    sns.histplot(np.log1p(df[price_col]), kde=True, ax=axes[1], color='lightcoral')
    axes[1].set_title(f'{price_col} Log-Transformed')
    
    plt.tight_layout()
    return fig

def plot_correlation_analysis(df, target_col='SalePrice', top_n=15, figsize=(14, 10)):
    """
    Correlation analysis and visualization
    """
    # Calculate correlations
    corr_with_target = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
    
    # Create correlation dataframe
    corr_df = pd.DataFrame({
        'Feature': corr_with_target.head(top_n).index,
        'Correlation': corr_with_target.head(top_n).values
    })
    
    # Plot correlation bar chart
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart
    top_corr = corr_with_target.head(top_n).iloc[1:]  # Exclude target variable itself
    colors = ['green' if x > 0.6 else 'blue' for x in top_corr.values]
    
    axes[0].barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(top_corr)))
    axes[0].set_yticklabels(top_corr.index)
    axes[0].set_xlabel('Correlation Coefficient')
    axes[0].set_title(f'Top {top_n-1} Features Correlated with {target_col}')
    
    # Add value labels
    for i, v in enumerate(top_corr.values):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')
    
    # Heatmap
    top_features = corr_with_target.head(top_n).index
    corr_matrix = df[top_features].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, ax=axes[1], cbar_kws={"shrink": .8})
    axes[1].set_title(f'Top {top_n} Features Correlation Heatmap')
    
    plt.tight_layout()
    return fig, corr_df