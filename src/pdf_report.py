import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime
import seaborn as sns

def create_pdf_report(train_df, corr_df, missing_df, new_features=None):
    """
    Create PDF EDA report
    """
    reports_dir = '../reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    pdf_path = f'{reports_dir}/eda_report.pdf'
    
    with PdfPages(pdf_path) as pdf:
        create_cover_page(pdf)
        create_executive_summary(pdf, train_df, corr_df, missing_df)
        create_data_overview(pdf, train_df)
        create_missing_analysis(pdf, missing_df)
        create_target_analysis(pdf, train_df)
        create_correlation_analysis(pdf, corr_df)
        
        if new_features:
            create_feature_engineering_page(pdf, new_features)
        
        create_conclusions_page(pdf)
    
    print(f"✅ PDF report generated: {pdf_path}")
    return pdf_path

def create_cover_page(pdf):
    """Create cover page"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    ax.text(0.5, 0.7, 'HOUSE PRICES EDA REPORT', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    ax.text(0.5, 0.6, 'Exploratory Data Analysis', 
            ha='center', va='center', fontsize=14)
    
    ax.text(0.5, 0.4, 'Kaggle House Prices Competition', 
            ha='center', va='center', fontsize=12)
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    ax.text(0.5, 0.2, f'Generated: {current_date}', 
            ha='center', va='center', fontsize=10)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_executive_summary(pdf, train_df, corr_df, missing_df):
    """Create executive summary page"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    y_position = 0.9
    line_height = 0.05
    
    ax.text(0.1, y_position, 'EXECUTIVE SUMMARY', fontsize=16, fontweight='bold')
    y_position -= line_height * 2
    
    # Data overview
    if train_df is not None:
        ax.text(0.1, y_position, 'DATA OVERVIEW:', fontsize=12, fontweight='bold')
        y_position -= line_height
        ax.text(0.15, y_position, f'Samples: {train_df.shape[0]:,}', fontsize=10)
        y_position -= line_height
        ax.text(0.15, y_position, f'Features: {train_df.shape[1]}', fontsize=10)
        y_position -= line_height
        if 'SalePrice' in train_df.columns:
            ax.text(0.15, y_position, f'Price Range: ${train_df["SalePrice"].min():,} - ${train_df["SalePrice"].max():,}', fontsize=10)
            y_position -= line_height
        y_position -= line_height
    
    # Missing values
    if missing_df is not None and len(missing_df) > 0:
        ax.text(0.1, y_position, 'DATA QUALITY:', fontsize=12, fontweight='bold')
        y_position -= line_height
        ax.text(0.15, y_position, f'Columns with Missing Values: {len(missing_df)}', fontsize=10)
        y_position -= line_height
        if 'Missing_Percent' in missing_df.columns:
            highest_missing = missing_df.iloc[0]['Missing_Percent']
            ax.text(0.15, y_position, f'Highest Missing: {missing_df.index[0]} ({highest_missing}%)', fontsize=10)
            y_position -= line_height
        y_position -= line_height
    
    # Key findings
    if corr_df is not None and len(corr_df) > 1:
        ax.text(0.1, y_position, 'KEY FINDINGS:', fontsize=12, fontweight='bold')
        y_position -= line_height
        top_features = corr_df.iloc[1:4]
        for _, row in top_features.iterrows():
            ax.text(0.15, y_position, f'{row["Feature"]}: Correlation {row["Correlation"]:.3f}', fontsize=10)
            y_position -= line_height
        y_position -= line_height
    
    # Conclusions
    ax.text(0.1, y_position, 'CONCLUSIONS:', fontsize=12, fontweight='bold')
    y_position -= line_height
    conclusions = [
        "Target variable is right-skewed, log transform recommended",
        "Strong correlated features identified for modeling",
        "Missing values need to be handled"
    ]
    for conclusion in conclusions:
        ax.text(0.15, y_position, f'- {conclusion}', fontsize=10)
        y_position -= line_height
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_data_overview(pdf, train_df):
    """Create data overview page"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    y_position = 0.9
    
    ax.text(0.1, y_position, 'DATASET OVERVIEW', fontsize=16, fontweight='bold')
    y_position -= 0.08
    
    if train_df is not None:
        ax.text(0.1, y_position, 'BASIC INFORMATION:', fontsize=12, fontweight='bold')
        y_position -= 0.05
        
        info_text = [
            f"Data Shape: {train_df.shape}",
            f"Numeric Features: {len(train_df.select_dtypes(include=['number']).columns)}",
            f"Categorical Features: {len(train_df.select_dtypes(include=['object']).columns)}",
            f"Memory Usage: {train_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        ]
        
        for text in info_text:
            ax.text(0.15, y_position, f'- {text}', fontsize=10)
            y_position -= 0.04
        
        y_position -= 0.02
        
        if 'SalePrice' in train_df.columns:
            ax.text(0.1, y_position, 'TARGET STATISTICS:', fontsize=12, fontweight='bold')
            y_position -= 0.05
            
            price_stats = train_df['SalePrice'].describe()
            stats_text = [
                f"Mean: ${price_stats['mean']:,.0f}",
                f"Median: ${price_stats['50%']:,.0f}",
                f"Standard Deviation: ${price_stats['std']:,.0f}",
                f"Skewness: {train_df['SalePrice'].skew():.3f}"
            ]
            
            for text in stats_text:
                ax.text(0.15, y_position, f'- {text}', fontsize=10)
                y_position -= 0.04
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_missing_analysis(pdf, missing_df):
    """Create missing values analysis page"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    y_position = 0.9
    
    ax.text(0.1, y_position, 'MISSING VALUES ANALYSIS', fontsize=16, fontweight='bold')
    y_position -= 0.08
    
    if missing_df is not None and len(missing_df) > 0:
        ax.text(0.1, y_position, 'SUMMARY:', fontsize=12, fontweight='bold')
        y_position -= 0.05
        
        total_missing = missing_df['Missing_Count'].sum() if 'Missing_Count' in missing_df.columns else 'N/A'
        avg_missing = missing_df['Missing_Percent'].mean() if 'Missing_Percent' in missing_df.columns else 'N/A'
        
        summary_text = [
            f"Columns with Missing Values: {len(missing_df)}",
            f"Total Missing Values: {total_missing}",
            f"Average Missing %: {avg_missing:.1f}%" if isinstance(avg_missing, (int, float)) else f"Average Missing %: {avg_missing}"
        ]
        
        for text in summary_text:
            ax.text(0.15, y_position, f'- {text}', fontsize=10)
            y_position -= 0.04
        
        y_position -= 0.02
        
        ax.text(0.1, y_position, 'TOP MISSING COLUMNS:', fontsize=12, fontweight='bold')
        y_position -= 0.05
        
        top_missing = missing_df.head(8)
        for idx, (col_name, row) in enumerate(top_missing.iterrows()):
            if y_position < 0.1:
                break
            missing_value = row['Missing_Percent'] if 'Missing_Percent' in row else 'N/A'
            ax.text(0.15, y_position, f'{idx+1}. {col_name}: {missing_value}%', fontsize=9)
            y_position -= 0.035
    else:
        ax.text(0.1, y_position, 'NO MISSING VALUES FOUND', fontsize=12)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_target_analysis(pdf, train_df):
    """Create target variable analysis page"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    y_position = 0.9
    
    ax.text(0.1, y_position, 'TARGET VARIABLE ANALYSIS', fontsize=16, fontweight='bold')
    y_position -= 0.08
    
    if train_df is not None and 'SalePrice' in train_df.columns:
        ax.text(0.1, y_position, 'STATISTICS:', fontsize=12, fontweight='bold')
        y_position -= 0.05
        
        stats = train_df['SalePrice'].describe()
        analysis_text = [
            f"Skewness: {train_df['SalePrice'].skew():.3f}",
            f"Kurtosis: {train_df['SalePrice'].kurt():.3f}",
            f"Coefficient of Variation: {stats['std']/stats['mean']:.3f}",
            "Recommendation: Log transformation"
        ]
        
        for text in analysis_text:
            ax.text(0.15, y_position, f'- {text}', fontsize=10)
            y_position -= 0.04
        
        y_position -= 0.02
        
        ax.text(0.1, y_position, 'DISTRIBUTION NOTES:', fontsize=12, fontweight='bold')
        y_position -= 0.05
        
        if train_df['SalePrice'].skew() > 1:
            distribution_notes = [
                "Distribution is right-skewed",
                "Few high-priced houses pull mean upward",
                "Log transformation will help"
            ]
        else:
            distribution_notes = [
                "Distribution is relatively symmetric",
                "Suitable for linear models",
                "Watch for outliers"
            ]
        
        for note in distribution_notes:
            ax.text(0.15, y_position, f'- {note}', fontsize=10)
            y_position -= 0.04
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_correlation_analysis(pdf, corr_df):
    """Create correlation analysis page"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    y_position = 0.9
    
    ax.text(0.1, y_position, 'CORRELATION ANALYSIS', fontsize=16, fontweight='bold')
    y_position -= 0.08
    
    if corr_df is not None and len(corr_df) > 1:
        strong_corr = corr_df[corr_df['Correlation'].abs() > 0.5]
        moderate_corr = corr_df[(corr_df['Correlation'].abs() > 0.3) & (corr_df['Correlation'].abs() <= 0.5)]
        
        ax.text(0.1, y_position, 'CORRELATION STRENGTH:', fontsize=12, fontweight='bold')
        y_position -= 0.05
        
        strength_text = [
            f"Strong (>0.5): {len(strong_corr)} features",
            f"Moderate (0.3-0.5): {len(moderate_corr)} features",
            f"Weak (<0.3): {len(corr_df) - len(strong_corr) - len(moderate_corr)} features"
        ]
        
        for text in strength_text:
            ax.text(0.15, y_position, f'- {text}', fontsize=10)
            y_position -= 0.04
        
        y_position -= 0.02
        
        ax.text(0.1, y_position, 'TOP FEATURES:', fontsize=12, fontweight='bold')
        y_position -= 0.05
        
        top_features = corr_df.iloc[1:6]
        for idx, (_, row) in enumerate(top_features.iterrows()):
            if y_position < 0.1:
                break
            correlation_strength = "Very Strong" if abs(row['Correlation']) > 0.7 else "Strong" if abs(row['Correlation']) > 0.5 else "Moderate"
            ax.text(0.15, y_position, f'{idx+1}. {row["Feature"]}: {row["Correlation"]:.3f}', fontsize=10)
            y_position -= 0.035
            ax.text(0.17, y_position, f'({correlation_strength})', fontsize=9, style='italic')
            y_position -= 0.025
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_feature_engineering_page(pdf, new_features):
    """Create feature engineering page"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    y_position = 0.9
    
    ax.text(0.1, y_position, 'FEATURE ENGINEERING', fontsize=16, fontweight='bold')
    y_position -= 0.08
    
    if new_features:
        ax.text(0.1, y_position, 'NEW FEATURES:', fontsize=12, fontweight='bold')
        y_position -= 0.05
        
        for feature in new_features:
            if y_position < 0.1:
                break
            ax.text(0.15, y_position, f'- {feature}', fontsize=10)
            y_position -= 0.035
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_conclusions_page(pdf):
    """Create conclusions page"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    y_position = 0.9
    
    ax.text(0.1, y_position, 'CONCLUSIONS AND RECOMMENDATIONS', fontsize=16, fontweight='bold')
    y_position -= 0.08
    
    ax.text(0.1, y_position, 'MAIN CONCLUSIONS:', fontsize=12, fontweight='bold')
    y_position -= 0.05
    
    conclusions = [
        "Good data quality with some missing values",
        "Target variable needs log transformation",
        "Strong features identified for modeling",
        "Feature engineering can improve performance"
    ]
    
    for conclusion in conclusions:
        ax.text(0.15, y_position, f'- {conclusion}', fontsize=10)
        y_position -= 0.04
    
    y_position -= 0.02
    
    ax.text(0.1, y_position, 'NEXT STEPS:', fontsize=12, fontweight='bold')
    y_position -= 0.05
    
    next_steps = [
        "1. Advanced feature engineering",
        "2. Handle categorical variables",
        "3. Build predictive models",
        "4. Model evaluation and tuning",
        "5. Results interpretation"
    ]
    
    for step in next_steps:
        ax.text(0.15, y_position, step, fontsize=10)
        y_position -= 0.035
    
    ax.text(0.1, 0.1, 'Report completed', fontsize=10, style='italic')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()