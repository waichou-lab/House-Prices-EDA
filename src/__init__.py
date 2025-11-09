"""
房屋價格 EDA 專案 - 原始碼模組
"""

from .data_loader import load_dataset, get_data_info
from .data_cleaner import check_missing_data, remove_high_missing_columns
from .visualization import plot_price_distribution, plot_correlation_analysis
from .feature_engineering import create_new_features, apply_log_transform
from .utils import setup_environment, detect_outliers_iqr
from .report_generator import save_analysis_results, save_visualizations, generate_summary_report

__version__ = "1.0.0"
__author__ = "Your Name"

print("✅ src 模組載入完成")