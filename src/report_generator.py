import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def create_reports_directory():
    """建立報告資料夾"""
    reports_dir = '../reports'
    figures_dir = f'{reports_dir}/figures'
    
    for directory in [reports_dir, figures_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ 建立資料夾: {directory}")
    
    return reports_dir, figures_dir

def save_analysis_results(train_df, corr_df, missing_df, new_features=None):
    """
    儲存分析結果到 CSV 檔案
    """
    reports_dir, figures_dir = create_reports_directory()
    
    # 儲存相關性分析結果
    if corr_df is not None:
        corr_df.to_csv(f'{reports_dir}/correlation_analysis.csv', index=False, encoding='utf-8-sig')
        print("✅ 已儲存: correlation_analysis.csv")
    
    # 儲存缺失值報告
    if missing_df is not None:
        missing_df.to_csv(f'{reports_dir}/missing_value_report.csv', index=True, encoding='utf-8-sig')
        print("✅ 已儲存: missing_value_report.csv")
    
    # 儲存基本統計量
    if train_df is not None:
        numeric_stats = train_df.describe()
        numeric_stats.to_csv(f'{reports_dir}/numeric_statistics.csv', encoding='utf-8-sig')
        print("✅ 已儲存: numeric_statistics.csv")
    
    # 儲存新特徵列表
    if new_features is not None:
        new_features_df = pd.DataFrame({'新特徵': new_features})
        new_features_df.to_csv(f'{reports_dir}/new_features.csv', index=False, encoding='utf-8-sig')
        print("✅ 已儲存: new_features.csv")

def save_visualizations(figures_dict):
    """
    儲存所有圖表
    """
    _, figures_dir = create_reports_directory()
    
    for name, fig in figures_dict.items():
        if fig is not None:
            # 清理檔案名稱
            safe_name = name.replace(' ', '_').replace(':', '').replace('/', '_')
            filepath = f'{figures_dir}/{safe_name}.png'
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 已儲存圖表: {safe_name}.png")

def generate_summary_report(train_df, corr_df, missing_df, new_features=None):
    """
    生成文字總結報告
    """
    reports_dir, _ = create_reports_directory()
    
    report_content = []
    report_content.append("=" * 60)
    report_content.append("🏠 房屋價格 EDA 分析報告")
    report_content.append("=" * 60)
    report_content.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("")
    
    # 資料基本資訊
    if train_df is not None:
        report_content.append("📊 資料基本資訊:")
        report_content.append(f"  • 資料形狀: {train_df.shape}")
        report_content.append(f"  • 數值特徵: {len(train_df.select_dtypes(include=['number']).columns)}")
        report_content.append(f"  • 類別特徵: {len(train_df.select_dtypes(include=['object']).columns)}")
        
        if 'SalePrice' in train_df.columns:
            report_content.append(f"  • 目標變數範圍: ${train_df['SalePrice'].min():,} - ${train_df['SalePrice'].max():,}")
        report_content.append("")
    
    # 缺失值資訊
    if missing_df is not None:
        report_content.append("🕳️ 缺失值分析:")
        report_content.append(f"  • 有缺失值的欄位: {len(missing_df)} 個")
        report_content.append(f"  • 總缺失值數量: {missing_df['缺失數量'].sum()}")
        if len(missing_df) > 0:
            top_missing = missing_df.head(3)
            for idx, (col, row) in enumerate(top_missing.iterrows()):
                report_content.append(f"  • {col}: {row['缺失比例%']}% 缺失")
        report_content.append("")
    
    # 相關性分析
    if corr_df is not None and len(corr_df) > 1:
        report_content.append("📈 重要特徵發現:")
        top_features = corr_df.iloc[1:6]  # 前5個最相關的特徵（排除SalePrice自己）
        for _, row in top_features.iterrows():
            correlation_strength = "極強" if abs(row['相關係數']) > 0.7 else "強" if abs(row['相關係數']) > 0.5 else "中等"
            report_content.append(f"  • {row['特徵']}: {row['相關係數']:.3f} ({correlation_strength})")
        report_content.append("")
    
    # 新特徵
    if new_features is not None:
        report_content.append("🛠️ 特徵工程:")
        report_content.append(f"  • 新增特徵數量: {len(new_features)}")
        for feature in new_features[:5]:  # 顯示前5個新特徵
            report_content.append(f"  • {feature}")
        if len(new_features) > 5:
            report_content.append(f"  • ... 還有 {len(new_features) - 5} 個特徵")
        report_content.append("")
    
    # 建議
    report_content.append("💡 後續建議:")
    report_content.append("  1. 進行進階特徵工程與選擇")
    report_content.append("  2. 處理類別變數編碼")
    report_content.append("  3. 建立預測模型")
    report_content.append("  4. 模型評估與超參數調優")
    report_content.append("")
    report_content.append("=" * 60)
    
    # 寫入檔案
    report_path = f'{reports_dir}/eda_summary_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f"✅ 已儲存: eda_summary_report.txt")
    
    # 同時在終端顯示
    print('\n'.join(report_content))