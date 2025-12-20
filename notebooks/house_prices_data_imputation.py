import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import KNNImputer
import warnings
import os
warnings.filterwarnings('ignore')

# 创建必要的输出目录
os.makedirs("missing_visualizations", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 加载数据 ====================
print("=" * 60)
print("1. 加载房价数据集")
print("=" * 60)

# 从data文件夹加载数据
try:
    df = pd.read_csv('data/train.csv')
    print("✓ 从 data/train.csv 加载数据")
except FileNotFoundError:
    try:
        # 尝试不同路径
        df = pd.read_csv('train.csv')
        print("✓ 从 train.csv 加载数据")
    except:
        # 如果本地没有，从GitHub加载
        url = "https://raw.githubusercontent.com/waichou-lab/House-Prices-EDA/main/data/train.csv"
        df = pd.read_csv(url)
        print("✓ 从GitHub加载数据")

print(f"数据形状: {df.shape}")
print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")

# ==================== 2. 缺失值总体分析 ====================
print("\n" + "=" * 60)
print("2. 缺失值总体分析")
print("=" * 60)

# 计算缺失统计
missing_stats = pd.DataFrame({
    '缺失数量': df.isnull().sum(),
    '缺失比例': df.isnull().sum() / len(df) * 100
})
missing_stats = missing_stats[missing_stats['缺失数量'] > 0].sort_values('缺失比例', ascending=False)

print("有缺失值的字段 (前15个):")
print(missing_stats.head(15))

# ==================== 3. missingno 可视化 - 分别保存四张图 ====================
print("\n" + "=" * 60)
print("3. 生成 missingno 可视化图表 (分别保存)")
print("=" * 60)

# 3.1 矩阵图 (Matrix) - 单独保存
plt.figure(figsize=(12, 8))
msno.matrix(df.sample(min(500, len(df))), fontsize=10)
plt.title('缺失值矩阵图 (Missing Data Matrix)', fontsize=14, fontweight='bold')
plt.xlabel('数据字段', fontsize=12)
plt.ylabel('样本索引', fontsize=12)
plt.tight_layout()
matrix_path = os.path.join("missing_visualizations", "1_missing_matrix.png")
plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 矩阵图已保存为: {matrix_path}")

# 3.2 条形图 (Bar Chart) - 单独保存
plt.figure(figsize=(12, 8))
msno.bar(df, fontsize=10, color='steelblue')
plt.title('缺失值条形图 (Missing Data Bar Chart)', fontsize=14, fontweight='bold')
plt.xlabel('数据字段', fontsize=12)
plt.ylabel('完整度 (%)', fontsize=12)
plt.tight_layout()
bar_path = os.path.join("missing_visualizations", "2_missing_bar.png")
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 条形图已保存为: {bar_path}")

# 3.3 热力图 (Heatmap) - 单独保存
plt.figure(figsize=(10, 8))
msno.heatmap(df, cmap='RdYlGn_r', fontsize=10)
plt.title('缺失值相关性热力图 (Missing Data Heatmap)', fontsize=14, fontweight='bold')
plt.tight_layout()
heatmap_path = os.path.join("missing_visualizations", "3_missing_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 热力图已保存为: {heatmap_path}")

# 3.4 树状图 (Dendrogram) - 单独保存
plt.figure(figsize=(10, 8))
msno.dendrogram(df, fontsize=10)
plt.title('缺失值聚类树状图 (Missing Data Dendrogram)', fontsize=14, fontweight='bold')
plt.tight_layout()
dendrogram_path = os.path.join("missing_visualizations", "4_missing_dendrogram.png")
plt.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 树状图已保存为: {dendrogram_path}")

# 3.5 额外：缺失统计条形图（按缺失比例排序）
plt.figure(figsize=(14, 8))
missing_stats_sorted = missing_stats.sort_values('缺失比例', ascending=True)

plt.barh(range(len(missing_stats_sorted)), missing_stats_sorted['缺失比例'], color='skyblue')
plt.yticks(range(len(missing_stats_sorted)), missing_stats_sorted.index)
plt.xlabel('缺失比例 (%)', fontsize=12)
plt.ylabel('数据字段', fontsize=12)
plt.title('各字段缺失比例排序图 (Sorted Missing Data Percentage)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# 在条形上添加百分比标签
for i, (idx, row) in enumerate(missing_stats_sorted.iterrows()):
    plt.text(row['缺失比例'] + 0.5, i, f"{row['缺失比例']:.1f}%", 
             va='center', fontsize=9)

plt.tight_layout()
sorted_bar_path = os.path.join("missing_visualizations", "5_missing_sorted_bar.png")
plt.savefig(sorted_bar_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 缺失比例排序图已保存为: {sorted_bar_path}")
print("所有可视化图表已保存在 'missing_visualizations' 文件夹中")

# ==================== 4. 实施插补策略 ====================
print("\n" + "=" * 60)
print("4. 实施数据插补策略")
print("=" * 60)

# 创建数据副本用于插补
df_filled = df.copy()
print("开始数据插补处理...")

# 4.1 MNAR 字段 - 补 "None" (类别)
mnar_categorical = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType'
]

for col in mnar_categorical:
    if col in df_filled.columns:
        df_filled[col] = df_filled[col].fillna('None')
        print(f"  ✓ {col}: 补 'None' (MNAR - 无该设施)")

# 4.2 MNAR 字段 - 补 0 (数值)
mnar_numeric = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
    'MasVnrArea'
]

for col in mnar_numeric:
    if col in df_filled.columns:
        df_filled[col] = df_filled[col].fillna(0)
        print(f"  ✓ {col}: 补 0 (MNAR - 无该设施)")

# 4.3 MAR 字段 - LotFrontage 按 Neighborhood 分组插补
if 'LotFrontage' in df_filled.columns and 'Neighborhood' in df_filled.columns:
    # 计算每个邻域的中位数
    neighborhood_medians = df_filled.groupby('Neighborhood')['LotFrontage'].median()
    
    # 填补缺失值
    def fill_lotfrontage(row):
        if pd.isnull(row['LotFrontage']):
            return neighborhood_medians.get(row['Neighborhood'], df_filled['LotFrontage'].median())
        return row['LotFrontage']
    
    df_filled['LotFrontage'] = df_filled.apply(fill_lotfrontage, axis=1)
    print(f"  ✓ LotFrontage: 按 Neighborhood 分组补中位数 (MAR)")

# 4.4 MCAR 字段 - Electrical 补众数
if 'Electrical' in df_filled.columns:
    mode_value = df_filled['Electrical'].mode()[0]
    df_filled['Electrical'] = df_filled['Electrical'].fillna(mode_value)
    print(f"  ✓ Electrical: 补众数 '{mode_value}' (MCAR - 随机缺失)")

# 4.5 剩余数值字段用中位数填补
numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_filled[col].isnull().sum() > 0:
        median_val = df_filled[col].median()
        df_filled[col] = df_filled[col].fillna(median_val)
        print(f"  ✓ {col}: 补中位数 {median_val:.2f}")

# 4.6 剩余类别字段用众数填补
categorical_cols = df_filled.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df_filled[col].isnull().sum() > 0:
        mode_val = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 'Unknown'
        df_filled[col] = df_filled[col].fillna(mode_val)
        print(f"  ✓ {col}: 补众数 '{mode_val}'")

# ==================== 5. 保存结果 ====================
print("\n" + "=" * 60)
print("5. 保存处理结果")
print("=" * 60)

# 保存插补后的数据到output文件夹
output_csv = os.path.join("output", "train_imputed.csv")
df_filled.to_csv(output_csv, index=False)
print(f"✓ 插补后的数据已保存为: {os.path.abspath(output_csv)}")

# 生成详细报告到output文件夹
output_txt = os.path.join("output", "imputation_report.txt")
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("        数据插补处理报告\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"处理时间: {pd.Timestamp.now()}\n")
    f.write(f"原始数据路径: data/train.csv\n")
    f.write(f"原始数据形状: {df.shape}\n")
    f.write(f"原始缺失值总数: {df.isnull().sum().sum()}\n")
    f.write(f"插补后缺失值总数: {df_filled.isnull().sum().sum()}\n")
    f.write(f"处理比例: {(df.isnull().sum().sum() - df_filled.isnull().sum().sum()) / df.isnull().sum().sum() * 100:.1f}%\n\n")
    
    f.write("=" * 40 + "\n")
    f.write("主要缺失字段处理方式\n")
    f.write("=" * 40 + "\n\n")
    
    f.write("1. MNAR (结构性缺失) -> 补 'None' 或 0:\n")
    for i, col in enumerate(mnar_categorical[:8], 1):
        f.write(f"   {i:2d}. {col}\n")
    f.write("   ... (共15个字段)\n\n")
    
    f.write("2. MAR (可预测缺失) -> 分组插补:\n")
    f.write("   • LotFrontage: 按 Neighborhood 分组补中位数\n\n")
    
    f.write("3. MCAR (随机缺失) -> 补众数:\n")
    f.write("   • Electrical: 补众数 'SBrkr'\n\n")
    
    f.write("4. 其他字段:\n")
    f.write("   • 数值字段: 补中位数\n")
    f.write("   • 类别字段: 补众数\n\n")
    
    f.write("=" * 40 + "\n")
    f.write("生成的可视化图表\n")
    f.write("=" * 40 + "\n\n")
    f.write("1. 1_missing_matrix.png     - 缺失值矩阵图\n")
    f.write("2. 2_missing_bar.png        - 缺失值条形图\n")
    f.write("3. 3_missing_heatmap.png    - 缺失相关性热力图\n")
    f.write("4. 4_missing_dendrogram.png - 缺失聚类树状图\n")
    f.write("5. 5_missing_sorted_bar.png - 缺失比例排序图\n")
    f.write("\n所有图表保存在 'missing_visualizations' 文件夹中\n")

print(f"✓ 处理报告已保存为: {os.path.abspath(output_txt)}")

# ==================== 6. 完成信息 ====================
print("\n" + "=" * 60)
print("6. 数据处理完成总结")
print("=" * 60)

print("✅ 完成的工作:")
print(f"  1. 可视化图表: 5张 (保存在 missing_visualizations/)")
print(f"  2. 插补后数据: {output_csv}")
print(f"  3. 处理报告: {output_txt}")
print(f"\n✅ 插补效果:")
print(f"  原始缺失值: {df.isnull().sum().sum()}")
print(f"  处理后缺失值: {df_filled.isnull().sum().sum()}")
print(f"  处理完成率: {(df.isnull().sum().sum() - df_filled.isnull().sum().sum()) / df.isnull().sum().sum() * 100:.1f}%")

print("\n" + "=" * 60)
print("所有任务已完成！可以开始录制口頭報告。")

print("=" * 60)

