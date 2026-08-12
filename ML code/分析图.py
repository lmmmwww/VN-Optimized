# =============== 导入库 ===============
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 设置中文显示和样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set_style("whitegrid")
plt.style.use('ggplot')

# =============== 数据加载 ===============
file_path = 'avb5_peptide_features _all.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"未找到文件 {file_path}，请检查文件路径")

df = pd.read_csv(file_path)
print("数据预览：")
print(df.head())

# =============== 数据预处理 ===============
# 定义要排除的列
exclude_cols = ['Sequence', 'MMGBSA dG Bind']  # 需要排除的列名
target_col = 'MMGBSA dG Bind'  # 目标列

# 自动获取特征列（排除指定列）
feature_cols = [col for col in df.columns if col not in exclude_cols]
print("\n将分析的特征列：", feature_cols)

# 检查缺失值
print("\n缺失值统计：")
print(df.isnull().sum())

# 填充缺失值
for col in feature_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"填充 {col} 列的 {df[col].isnull().sum()} 个缺失值")

# =============== 特征分布分析 ===============
# 创建输出目录
os.makedirs('results', exist_ok=True)


# 1. 绘制箱线图（自动适应特征数量）
def plot_boxplots():
    n_features = len(feature_cols)
    ncols = 3  # 每行显示3个子图
    nrows = int(np.ceil(n_features / ncols))

    plt.figure(figsize=(18, 6 * nrows))
    for i, feature in enumerate(feature_cols, 1):
        plt.subplot(nrows, ncols, i)
        sns.boxplot(x=target_col, y=feature, data=df, width=0.5)

        # 添加统计信息
        class0 = df[df[target_col] == 0][feature]
        class1 = df[df[target_col] == 1][feature]
        t_stat, p_val = stats.ttest_ind(class0, class1, equal_var=False)

        plt.title(f"{feature}\n(p-value: {p_val:.3e})", fontsize=12)
        plt.xlabel("结合类别 (0:不结合, 1:结合)")
        plt.ylabel(feature)

    plt.tight_layout()
    plt.savefig('results/feature_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()


# 2. 绘制小提琴图（自动适应特征数量）
def plot_violins():
    n_features = len(feature_cols)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))

    plt.figure(figsize=(18, 6 * nrows))
    for i, feature in enumerate(feature_cols, 1):
        plt.subplot(nrows, ncols, i)
        sns.violinplot(x=target_col, y=feature, data=df, cut=0)

        # 添加中位数线
        for cls in [0, 1]:
            median_val = df[df[target_col] == cls][feature].median()
            plt.hlines(median_val, cls - 0.2, cls + 0.2, colors='red', linestyles='dashed')

        plt.title(f"{feature} 分布", fontsize=12)
        plt.xlabel("结合类别 (0:不结合, 1:结合)")
        plt.ylabel(feature)

    plt.tight_layout()
    plt.savefig('results/feature_violins.png', dpi=300, bbox_inches='tight')
    plt.show()


# 3. 统计检验表格
def generate_stats_table():
    stats_table = pd.DataFrame()

    for feature in feature_cols:
        for cls in [0, 1]:
            subset = df[df[target_col] == cls][feature]
            stats_table.loc[feature, f'Class_{cls}_Mean'] = subset.mean()
            stats_table.loc[feature, f'Class_{cls}_Median'] = subset.median()
            stats_table.loc[feature, f'Class_{cls}_Std'] = subset.std()

        # 计算效应量（Cohen's d）
        n0 = len(df[df[target_col] == 0])
        n1 = len(df[df[target_col] == 1])
        pooled_std = np.sqrt(((n0 - 1) * stats_table.loc[feature, 'Class_0_Std'] ** 2 +
                              (n1 - 1) * stats_table.loc[feature, 'Class_1_Std'] ** 2) / (n0 + n1 - 2))
        stats_table.loc[feature, "Cohen's d"] = (
                (stats_table.loc[feature, 'Class_1_Mean'] - stats_table.loc[feature, 'Class_0_Mean']) / pooled_std
        )

        # 统计检验
        _, p_val = stats.ttest_ind(
            df[df[target_col] == 0][feature],
            df[df[target_col] == 1][feature],
            equal_var=False
        )
        stats_table.loc[feature, 'p-value'] = p_val

    # 保存统计结果
    stats_table.to_excel('results/feature_stats.xlsx')
    print("\n特征统计结果已保存到 results/feature_stats.xlsx")

    return stats_table


# =============== 执行分析 ===============
print("\n正在绘制箱线图...")
plot_boxplots()

print("\n正在绘制小提琴图...")
plot_violins()

print("\n正在计算统计指标...")
stats_df = generate_stats_table()

print("\n=== 分析完成 ===")
print("生成的图表和结果已保存到 results 目录")
print("\n统计结果摘要：")
print(stats_df[['Class_0_Mean', 'Class_1_Mean', "Cohen's d", 'p-value']])