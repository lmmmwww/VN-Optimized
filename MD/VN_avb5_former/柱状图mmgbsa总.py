import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 配置全局字体和符号
plt.rcParams['text.usetex'] = False  # 禁用LaTeX（如果未安装）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
data = pd.read_csv('vn_qian_all_mmgbsa.csv', encoding='ISO-8859-1')
data.columns = data.columns.str.replace('¡÷', 'Δ')  # 替换为Unicode Delta

# 绘图
fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(x='Sequence', y='Δ G binding (kcal/mol)', hue='Receptor',
            data=data, palette=["black", "red"], ax=ax,
            edgecolor='black', linewidth=1.5)  # 设置柱子边框

# 使用Unicode符号设置标签
ax.set_xlabel('Sequence', fontsize=26)
ax.set_ylabel('ΔG binding (kcal/mol)', fontsize=26)
ax.set_title('ΔG Binding Energy', fontsize=26)

# 其他格式调整
ax.tick_params(axis='both', labelsize=20, rotation=45)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# 设置横纵比为12:9
ax.set_aspect(9/12/ax.get_data_ratio())

plt.tight_layout()
plt.savefig('vn_qian_mmgbsa_all.eps', format='eps', dpi=300, bbox_inches='tight')
plt.show()