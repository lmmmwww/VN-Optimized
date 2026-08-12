import matplotlib.pyplot as plt
import numpy as np

# 2×2混淆矩阵数据
confusion_data = np.array([
    [2721, 446],  # TP, FN
    [523, 1273]   # FP, TN
])

# 设置标签
x_labels = ['0', '1']  # 横坐标标签（预测标签）
y_labels = ['0', '1']  # 纵坐标标签（真实标签）

# 设置字体为可编辑的矢量字体
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(confusion_data, cmap='Blues', interpolation='nearest')

# 添加数值标注
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(confusion_data[i, j]),
                ha="center", va="center",
                color="white" if confusion_data[i, j] > confusion_data.max()/2 else "black",
                fontsize=16, weight='bold')

# 设置坐标轴标签
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)

# 添加轴标签
ax.set_xlabel('Predicted Label', fontfamily='sans-serif')
ax.set_ylabel('Actual Label', fontfamily='sans-serif')

# 添加标题
ax.set_title('Confusion Matrix', fontfamily='sans-serif')

plt.tight_layout()
plt.savefig('confusion_matrix_editable.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('confusion_matrix_editable.pdf', format='pdf', dpi=300, bbox_inches='tight')
print('confusion_matrix_editable.eps')
print('confusion_matrix_editable.pdf')
plt.show()