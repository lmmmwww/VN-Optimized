import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.preprocessing import StandardScaler

# 1. 加载数据
df = pd.read_csv("avb5_peptide_features _all.csv", encoding='gbk')  # 替换为你的文件路径
print("原始数据列名:", df.columns.tolist())

# 2. 数据预处理
# 2.1 定义特征和目标变量
# 假设您的二分类标签列名为"Binding_Label"（请替换为实际列名）
target_col = "MMGBSA dG Bind"  # 修改为您的二分类标签列名
exclude_cols = [ "Sequence", "MMGBSA dG Bind", target_col]  # 非特征列
X = df.drop(columns=exclude_cols)
y = df[target_col]  # 直接使用现有的二分类标签

# 检查类别分布
print("\n类别分布:")
print(y.value_counts())

# 2.2 智能处理缺失值（不处理0值）
print("\n缺失值统计（True表示缺失）:")
# 只识别np.nan、None等真正的缺失值，不包括0
missing_stats = X.isna().sum()
print(missing_stats[missing_stats > 0])

# 仅对真正缺失的值用中位数填充（跳过0值）
for col in X.columns:
    # 找出非0的缺失值位置
    missing_mask = X[col].isna() & (X[col] != 0)
    if missing_mask.sum() > 0:
        median_val = X[col].median()
        X.loc[missing_mask, col] = median_val
        print(f"列 {col}: 填充了 {missing_mask.sum()} 个缺失值（非0）")

# 2.4 划分训练集和测试集（保持类别比例）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集样本: {X_train.shape[0]}, 测试集样本: {X_test.shape[0]}")
print("训练集类别分布:", pd.Series(y_train).value_counts())
print("测试集类别分布:", pd.Series(y_test).value_counts())

# 3. 网格搜索调参
param_grid = {
    'n_estimators': [100, 200, 300],    # 树数量，需平衡性能与计算效率:ml-citation{ref="2,7" data="citationList"}
    'max_depth': [5, 10, 15, None],     # 树深度，None允许完全生长但可能过拟合:ml-citation{ref="2,4" data="citationList"}
    'min_samples_split': [2, 5, 10],    # 分裂节点最小样本数，值越大越保守:ml-citation{ref="2,7" data="citationList"}
    'min_samples_leaf': [1, 2, 4],      # 叶节点最小样本数，控制模型复杂度:ml-citation{ref="7" data="citationList"}
    'max_features': ['sqrt', 'log2'],   # 特征选择方式，减少过拟合风险:ml-citation{ref="2,4" data="citationList"}
    'class_weight': [                   # 修正后的类别权重参数
        {0: 1, 1: 3.33},                # 手动加权（6120/1840≈3.33）
        'balanced'                      # 自动平衡类别权重:ml-citation{ref="2,7" data="citationList"}
    ]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',  # 使用AUC作为评估指标
    n_jobs=-1,
    verbose=2
)

print("\n开始网格搜索调参...")
grid_search.fit(X_train, y_train)

# 4. 输出最佳参数和模型性能
print("\n=== 调参结果 ===")
print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证 AUC:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# 5. 在测试集上评估最佳模型
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 预测概率

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n=== 测试集性能 ===")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")
print("\n混淆矩阵:")
print(conf_matrix)

# 6. 特征重要性分析
importance = best_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values("Importance", ascending=False)

# 保存全部特征重要性到 CSV
importance_df.to_csv("feature_importance0916.csv", index=False, encoding="utf-8-sig")

print("\nTop 10重要特征:")
print(importance_df.head(10))

# 7. 可视化
plt.figure(figsize=(15, 12))

# 7.1 特征重要性
plt.subplot(2, 2, 1)
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df.head(15),
    palette="viridis"
)
plt.title("Top 15 Feature Importances", fontsize=14)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("")
plt.tight_layout()
plt.show()


# 7.2 混淆矩阵
plt.subplot(2, 2, 2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)

# 7.3 ROC曲线
plt.subplot(2, 2, 3)
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title("ROC Curve", fontsize=14)
plt.plot([0, 1], [0, 1], 'r--')

# 7.4 Precision-Recall曲线
plt.subplot(2, 2, 4)
PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Precision-Recall Curve", fontsize=14)

plt.tight_layout()
plt.savefig("rf_classification_results.png", dpi=300)
plt.show()

# 8. 保存最佳模型
import joblib
joblib.dump(best_model, 'best_rf_classifier.pkl')
print("\n模型已保存为 'best_rf_classifier.pkl'")