# =============== 导入库 ===============
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             RocCurveDisplay, PrecisionRecallDisplay,
                             classification_report, precision_recall_curve)
from sklearn.preprocessing import StandardScaler

# =============== 配置参数 ===============
RANDOM_STATE = 42
TARGET_COL = "MMGBSA dG Bind"
EXCLUDE_COLS = ["Sequence", "MMGBSA dG Bind",TARGET_COL]
DATA_PATH = "avb3_peptide_features_all.csv"
MODEL_SAVE_PATH = "avb3_best_model.pkl"

# =============== 数据准备 ===============
print("\n=== 数据加载 ===")
df = pd.read_csv(DATA_PATH, encoding='gbk')
print(f"原始数据维度: {df.shape}")

print("\n=== 数据预处理 ===")
X = df.drop(columns=EXCLUDE_COLS)
y = df[TARGET_COL]

for col in X.columns:
    missing_mask = X[col].isna()
    if missing_mask.sum() > 0:
        median_val = X[col].median()
        X.loc[missing_mask, col] = median_val
        print(f"列 {col}: 填充了 {missing_mask.sum()} 个缺失值")

# =============== 特征工程 ===============
print("\n=== 特征选择 ===")
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=RANDOM_STATE),
    threshold='median'
)
X_reduced = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X = X[selected_features]
print(f"特征数量: {len(selected_features)}")

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# =============== 数据划分 ===============
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)
print("\n训练集分布:", y_train.value_counts())
print("测试集分布:", y_test.value_counts())

# =============== 模型训练 ===============
print("\n=== 应用SMOTE过采样 ===")
smote = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['sqrt', 0.5, 0.3],
    'class_weight': [{0:1, 1:4}, {0:1, 1:5}, 'balanced'],
    'bootstrap': [True, False]
}

print("\n=== 开始网格搜索 ===")
rf = BalancedRandomForestClassifier(sampling_strategy='auto', replacement=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_res, y_train_res)

# =============== 阈值优化 + 预测 ===============
y_train_proba = grid_search.best_estimator_.predict_proba(X_train_res)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_train_res, y_train_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
optimal_threshold = thresholds[np.argmax(f1_scores)]

y_pred = (grid_search.best_estimator_.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)
y_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]

# 保存模型
joblib.dump(grid_search.best_estimator_, MODEL_SAVE_PATH)

# =============== 指标评估 ===============
print("\n=== 模型评估指标 ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_proba):.3f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# =============== 可视化分析 ===============
best_model = joblib.load(MODEL_SAVE_PATH)

# 1. 特征重要性
plt.figure(figsize=(12, 8))
importances = best_model.feature_importances_
sorted_idx = importances.argsort()[::-1][:20]
sns.barplot(
    x=importances[sorted_idx],
    y=X.columns[sorted_idx],
    palette='viridis'
)
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# 2. ROC 曲线
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(
    best_model, X_test, y_test, name='Optimized Model', color='darkorange'
)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.savefig('roc_curve.png', dpi=300)
plt.show()

# 3. 混淆矩阵
plt.figure(figsize=(6, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# 4. 概率分布
plt.figure(figsize=(10, 6))
sns.histplot(
    x=y_proba,
    hue=y_test,
    bins=30,
    kde=True,
    element='step',
    palette={0: 'blue', 1: 'red'}
)
plt.axvline(optimal_threshold, color='black', linestyle='--', label=f'Threshold: {optimal_threshold:.2f}')
plt.title("Predicted Probability Distribution")
plt.legend()
plt.savefig('probability_dist.png', dpi=300)
plt.show()

# 5. PR曲线
plt.figure(figsize=(8, 6))
PrecisionRecallDisplay.from_predictions(
    y_test,
    y_proba,
    name='PR Curve',
    color='purple'
)
plt.title("Precision-Recall Curve")
plt.savefig("pr_curve.png", dpi=300)
plt.show()

# 6. 预测类别分布
plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred, palette="Set2")
plt.title("Predicted Class Distribution")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.savefig("predicted_class_distribution.png", dpi=300)
plt.show()

# 7. 各类指标柱状图
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "ROC AUC": roc_auc_score(y_test, y_proba)
}

plt.figure(figsize=(8, 5))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="Set3")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Evaluation Metrics Comparison")
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=300)
plt.show()

print("\n=== 训练完成 ===")
print(f"最佳参数: {grid_search.best_params_}")
