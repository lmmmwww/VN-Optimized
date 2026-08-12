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
                             RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve)
from sklearn.preprocessing import StandardScaler

# =============== 配置参数 ===============
RANDOM_STATE = 42
TARGET_COL = "MMGBSA dG Bind"  # 修改为实际目标列名
EXCLUDE_COLS = [ "Sequence", "MMGBSA dG Bind"]  # 非特征列
DATA_PATH = "G:\\VX\\特征筛选\\avb3_peptide_features.csv"  # 数据文件路径
MODEL_SAVE_PATH = "avb3_best_model.pkl"

# =============== 数据准备 ===============
# 1. 加载数据
print("\n=== 数据加载 ===")
df = pd.read_csv(DATA_PATH, encoding='gbk')
print(f"原始数据维度: {df.shape}")

# 2. 数据预处理
print("\n=== 数据预处理 ===")
X = df.drop(columns=EXCLUDE_COLS)
y = df[TARGET_COL]

# 处理缺失值（仅处理真正缺失值）
for col in X.columns:
    missing_mask = X[col].isna()
    if missing_mask.sum() > 0:
        median_val = X[col].median()
        X.loc[missing_mask, col] = median_val
        print(f"列 {col}: 填充了 {missing_mask.sum()} 个缺失值")

# =============== 特征工程 ===============
# 特征选择
print("\n=== 特征选择 ===")
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=RANDOM_STATE),
    threshold='median'
)
X_reduced = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X = X[selected_features]
print(f"特征数量: {len(selected_features)}")

# 标准化
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
# 过采样处理
print("\n=== 应用SMOTE过采样 ===")
smote = SMOTE(
    sampling_strategy=0.5,
    k_neighbors=5,
    random_state=RANDOM_STATE
)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 参数网格
param_grid = {
    'n_estimators': [200, 300, 400],  # 增加树的数量
    'max_depth': [10, 15, 20],       # 限制最大深度
    'min_samples_split': [5, 10, 15],# 增强正则化
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['sqrt', 0.5, 0.3],  # 更严格的特征选择
    'class_weight': [
        {0:1, 1:4},  # 增强少数类权重
        {0:1, 1:5},
        'balanced'
    ],
    'bootstrap': [True, False]  # 新增参数
}

# 网格搜索
print("\n=== 开始网格搜索 ===")
rf = BalancedRandomForestClassifier(
    sampling_strategy='auto',
    replacement=True,
    random_state=RANDOM_STATE
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_res, y_train_res)

# =============== 模型评估 ===============
# 阈值优化
y_train_proba = grid_search.best_estimator_.predict_proba(X_train_res)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_train_res, y_train_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# 测试集预测
y_pred = (grid_search.best_estimator_.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)

# 保存模型
joblib.dump(grid_search.best_estimator_, MODEL_SAVE_PATH)

# =============== 可视化分析 ===============
# 加载模型
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

# 2. ROC曲线
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(
    best_model,
    X_test,
    y_test,
    name='Optimized Model',
    color='darkorange'
)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.savefig('roc_curve.png', dpi=300)
plt.show()

# 3. 混淆矩阵
plt.figure(figsize=(6, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# 4. 概率分布
plt.figure(figsize=(10, 6))
y_proba = best_model.predict_proba(X_test)[:, 1]
sns.histplot(
    x=y_proba,
    hue=y_test,
    bins=30,
    kde=True,
    element='step',
    palette={0: 'blue', 1: 'red'}
)
plt.axvline(optimal_threshold, color='black', linestyle='--',
           label=f'Threshold: {optimal_threshold:.2f}')
plt.title("Predicted Probability Distribution")
plt.legend()
plt.savefig('probability_dist.png', dpi=300)
plt.show()

print("\n=== 训练完成 ===")
print(f"最佳参数: {grid_search.best_params_}")
print(f"测试集F1: {f1_score(y_test, y_pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")