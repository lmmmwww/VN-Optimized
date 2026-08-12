# 新增库导入
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve)
from sklearn.preprocessing import StandardScaler

# 1. 加载数据
df = pd.read_csv("G:\\VX\\特征筛选\\avb3_peptide_features.csv", encoding='gbk')  # 替换为你的文件路径
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


# 2.3 新增特征筛选（示例）
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42),
    threshold='median'  # 选择重要性高于中位数的特征
)
X_reduced = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print(f"\n特征筛选结果: 从{len(X.columns)}减少到{len(selected_features)}个特征")
X = X[selected_features]  # 覆盖原始特征矩阵

# 2.4 新增标准化处理（针对树模型可选步骤）
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 3. 修改模型结构和参数网格
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

# 改用BalancedRandomForestClassifier
rf = BalancedRandomForestClassifier(
    sampling_strategy='auto',  # 自动平衡类别采样
    replacement=True,          # 允许重复采样
    random_state=42
)

# 新增过采样处理
print("\n应用SMOTE过采样...")
smote = SMOTE(
    sampling_strategy=0.5,  # 设置少数类达到多数类的50%
    k_neighbors=5,
    random_state=42
)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 修改网格搜索配置
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),  # 分层交叉验证
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

print("\n开始优化版网格搜索...")
grid_search.fit(X_train_res, y_train_res)  # 使用过采样后的数据

# 新增模型解释模块
print("\n=== 特征重要性分析 ===")
best_model = grid_search.best_estimator_
importances = best_model.feature_importances_
sorted_idx = importances.argsort()[::-1]


# 新增阈值优化部分
print("\n执行阈值优化...")
y_train_proba = best_model.predict_proba(X_train_res)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_train_res, y_train_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# 应用最优阈值
y_pred_optimized = (best_model.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)

# 4. 保存最佳模型
import joblib
joblib.dump(grid_search.best_estimator_, 'best_model.pkl')


# 更新评估指标
accuracy = accuracy_score(y_test, y_pred_optimized)
precision = precision_score(y_test, y_pred_optimized)
recall = recall_score(y_test, y_pred_optimized)
f1 = f1_score(y_test, y_pred_optimized)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

print("\n=== 优化后测试集性能 ===")
print(f"Optimal Threshold: {optimal_threshold:.3f}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred_optimized))
