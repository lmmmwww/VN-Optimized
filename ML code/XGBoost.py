import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("peptide_features.csv")  # 替换为你的文件名
print(df.columns.tolist())

# 要排除的列
exclude_cols = ["Sequence", "MMGBSA dG Bind"]

# 构造 X（特征） 和 y（目标变量）
X = df.drop(columns=exclude_cols)
y = df["MMGBSA dG Bind"]

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 XGBoost 模型
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Mean Squared Error: {mse:.3f}")
print(f"✅ R^2 Score: {r2:.3f}")

# 特征重要性分析
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# 显示前 15 个最重要的特征
print("\n🔥 Top 15 Important Features:")
print(importance_df.head(15))

# 可视化
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"].head(15)[::-1], importance_df["Importance"].head(15)[::-1], color='skyblue')
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()
