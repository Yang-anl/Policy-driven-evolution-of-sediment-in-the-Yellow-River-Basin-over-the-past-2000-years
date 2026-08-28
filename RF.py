import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import joblib

# 读取Excel文件
df = pd.read_excel(r"C:\Users\Desktop\part\humanact\data5.xlsx", engine='openpyxl')

# 选择特征和目标变量
X = df.drop(['re'], axis=1)  # 特征列
y = df['re'].values  # 目标变量

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 设置K折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# 存储每一轮的评价指标
mae_scores = []
rmse_scores = []
r2_scores = []

# 确保保存模型的目录存在
model_save_dir = 'saved_models'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# 保存标准化器
scaler_save_path = os.path.join(model_save_dir, f'scaler_RF.pkl')
joblib.dump(scaler, scaler_save_path)

# K折交叉验证
for fold, (train_index, test_index) in enumerate(kf.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 构建随机森林模型
    model = RandomForestRegressor(n_estimators=60, random_state=3)

    # 训练模型
    model.fit(X_train, y_train)

    # 保存模型
    model_save_path = os.path.join(model_save_dir, f'RFmodel_fold_{fold}.pkl')
    joblib.dump(model, model_save_path)

    # 进行预测
    y_pred = model.predict(X_test)

    # 计算评价指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 存储结果
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    # 调试输出
    print(f"Fold {fold} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

# 确保每一轮都有数据
if len(mae_scores) != 5:
    raise ValueError("MAE scores list does not have 5 elements.")
if len(rmse_scores) != 5:
    raise ValueError("RMSE scores list does not have 5 elements.")
if len(r2_scores) != 5:
    raise ValueError("R2 scores list does not have 5 elements.")

results = pd.DataFrame({
    'Fold': [1, 2, 3, 4, 5],
    'MAE': mae_scores,
    'RMSE': rmse_scores,
    'R^2': r2_scores
})

# 保存到Excel文件
results.to_excel(r'C:\Users\Desktop\part\humanact\result\rf_results.xlsx', index=False)
print("结果已保存到 Excel 文件")

# 创建图形和子图数组，1行4列，横向排列
fig, axs = plt.subplots(1, 4, figsize=(20, 4))  # 宽度20英寸，高度5英寸

# MAE图，柱状图
colors = ['plum', 'Coral', 'Khaki', 'SteelBlue', 'Gray']  # 定义颜色列表
for i in range(5):
    axs[0].bar(i + 1, mae_scores[i], color=colors[i], width=0.6)

axs[0].set_title('MAE')
axs[0].set_xlabel('Fold')
axs[0].set_ylabel('MAE')
axs[0].grid(True)

# RMSE图，柱状图
for i in range(5):
    axs[1].bar(i + 1, rmse_scores[i], color=colors[i], width=0.6)

axs[1].set_title('RMSE')
axs[1].set_xlabel('Fold')
axs[1].set_ylabel('RMSE')
axs[1].set_ylim([0, np.max(rmse_scores) * 1.1])
axs[1].grid(True)

# R^2图，柱状图
for i in range(5):
    axs[2].bar(i + 1, r2_scores[i], color=colors[i], width=0.6)

axs[2].set_title('R^2')
axs[2].set_xlabel('Fold')
axs[2].set_ylabel('R^2')
axs[2].grid(True)

# 散点图
axs[3].scatter(y_test, y_pred, color='red')
axs[3].set_title('DT Prediction')
axs[3].set_xlabel('Real')
axs[3].set_ylabel('Predicted')

# 在图上添加 RMSE, MAE, R^2 信息
plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.text(0.05, 0.90, f'MAE: {mae:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.text(0.05, 0.85, f'R^2: {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.tight_layout()
# plt.savefig(r'C:\Users\wangchang\Desktop\part\humanact\result\RF.png', dpi=300, bbox_inches='tight')
plt.show()

# 热力图
plt.figure(figsize=(8, 6))  # 设置热力图大小
correlation_matrix = pd.DataFrame(X_scaled, columns=X.columns).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=0.5, linecolor='white', cbar=True)
plt.title('Feature Correlation Heatmap')
# plt.savefig(r'C:\Users\wangchang\Desktop\part\humanact\result\RF_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 生成特征与目标变量的关系散点图
features = X.columns  # 获取所有特征列名
fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # 设置2行3列子图布局

# 遍历每个特征，绘制散点图
for idx, feature in enumerate(features):
    ax = axs[idx // 3, idx % 3]  # 获取当前子图位置
    scatter = ax.scatter(X[feature], y, c=y, cmap='viridis', alpha=0.7)  # 绘制散点图
    ax.set_title(f"{feature} vs Target")  # 设置标题
    ax.set_xlabel(feature)  # 设置X轴标签
    ax.set_ylabel('tav')  # 设置Y轴标签
    plt.colorbar(scatter, ax=ax)  # 添加颜色条

plt.tight_layout()
plt.savefig(r'C:\Users\Desktop\part\humanact\result\RF_feature_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 添加特征重要性分析
feature_importances = np.mean([model.feature_importances_ for fold in range(1, kf.n_splits + 1)], axis=0)

# 创建特征重要性 DataFrame
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# 打印和保存特征重要性
importance_save_path = r'C:\Users\Desktop\part\humanact\result\RF_feature_importance.xlsx'
importance_df.to_excel(importance_save_path, index=False)
print(f"特征重要性已保存到: {importance_save_path}")

# 可视化特征重要性
plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance in Random Forest', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()

# 保存特征重要性图像
feature_importance_plot_path = r'C:\Users\Desktop\part\humanact\result\RF_feature_importance.png'
plt.savefig(feature_importance_plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"特征重要性图已保存到: {feature_importance_plot_path}")
