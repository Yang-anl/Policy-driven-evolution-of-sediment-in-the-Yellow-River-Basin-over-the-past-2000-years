import pandas as pd
import joblib
import numpy as np


def predict_with_removed_factors(input_file, output_file, feature_columns, remove_columns, scaler_path, rf_model_path):
    """
    使用模型计算去除指定因素后的预测结果。

    Args:
        input_file (str): 输入归一化后的 Excel 文件路径。
        output_file (str): 输出结果的 Excel 文件路径。
        feature_columns (list): 用于预测的所有特征列。
        remove_columns (list): 需要去除的特征列名称。
        scaler_path (str): 归一化模型文件路径。
        rf_model_path (str): 训练好的模型文件路径。
    """
    # 1. 加载模型
    scaler = joblib.load(scaler_path)  # 归一化模型
    rf_model = joblib.load(rf_model_path)  # 随机森林模型

    # 2. 读取输入文件
    data = pd.read_excel(input_file)

    # 确保输入数据包含指定列
    if not set(feature_columns).issubset(data.columns):
        raise ValueError("特征列与 Excel 数据列名不匹配，请检查 feature_columns 参数")

    # 3. 创建去除因素的数据副本
    adjusted_data = data.copy()

    # 将去除的因素设置为常量（如均值或0）
    for col in remove_columns:
        if col in adjusted_data.columns:
            adjusted_data[col] = 0  # 或者 np.mean(data[col])

    # 4. 使用模型进行预测
    predictions = rf_model.predict(adjusted_data[feature_columns])

    # 5. 将预测结果保存到新的 Excel 文件
    adjusted_data['Predicted_Result'] = predictions
    adjusted_data.to_excel(output_file, index=False)
    print(f"去除因素后的预测结果已保存到文件: {output_file}")


# 示例用法
input_file = r"C:\Users\Desktop\part\humanact\data2.xlsx"  # 输入归一化后的数据文件
output_file = r"C:\Users\Desktop\part\humanact\adjusted_predictions.xlsx"  # 输出结果文件
feature_columns = ['a', 'b', 'c', 'd', 'e', 'f']  # 所有特征列
remove_columns = ['d', 'e']  # 去除的两个因素
scaler_path = r'C:\Users\PycharmProjects\PythonProject\hunmanact\saved_models\scaler_RF.pkl'   # 归一化模型文件
rf_model_path = r'C:\Users\PycharmProjects\PythonProject\hunmanact\saved_models\RFmodel_fold_5.pkl'  # 随机森林模型文件

# 调用函数
predict_with_removed_factors(input_file, output_file, feature_columns, remove_columns, scaler_path, rf_model_path)
