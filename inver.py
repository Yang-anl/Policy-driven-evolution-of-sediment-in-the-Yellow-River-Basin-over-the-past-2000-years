import pandas as pd
import joblib  # 用于加载保存的模型
from sklearn.preprocessing import StandardScaler


def predict_and_save(input_excel, output_excel, feature_columns, scaler_path, rf_model_path):
    """
    从 Excel 文件读取数据，进行预测，并将结果写入新的 Excel 文件。

    Args:
        input_excel (str): 输入 Excel 文件路径。
        output_excel (str): 输出 Excel 文件路径。
        feature_columns (list): 特征列名称列表。
        scaler_path (str): 标准化模型文件路径。
        rf_model_path (str): 随机森林模型文件路径。
    """
    # 1. 加载模型
    scaler = joblib.load(scaler_path)  # 标准化模型
    rf_model = joblib.load(rf_model_path)  # 随机森林模型

    # 2. 读取输入 Excel 文件
    data = pd.read_excel(input_excel)

    # 检查特征列是否存在
    if not set(feature_columns).issubset(data.columns):
        raise ValueError("特征列与 Excel 数据列名不匹配，请检查 feature_columns 参数")

    # 3. 数据标准化
    data_to_predict = scaler.transform(data[feature_columns])  # 标准化特征列

    # 4. 进行预测
    predictions = rf_model.predict(data_to_predict)

    # 5. 将预测结果添加到 DataFrame 中
    data['re'] = predictions


    # 6. 将结果保存到新的 Excel 文件
    data.to_excel(output_excel, index=False)
    print(predictions)
    print(f"预测结果已保存到文件: {output_excel}")


# 示例用法
input_excel = r"C:\Users\Desktop\part\humanact\data4.xlsx"  # 输入文件路径
output_excel = r"C:\Users\Desktop\part\humanact\predicted_results.xlsx"  # 输出文件路径
feature_columns = ['a', 'b', 'c', 'f']  # 替换为你的特征列
scaler_path = r'C:\Users\PycharmProjects\PythonProject\hunmanact\saved_models\scaler_RF.pkl' # 标准化模型路径
rf_model_path = r'C:\Users\PycharmProjects\PythonProject\hunmanact\saved_models\RFmodel_fol.pkl' # 随机森林模型路径

predict_and_save(input_excel, output_excel, feature_columns, scaler_path, rf_model_path)
