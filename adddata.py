import pandas as pd
import numpy as np


def interpolate_columns(file_path, sheet_name=0, num_points=5):
    """
    对 Excel 文件中的每一列进行线性插值。

    Args:
        file_path (str): Excel 文件路径。
        sheet_name (int/str): 表名或索引，默认读取第一个表。
        num_points (int): 每对相邻点之间的插值数量，包括两端点。

    Returns:
        pd.DataFrame: 插值后的数据表。
    """
    # 读取 Excel 文件
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 存储插值结果
    interpolated_data = {}

    # 对每一列进行插值
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:  # 仅对数值列插值
            col_values = df[col].dropna().values  # 去掉缺失值
            interpolated_col = []

            # 对每一对相邻点进行插值
            for i in range(len(col_values) - 1):
                start, end = col_values[i], col_values[i + 1]
                interpolated_col.extend(np.linspace(start, end, num_points, endpoint=False).tolist())

            # 添加最后一个点
            interpolated_col.append(col_values[-1])

            # 存入结果字典
            interpolated_data[col] = interpolated_col
        else:
            # 对非数值列，直接复制
            interpolated_data[col] = df[col].tolist()

    # 转换为 DataFrame
    interpolated_df = pd.DataFrame(interpolated_data)
    return interpolated_df


# 示例用法
file_path = r"C:\Users\Desktop\part\humanact\data1.xlsx"  # 替换为实际文件路径
sheet_name = 0  # 可以指定表名或索引
num_points = 30  # 两点之间插值数量，包括两端点

# 运行插值函数
result = interpolate_columns(file_path, sheet_name, num_points)

# 显示结果
print(result)

# 保存结果到新的 Excel 文件
result.to_excel(r"C:\Users\Desktop\part\humanact\data2.xlsx", index=False)
