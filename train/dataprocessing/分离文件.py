import pandas as pd

# 读取 CSV 文件，无表头
input_file = 'fdp1.csv'  # 替换成你的 CSV 文件路径
# 读取 CSV 文件时设置 low_memory=False
df = pd.read_csv(input_file, header=None, low_memory=False)

# 获取第一列的列索引（类别列）
category_column_index = 0

# 根据第一列的值进行分组
grouped = df.groupby(category_column_index)

# 对每个分组创建一个新的 CSV 文件
for category, group in grouped:
    # 删除原来的第一列（类别列）
    group_without_category = group.drop(columns=[category_column_index])

    # 创建新的文件名，文件名为该类别的名字
    output_file = f'{category}.csv'

    # 保存分组数据到新的 CSV 文件
    group_without_category.to_csv(output_file, header=False, index=False)

    print(f"文件 {output_file} 已创建。")
