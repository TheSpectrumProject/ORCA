import pandas as pd

# 读取 CSV 文件，无表头，避免 DtypeWarning
input_file = 'data.csv'  # 替换成你的 CSV 文件路径
df = pd.read_csv(input_file, header=None, low_memory=False)

# 检查列数是否为 10
if df.shape[1] != 8:
    print(f"错误：CSV 文件的列数不是 10，而是 {df.shape[1]} 列。")
else:
    # 检查每个位置是否为空，并输出空值的位置
    null_positions = df.isnull()

    if null_positions.any().any():
        print("错误：CSV 文件中有空值，空值位置如下：")

        # 找出所有空值的位置
        # 使用 stack() 获取所有空值的位置
        null_indices = null_positions[null_positions == True].stack().index

        for row, col in null_indices:
            print(f"空值位置 - 行: {row + 1}, 列: {col + 1}")
    else:
        print("CSV 文件的列数为 8，且没有空值。")
