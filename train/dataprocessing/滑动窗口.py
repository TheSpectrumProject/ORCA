import pandas as pd
import os


def slide_window_and_save(input_csv, output_dir, window_size=40, slide_size=20, prefix='fdp_'):
    # 读取没有表头的 CSV 文件
    df = pd.read_csv(input_csv, header=None)

    # 如果输出目录不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取数据的总行数
    total_rows = len(df)

    # 滑动窗口处理
    start = 0
    index = 0
    while start + window_size <= total_rows:
        # 获取当前窗口的数据
        window_data = df.iloc[start:start + window_size]

        # 构造输出文件的路径和名称
        output_file = os.path.join(output_dir, f"{prefix}{index}.csv")

        # 将当前窗口的数据保存为 CSV 文件，不包含表头
        window_data.to_csv(output_file, index=False, header=False)

        # 打印保存的文件路径（可选）
        print(f"保存文件: {output_file}")

        # 滑动窗口
        start += slide_size
        index += 1


# 示例使用
input_csv = 'fdp.csv'  # 输入 CSV 文件路径
output_dir = './fdp'  # 输出文件夹路径
slide_window_and_save(input_csv, output_dir)
