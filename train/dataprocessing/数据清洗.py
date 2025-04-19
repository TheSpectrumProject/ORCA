import os
import pandas as pd

# 设置你要检查的文件夹路径
folder_path = './fdp'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 只处理CSV文件
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 检查是否有足够的列
            if df.shape[1] >= 7:
                # 获取第4、7、6列的数据（注意索引从0开始）
                col_4_sum = df.iloc[:, 3].sum()  # 第4列
                col_7_sum = df.iloc[:, 6].sum()  # 第7列
                col_6_sum = df.iloc[:, 5].sum()  # 第6列

                # 如果任意一列的和小于3，删除该CSV文件
                if col_4_sum < 1 or col_7_sum < 3 or col_6_sum < 2:
                    os.remove(file_path)
                    print(f"删除文件: {filename}")
        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}")
