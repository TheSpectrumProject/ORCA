import os

# 设置你要检查的文件夹路径
folder_path = './legit-20min'

# 统计CSV文件个数
csv_count = 0

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 只处理CSV文件
    if filename.endswith('.csv'):
        csv_count += 1

# 输出结果
print(f"Legit文件夹中的CSV文件个数: {csv_count}")

# 设置你要检查的文件夹路径
folder_path = './rise'

# 统计CSV文件个数
csv_count = 0

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 只处理CSV文件
    if filename.endswith('.csv'):
        csv_count += 1

# 输出结果
print(f"Rise文件夹中的CSV文件个数: {csv_count}")