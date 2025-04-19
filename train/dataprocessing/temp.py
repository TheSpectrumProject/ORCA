import os

# 设置文件夹路径
folder_path = './data/train/rise'  # 替换成你存储CSV文件的文件夹路径

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查是否是CSV文件
    if filename.endswith('.csv'):
        # 构造新的文件名
        new_name = 'rise_' + filename.split('_', 1)[-1]
        # 构造完整的旧文件路径和新文件路径
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        # 重命名文件
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_name}")
