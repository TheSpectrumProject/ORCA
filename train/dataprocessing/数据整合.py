import os
import glob
import shutil


def rename_and_copy_csv_files(src_folder, dest_folder, start_index):
    # 获取所有CSV文件路径
    csv_files = glob.glob(os.path.join(src_folder, '*.csv'))

    # 按照文件名排序
    csv_files.sort()

    # 确保目标文件夹存在
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 遍历每个文件并重命名，复制到新文件夹
    index = start_index
    for file_path in csv_files:
        # 获取文件的扩展名
        file_extension = os.path.splitext(file_path)[1]
        # 构建新的文件名，格式为 'legit_编号.csv'
        new_file_name = f"legit_{index}{file_extension}"
        # 构建新的完整路径
        new_file_path = os.path.join(dest_folder, new_file_name)

        # 复制文件并重命名到新文件夹
        shutil.copy(file_path, new_file_path)
        print(f"Copied and renamed: {file_path} -> {new_file_path}")

        # 更新编号
        index += 1

    return index  # 返回当前编号，以便下次继续递增


def main():
    # 设置源文件夹和目标文件夹路径
    folder1 = "./rise"  # 第一个文件夹
    folder2 = "./riseonly"  # 第二个文件夹
    destination_folder = "./data/train/rise"  # 目标文件夹

    # 处理第一个文件夹，编号从1开始
    start_index = 1
    start_index = rename_and_copy_csv_files(folder1, destination_folder, start_index)

    # 处理第二个文件夹，继续从上次的编号开始
    rename_and_copy_csv_files(folder2, destination_folder, start_index)


if __name__ == "__main__":
    main()
