import csv

# 检查每行数据是否符合要求的函数
def is_valid_row(row):
    if len(row) != 8:  # 检查行是否有8个元素
        return False
    if not isinstance(row[0], str) or not row[0]:  # 第一个元素必须是非空字符串
        return False
    # 后面的7个元素必须是数字
    for value in row[1:]:
        try:
            float(value)  # 尝试将值转换为浮动数字
        except ValueError:
            return False
    return True

# 读取CSV文件并筛选符合条件的行
def filter_csv(input_filename, output_filename):
    valid_rows = []
    with open(input_filename, newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if is_valid_row(row):
                valid_rows.append(row)
            else:
                print(f"无效行: {row}")

    # 将符合条件的行写入新的CSV文件
    with open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(valid_rows)
    print(f"已保存有效数据到: {output_filename}")

# 使用示例
input_filename = 'fdp.csv'  # 输入的CSV文件
output_filename = 'fdp1.csv'  # 输出的CSV文件
filter_csv(input_filename, output_filename)
