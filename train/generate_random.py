import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os

def generate_csv_files():
    try:
        # 获取用户选择的文件夹路径
        folder_path = folder_entry.get()

        # 检查文件夹路径是否存在，不存在则创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 生成10个CSV文件
        for i in range(10):
            # 随机生成行数（10到20行）
            rows = np.random.randint(10, 21)

            # 生成包含随机数的DataFrame，范围是[10, 20]，7列
            df = pd.DataFrame(np.random.randint(10, 21, size=(rows, 8)))

            # 生成文件名
            filename = f"random_data_{i + 1}.csv"

            # 保存到指定文件夹
            file_path = os.path.join(folder_path, filename)
            df.to_csv(file_path, index=False)

        messagebox.showinfo("Success", "CSV files generated successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def choose_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        folder_entry.delete(0, tk.END)
        folder_entry.insert(0, folder_selected)

# 创建GUI窗口
root = tk.Tk()
root.title("CSV File Generator")

# 创建标签和输入框
tk.Label(root, text="Select the folder to save CSV files:").pack(pady=10)

# 创建输入框
folder_entry = tk.Entry(root, width=50)
folder_entry.pack(pady=5)

# 创建选择按钮
choose_button = tk.Button(root, text="Choose Folder", command=choose_folder)
choose_button.pack(pady=5)

# 创建生成按钮
generate_button = tk.Button(root, text="Generate CSV Files", command=generate_csv_files)
generate_button.pack(pady=20)

# 运行GUI主循环
root.mainloop()
