import os
import json
import shutil

# 定义文件夹路径
folder_path = r'D:\jinju\json'

# 获取文件夹内所有的 JSON 文件路径
json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]

# 遍历每个 JSON 文件
for json_file in json_files:
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 遍历每个形状
    for shape in data['shapes']:
        # 检查点数是否小于等于 2
        if shape['shape_type'] == 'polygon' and len(shape['points']) <= 2:
            # 输出文件名和形状名称
            print(f"File: {json_file}, Shape: {shape['label']}")

            # 获取文件名（不包含路径）
            file_name = os.path.basename(json_file)

            # 移动文件到上层目录
            destination_path = os.path.join(os.path.dirname(folder_path), file_name)
            shutil.move(json_file, destination_path)
            print(f"Moved file to: {destination_path}")

# 提示处理完成
print("All files processed.")
