import os
import json
from collections import defaultdict

def count_labels(json_folder):
    # 用于存储每个标签的计数
    label_counts = defaultdict(int)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(json_folder, filename)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 检查 'shapes' 字段是否存在
                if 'shapes' in data:
                    for shape in data['shapes']:
                        label = shape['label']
                        label_counts[label] += 1

    return label_counts

if __name__ == "__main__":
    # 设置包含JSON文件的文件夹路径
    json_folder = f'D:\s\liafter\json\json'

    label_counts = count_labels(json_folder)

    print("标签统计：")
    for label, count in label_counts.items():
        print(f"标签: {label}, 数量: {count}")
