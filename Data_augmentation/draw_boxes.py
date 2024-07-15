import cv2
import os

def draw_yolo_bboxes(image_dir, label_dir, save_dir, display=True):
    # 获取图像文件列表
    images = [img for img in os.listdir(image_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 定义颜色列表用于不同类别
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    for img_filename in images:
        image_path = os.path.join(image_dir, img_filename)
        label_path = os.path.join(label_dir, os.path.splitext(img_filename)[0] + '.txt')

        # 检查标注文件是否存在
        if not os.path.exists(label_path):
            print(f"No label file found for {img_filename}, skipping.")
            continue

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            continue

        h, w, _ = image.shape  # 获取图像的高度和宽度

        # 读取标注文件
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                width = float(parts[3]) * w
                height = float(parts[4]) * h

                # 计算标注框的左上角和右下角坐标
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)

                # 选择颜色
                color = colors[class_id % len(colors)]

                # 在图像上画框
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # 显示图像
        if display:
            cv2.imshow(f"Image with BBoxes - {img_filename}", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 保存图像
        save_path = os.path.join(save_dir, img_filename)
        cv2.imwrite(save_path, image)
        print(f"Saved image with bounding boxes to {save_path}")

# 使用示例
image_dir = 'D:\\s\\liafter\\json\\images\\images\\audataTEST\\limg'
label_dir = 'D:\\s\\liafter\\json\\images\\images\\audataTEST\\llable'
save_dir = 'D:\\s\\liafter\\json\\images\\images\\audataTEST\\showimg'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

draw_yolo_bboxes(image_dir, label_dir, save_dir)
