import cv2
import numpy as np
import os

def calculate_speed(prev_frame, curr_frame):
    """
    使用光流法计算相邻帧之间的速度信息
    """
    # 将图像转换为灰度
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # 使用光流法计算速度
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 计算速度的大小（模长）
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 返回平均速度
    avg_speed = np.mean(magnitude)
    return avg_speed

def detect_occluded_images(reference_image_path, images_folder, speed_threshold=5.0):
    """
    使用速度信息来检测遮挡区域图像
    """
    # 读取参考图像作为第一帧
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        print(f"无法读取参考图像: {reference_image_path}")
        return

    # 读取文件夹中的所有图像并排序
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png'))])
    previous_image = reference_image

    # 遍历图像序列
    for i, image_filename in enumerate(image_files):
        image_path = os.path.join(images_folder, image_filename)
        current_image = cv2.imread(image_path)

        # 如果无法读取图像，则跳过
        if current_image is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 计算当前帧与上一帧之间的速度
        speed = calculate_speed(previous_image, current_image)

        # 如果速度超过设定阈值，则可能是遮挡区域
        if speed < speed_threshold:
            print(f"图像 {image_filename} 被检测为可能的遮挡区域，平均速度: {speed:.2f}")

        # 更新上一帧
        previous_image = current_image

# 示例用法
if __name__ == "__main__":
    # 设定参考图像路径
    reference_image_path = 'path/to/reference_image.jpg'

    # 设置待检测的图像文件夹
    images_folder = 'path/to/images_folder'

    # 调用检测函数，设定速度阈值为5.0（根据实际情况调整）
    detect_occluded_images(reference_image_path, images_folder, speed_threshold=5.0)
