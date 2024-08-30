import cv2
import numpy as np


def detect_circle_size(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 查找边缘图像中的轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化最大圆形的直径
    max_diameter = 0
    max_circle = None

    # 遍历所有轮廓
    for contour in contours:
        # 使用最小外接圆拟合轮廓
        (x, y), radius = cv2.minEnclosingCircle(contour)
        diameter = 2 * radius

        # 只考虑较大的轮廓
        if diameter > max_diameter:
            max_diameter = diameter
            max_circle = (int(x), int(y), int(radius))

    # 如果找到圆形物体，则绘制并显示其宽度和高度
    if max_circle is not None:
        # 提取圆心和半径
        x, y, radius = max_circle

        # 在图像上绘制圆形
        cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        # 输出圆形物体的宽高（即直径）
        width = height = max_diameter
        print(f"圆形物体的宽度和高度为：{width:.2f} 像素")

        # 显示结果
        cv2.imshow("Detected Circle", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未检测到圆形物体。")


# 示例用法
if __name__ == "__main__":
    # 设置图像路径
    image_path = "path/to/your/image.jpg"

    # 检测圆形物体的宽高
    detect_circle_size(image_path)
