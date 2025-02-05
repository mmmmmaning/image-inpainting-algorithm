import cv2
import numpy as np


def detect_circle_size(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_diameter = 0
    max_circle = None

    for contour in contours:

        (x, y), radius = cv2.minEnclosingCircle(contour)
        diameter = 2 * radius


        if diameter > max_diameter:
            max_diameter = diameter
            max_circle = (int(x), int(y), int(radius))

    # 如果找到圆形物体，则绘制并显示其宽度和高度
    if max_circle is not None:

        x, y, radius = max_circle


        cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)


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

    image_path = "path/to/your/image.jpg"

    detect_circle_size(image_path)
