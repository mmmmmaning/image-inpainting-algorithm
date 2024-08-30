import cv2
import numpy as np

# 全局变量
ref_point = []  # 存储选择框的坐标
cropping = False  # 是否正在裁剪

def select_area(event, x, y, flags, param):
    # 访问全局变量
    global ref_point, cropping

    # 当左键单击时记录起始点，并开始裁剪
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    # 检查是否左键释放，以确定裁剪的区域
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # 绘制矩形
        cv2.rectangle(image, ref_point[0], ref_point[1], (255, 255, 255), 2)
        cv2.imshow("image", image)

# 加载图像
image_path = 'paper/bgz.png'  # 替换为你的图像文件路径
image = cv2.imread(image_path)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", select_area)

# 保持图像窗口打开，直到按下 'q' 键
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # 如果按下 'r' 键，则重置裁剪区域
    if key == ord("r"):
        image = clone.copy()


    # 如果按下 'c' 键，则裁剪区域并显示
    elif key == ord("c"):
        if len(ref_point) == 2:
            # 创建掩模
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, ref_point[0], ref_point[1],
                          255, -1)
            image[mask == 255] = [255, 255, 255]  # 将选定区域置为白色
            ref_point = []  # 重置坐标点


    # 如果按下 'q' 键，则退出循环并保存图像
    elif key == ord("q"):
        # 保存图像到文件
        save_path = 'paper/kuang5.jpg'  # 替换为你的保存文件路径
        cv2.imwrite(save_path, image)
        break

cv2.destroyAllWindows()
