import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor


def process_image(img):
    # 对图像进行预处理（例如边缘检测）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def ransac_circle_fitting(edges):
    # 使用RANSAC进行粗略的圆形拟合，获取滚轮罐耳的中心(x, y)和半径r
    points = np.column_stack(np.where(edges > 0))
    ransac = RANSACRegressor()
    ransac.fit(points, points[:, 0] ** 2 + points[:, 1] ** 2)
    r = np.sqrt(ransac.estimator_.intercept_ / 2)
    x, y = ransac.estimator_.coef_ / 2
    return x, y, r


def extract_annular_sector_grids(img, n, s, epsilon=0):
    # 步骤1：预处理图像获取边缘
    edges = process_image(img)

    # 步骤2：使用RANSAC进行圆形拟合
    x, y, r = ransac_circle_fitting(edges)

    # 步骤3-6：计算内外半径 r_outer 和 r_inner
    annular_sectors = []
    for i in range(1, n + 2):
        r_outer = (r + epsilon) * i / n
        r_inner = (r + epsilon) * (i - 1) / n
        for j in range(s):
            S_start = j * 360 / s
            S_end = (j + 1) * 360 / s
            annular_sectors.append((r_outer, r_inner, S_start, S_end))

    return x, y, annular_sectors


# 使用实例
if __name__ == "__main__":
    img_path = "1.jpg"  # 输入图像路径
    img = cv2.imread(img_path)

    n = 10  # 环扇形的数量
    s = 8  # 角度划分的数量
    epsilon = 10  # 调整半径的偏移量

    x, y, sectors = extract_annular_sector_grids(img, n, s, epsilon)
    print("圆心坐标:", x, y)
    print("环扇形网格参数:")
    for sector in sectors:
        print("r_outer:", sector[0], "r_inner:", sector[1], "S_start:", sector[2], "S_end:", sector[3])
