import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor


def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def ransac_circle_fitting(edges):
    points = np.column_stack(np.where(edges > 0))
    ransac = RANSACRegressor()
    ransac.fit(points, points[:, 0] ** 2 + points[:, 1] ** 2)
    r = np.sqrt(ransac.estimator_.intercept_ / 2)
    x, y = ransac.estimator_.coef_ / 2
    return x, y, r


def extract_annular_sector_grids(img, n, s, epsilon=0):
    edges = process_image(img)

    x, y, r = ransac_circle_fitting(edges)

    annular_sectors = []
    for i in range(1, n + 2):
        r_outer = (r + epsilon) * i / n
        r_inner = (r + epsilon) * (i - 1) / n
        for j in range(s):
            S_start = j * 360 / s
            S_end = (j + 1) * 360 / s
            annular_sectors.append((r_outer, r_inner, S_start, S_end))

    return x, y, annular_sectors


if __name__ == "__main__":
    img_path = "1.jpg" 
    img = cv2.imread(img_path)

    n = 10  
    s = 8 
    epsilon = 10 

    x, y, sectors = extract_annular_sector_grids(img, n, s, epsilon)
    print("圆心坐标:", x, y)
    print("环扇形网格参数:")
    for sector in sectors:
        print("r_outer:", sector[0], "r_inner:", sector[1], "S_start:", sector[2], "S_end:", sector[3])
