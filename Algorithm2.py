import cv2
import numpy as np
from skimage.metrics import normalized_mutual_information as nmi
mu =**
theta = ***
max_leaf_nodes = **
def mutual_information(img1, img2):

    return nmi(img1, img2)
def extract_annular_sector_grids(img, n, P=0, epsilon=0):
    s=2**P
    x, y, r = 100, 100, 50
    annular_sectors = []
    for i in range(1, n + 2):
        r_outer = (r + epsilon) * i / n
        r_inner = (r + epsilon) * (i - 1) / n
        for j in range(s):
            S_start = j * 360 / s
            S_end = (j + 1) * 360 / s
            annular_sectors.append((r_outer, r_inner, S_start, S_end))
    return x, y, annular_sectors , s


def binary_tree_occlusion_detection(A, B):

    x, y, grids = extract_annular_sector_grids(A, n=3, P=0)
    k = 0
    max_leaf_nodes = 4
    template_img = A.copy()

    while k < max_leaf_nodes:
        for i, grid in enumerate(grids):
            I1 = A
            I2 = B
            M_value = mutual_information(I1, I2)

            if M_value > mu:
                return 0
            elif M_value < theta:
                return template_img
            else:
                k += 1
                P=k
    return template_img


if __name__ == "__main__":
    A = cv2.imread("image_A.jpg", 0)
    B = cv2.imread("image_B.jpg", 0)

    result = binary_tree_occlusion_detection(A, B)

    if isinstance(result, np.ndarray):
        print("Detected occluded region.")
        cv2.imwrite("occluded_region.jpg", result)
    else:
        print("No occlusion detected.")
