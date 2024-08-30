import cv2
import os

def sift_feature_match(img1, img2, threshold=50):
    """
    使用 SIFT 进行特征点匹配，并返回匹配的特征点数量
    """
    # 创建 SIFT 对象
    sift = cv2.SIFT_create()

    # 计算图像的特征点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 使用 BFMatcher 进行特征点匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # 根据描述符的距离排序匹配结果
    matches = sorted(matches, key=lambda x: x.distance)

    # 返回匹配点数量
    return len(matches)

def detect_occluded_images(reference_image_path, images_folder, output_folder, match_threshold=50):
    """
    使用 SIFT 特征点匹配来检测遮挡区域图像
    """
    # 读取参考图像
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if reference_image is None:
        print(f"无法读取参考图像: {reference_image_path}")
        return

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历待检测的图像
    for image_filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 如果无法读取图像，则跳过
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 计算特征点匹配数量
        match_count = sift_feature_match(reference_image, image, match_threshold)

        # 如果匹配点数少于阈值，则认为是目标图像并保存
        if match_count < match_threshold:
            print(f"图像 {image_filename} 被检测为遮挡区域，匹配点数: {match_count}")
            output_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(output_path, image)

# 示例用法
if __name__ == "__main__":
    # 设定参考图像路径
    reference_image_path = 'path/to/reference_image.jpg'

    # 设置待检测的图像文件夹
    images_folder = 'path/to/images_folder'

    # 设置输出文件夹，用于保存目标图像
    output_folder = 'path/to/output_folder'

    # 调用检测函数
    detect_occluded_images(reference_image_path, images_folder, output_folder, match_threshold=50)
