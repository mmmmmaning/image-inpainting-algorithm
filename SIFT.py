import cv2
import os

def sift_feature_match(img1, img2, threshold=50):

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)


    matches = sorted(matches, key=lambda x: x.distance)


    return len(matches)

def detect_occluded_images(reference_image_path, images_folder, output_folder, match_threshold=50):

    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if reference_image is None:
        print(f"无法读取参考图像: {reference_image_path}")
        return

    os.makedirs(output_folder, exist_ok=True)


    for image_filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            print(f"无法读取图像: {image_path}")
            continue

        match_count = sift_feature_match(reference_image, image, match_threshold)

        if match_count < match_threshold:
            print(f"图像 {image_filename} 被检测为遮挡区域，匹配点数: {match_count}")
            output_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(output_path, image)


if __name__ == "__main__":
    reference_image_path = 'path/to/reference_image.jpg'
    images_folder = 'path/to/images_folder'
    output_folder = 'path/to/output_folder'
    detect_occluded_images(reference_image_path, images_folder, output_folder, match_threshold=50)
