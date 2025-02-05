import cv2
import numpy as np
import os

def calculate_speed(prev_frame, curr_frame):

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    avg_speed = np.mean(magnitude)
    return avg_speed

def detect_occluded_images(reference_image_path, images_folder, speed_threshold=5.0):


    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        print(f"无法读取参考图像: {reference_image_path}")
        return

    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png'))])
    previous_image = reference_image
    for i, image_filename in enumerate(image_files):
        image_path = os.path.join(images_folder, image_filename)
        current_image = cv2.imread(image_path)

        if current_image is None:
            print(f"无法读取图像: {image_path}")
            continue

        speed = calculate_speed(previous_image, current_image)

        if speed < speed_threshold:
            print(f"图像 {image_filename} 被检测为可能的遮挡区域，平均速度: {speed:.2f}")

        previous_image = current_image

if __name__ == "__main__":
    reference_image_path = 'path/to/reference_image.jpg'
    images_folder = 'path/to/images_folder'
    detect_occluded_images(reference_image_path, images_folder, speed_threshold=5.0)
