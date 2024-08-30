import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_ssim(imageA, imageB):
    # Convert images to grayscale if they are RGB
    if len(imageA.shape) == 3:
        imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    if len(imageB.shape) == 3:
        imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM between two images
    score, _ = ssim(imageA, imageB, full=True)
    return score


def calculate_psnr(imageA, imageB):
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((imageA - imageB) ** 2)

    # If MSE is zero, return infinite PSNR (images are identical)
    if mse == 0:
        return float('inf')

    # Maximum pixel value of the image (usually 255 for 8-bit images)
    max_pixel = 255.0

    # Calculate PSNR
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# Example usage
if __name__ == "__main__":
    # Load two images (replace with actual paths)
    image1 = cv2.imread('image1.jpg')
    image2 = cv2.imread('image2.jpg')

    # Ensure the images are the same size
    if image1.shape != image2.shape:
        raise ValueError("The input images must have the same dimensions.")

    # Calculate SSIM
    ssim_value = calculate_ssim(image1, image2)
    print(f"SSIM: {ssim_value}")

    # Calculate PSNR
    psnr_value = calculate_psnr(image1, image2)
    print(f"PSNR: {psnr_value}")
