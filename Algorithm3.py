import cv2
import numpy as np


def feature_match(imageA, imageB):

    sift = cv2.SIFT_create()

    kpA, desA = sift.detectAndCompute(imageA, None)
    kpB, desB = sift.detectAndCompute(imageB, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desA, desB, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return len(good_matches), good_matches, kpA, kpB


def stitch_images(imageA, imageB, matches, kpA, kpB):
    src_pts = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    height, width = imageA.shape[:2]
    result = cv2.warpPerspective(imageB, H, (width + imageB.shape[1], height))
    result[0:height, 0:width] = imageA

    return result


def unordered_image_stitching(image_list):

    R = None

    while image_list:
        X1 = image_list.pop(0)
        max_matches = 0
        best_match_img = None
        best_matches = []
        kpA, kpB = None, None

        for i, img in enumerate(image_list):
            num_matches, matches, keypointsA, keypointsB = feature_match(X1, img)
            if num_matches > max_matches:
                max_matches = num_matches
                best_match_img = img
                best_matches = matches
                kpA, kpB = keypointsA, keypointsB

        if best_match_img is not None:
            stitched_image = stitch_images(X1, best_match_img, best_matches, kpA, kpB)
            image_list.remove(best_match_img)
            image_list.append(stitched_image)

            if R is None:
                R = stitched_image
            else:
                R = stitch_images(R, stitched_image, best_matches, kpA, kpB)
        else:
            break

    return R


if __name__ == "__main__":
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    images = [cv2.imread(path) for path in image_paths]
    result = unordered_image_stitching(images)
    cv2.imshow("Stitched Image", result)
    cv2.imwrite("stitched_result.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
