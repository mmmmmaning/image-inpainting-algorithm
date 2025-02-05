import cv2
import numpy as np


ref_point = []  
cropping = False  

def select_area(event, x, y, flags, param):

    global ref_point, cropping


    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        cv2.rectangle(image, ref_point[0], ref_point[1], (255, 255, 255), 2)
        cv2.imshow("image", image)

image_path = 'paper/bgz.png' 
image = cv2.imread(image_path)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", select_area)
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        image = clone.copy()

    elif key == ord("c"):
        if len(ref_point) == 2:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, ref_point[0], ref_point[1],
                          255, -1)
            image[mask == 255] = [255, 255, 255]  
            ref_point = [] 



    elif key == ord("q"):
        save_path = 'paper/kuang5.jpg' 
        cv2.imwrite(save_path, image)
        break

cv2.destroyAllWindows()
