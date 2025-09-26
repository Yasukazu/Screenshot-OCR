from collections import deque
import cv2
import numpy as np

def main(filename: str):
    src = cv2.imread(filename)

    # HSV thresholding to get rid of as much background as possible
    hsv = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(src, src, mask=mask)
    b, g, r = cv2.split(result)
    g = clahe(g, 5, (3, 3))

    # Adaptive Thresholding to isolate the bed
    img_blur = cv2.blur(g, (9, 9))
    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 51, 2)

    contours, hierarchy = cv2.findContours(img_th,
                                            cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)

    # Filter the rectangle by choosing only the big ones
    # and choose the brightest rectangle as the bed
    max_brightness = 0
    brighter_len = 4
    canvas = src.copy()
    brightest_rectangle = None
    src_whr = src.shape[1] * src.shape[0] / 8
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        if w*h > src_whr: # 40000:
            mask = np.zeros(src.shape, np.uint8)
            mask[y:y+h, x:x+w] = src[y:y+h, x:x+w]
            cropped = src[y:y+h, x:x+w]
            brightness = np.sum(mask)
            if brightness > max_brightness:
                brightest_rectangle = rect
                max_brightness = brightness
            # cv2.imshow("mask", mask)
            cv2.imshow("cropped", cropped)
            cv2.waitKey(0)

    if brightest_rectangle:
        x, y, w, h = brightest_rectangle
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("canvas", canvas)
        cv2.imwrite("result.jpg", canvas)
    cv2.waitKey(0)

def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

if __name__ == '__main__':
    from sys import argv
    main(argv[1])
