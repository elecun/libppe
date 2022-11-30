'''
Tool-tip detector from image
'''

import cv2
from cv2 import COLOR_GRAY2BGR
import numpy as np
from cv2 import COLOR_RGB2GRAY
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load Images
    raw = cv2.imread("./experiments/0216_tool_1/25.jpg")

    # convert grayscale
    raw_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    # Camera Calibration (for Arducam US762C Camera Model)
    h,w = raw_gray.shape[:2]
    mtx = np.matrix([[2517.792, 0., 814.045],[0., 2514.767, 567.330],[0., 0., 1.]])
    dist = np.matrix([[-0.361044, 0.154482, 0.000808, 0.000033, 0.]])
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    raw_gray = cv2.undistort(raw_gray, mtx, dist, None, newcameramtx)
    raw_color = cv2.cvtColor(raw_gray, COLOR_GRAY2BGR)

    # apply smoothing
    raw_blur = cv2.GaussianBlur(raw_gray, (9,9), 1)
    raw_canny = cv2.Canny(raw_blur, 20, 60)

    #kernel = np.ones((2))
    #raw_canny = cv2.dilate(raw_canny, kernel, iterations=1)

    # find contour
    contours, hierarchy = cv2.findContours(raw_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    hull = cv2.convexHull(cnt)

    cv2.drawContours(raw_color, [hull], -1, (255, 0, 0), 2)

    M = cv2.moments(cnt)

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cv2.circle(img_contour, (cx, cy), 20, (255,0,255))

    kernel = np.ones((2))
    raw_dilated = cv2.dilate(raw_canny, kernel, iterations=1)
    


