
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
    tool = cv2.imread("./experiments/tool-line.jpg")
    raw_gray = cv2.cvtColor(raw, COLOR_RGB2GRAY)
    tool_gray = cv2.cvtColor(tool, COLOR_RGB2GRAY)

    # Camera Calibration (for Arducam US762C Camera Model)
    h,w = raw_gray.shape[:2]
    mtx = np.matrix([[2517.792, 0., 814.045],[0., 2514.767, 567.330],[0., 0., 1.]])
    dist = np.matrix([[-0.361044, 0.154482, 0.000808, 0.000033, 0.]])
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    raw_gray = cv2.undistort(raw_gray, mtx, dist, None, newcameramtx)
    raw_color = cv2.cvtColor(raw_gray, COLOR_GRAY2BGR)


    # Get Edge Image & Image Overlap
    canny = cv2.Canny(raw_gray, 50, 250) # min 50, max 250
    tool_edge = cv2.cvtColor(canny, COLOR_GRAY2BGR)
    h,w = raw_gray.shape[:2]
    edge = np.zeros((h,w,3), np.uint8)
    edge[np.where((tool_edge==[255,255,255]).all(axis=2))] = [0,0,255]
    dst = cv2.addWeighted(raw_color,1.0,edge,0.5,0)


    # create blank image
    #blank = np.zeros((h,w,3), np.uint8)


    #binarization
    raw_bin = cv2.adaptiveThreshold(raw_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    contrast2 = raw_bin.std()

    tool_bin = cv2.adaptiveThreshold(tool_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    tool_bin = cv2.bitwise_not(tool_bin)

    # apply contour
    contours, hierarchy = cv2.findContours(raw_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt)>500:
            cv2.drawContours(blank, [cnt], 0, (255, 0, 0), 1)  # blue

    #key point detection (ORB)---------------
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp1, des1 = orb.detectAndCompute(raw_bin, None)
    kp2, des2 = orb.detectAndCompute(tool_bin, None)
    # draw only keypoints location,not size and orientation
    raw_keypoints = cv2.drawKeypoints(raw_bin, kp1, None, color=(0,255,0), flags=0)
    tool_keypoints = cv2.drawKeypoints(tool_bin, kp2, None, color=(0,255,0), flags=0)
    #cv2.imshow("raw_keypoints", raw_keypoints)
    #cv2.imshow("tool_keypoints", tool_keypoints)

    #lsd detection
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(raw_bin)[0] #Position 0 of the returned tuple are the detected lines

    #Draw detected lines in the image
    drawn_img = lsd.drawSegments(blank,lines)

    #Show image
    cv2.imshow("LSD",drawn_img )

    #ret, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # contour
    contours, hierarchy = cv2.findContours(raw_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    # for cnt in contours:
    #     if cv2.contourArea(cnt)>500:
    #         cv2.drawContours(out, [cnt], 0, (0, 255, 0), 3)  # blue
    
    
    # for cnt in contours:
    #     epsilon = 0.02 * cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    #     print( len(approx))
    #     cv2.drawContours(raw,[approx],0,(0,255,255),5)
    

    # fast = cv2.FastFeatureDetector_create(3)
    # keypoints = fast.detect(gray, None)
    # img = cv2.drawKeypoints(raw, keypoints, None)
    

    
    while True:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break