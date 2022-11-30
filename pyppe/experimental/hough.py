
'''
Linea Feature with Hough Line
'''

import cv2
from cv2 import COLOR_GRAY2BGR
import numpy as np
from cv2 import COLOR_RGB2GRAY
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load Images
    raw1 = cv2.imread("./pyppe/experimental/capture1.png")
    raw2 = cv2.imread("./pyppe/experimental/capture2.png")

    print("Raw Image #1 Shape : {}".format(raw1.shape))
    print("Raw Image #2 Shape : {}".format(raw2.shape))

    h, w, _ = raw1.shape
    mtx = np.matrix([[2517.792, 0., 814.045],[0., 2514.767, 567.330],[0., 0., 1.]])
    dist = np.matrix([[-0.361044, 0.154482, 0.000808, 0.000033, 0.]])
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))

    raw1 = cv2.undistort(raw1, mtx, dist, None, newcameramtx)
    raw2 = cv2.undistort(raw2, mtx, dist, None, newcameramtx)

    print("Undistorted Image #1 Shape : {}".format(raw1.shape))
    print("Undistorted Image #2 Shape : {}".format(raw2.shape))

    raw1_gray = cv2.cvtColor(raw1, COLOR_RGB2GRAY)
    raw2_gray = cv2.cvtColor(raw2, COLOR_RGB2GRAY)

    # edge aware filtering
    raw1_gray = cv2.bilateralFilter(raw1_gray, -1, 10, 55)
    raw2_gray = cv2.bilateralFilter(raw2_gray, -1, 10, 55)

    _, raw1_binary = cv2.threshold(raw1_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, raw2_binary = cv2.threshold(raw2_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    raw1_edge = cv2.Canny(raw1_binary, 10, 5)
    raw2_edge = cv2.Canny(raw2_binary, 10, 5)

    cv2.imwrite("raw1_edge.png", raw1_edge)
    cv2.imwrite("raw2_edge.png", raw2_edge)

    raw2_edge_color = cv2.cvtColor(raw2_edge, COLOR_GRAY2BGR)

    raw2_lines = cv2.HoughLines(raw2_edge, 2, np.pi/180, 190)
    
    for line in raw2_lines: 
        r, theta = line[0] # 거리와 각도
        tx, ty = np.cos(theta), np.sin(theta) # x, y축에 대한 삼각비
        x0, y0 = tx*r, ty*r  #x, y 기준(절편) 좌표

        # 기준 좌표에 빨강색 점 그리기
        cv2.circle(raw2_edge_color, (abs(int(x0)), abs(int(y0))), 3, (0,0,255), -1)

        # 직선 방정식으로 그리기 위한 시작점, 끝점 계산
        x1, y1 = int(x0 + w*(-ty)), int(y0 + h * tx)
        x2, y2 = int(x0 - w*(-ty)), int(y0 - h * tx)

        # 선그리기
        cv2.line(raw2_edge_color, (x1, y1), (x2, y2), (0,255,0), 1)


    
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + w*(-b))
        y1 = int(y0 + h*(a))
        x2 = int(x0 - w*(-b))
        y2 = int(y0 - h*(a))

        cv2.line(raw2_edge_color, (x1,y1),(x2,y2),(0,0,255),1)       


    #Show image
    cv2.imshow("Line",raw2_edge_color)
    #cv2.imshow("Edge",raw2_edge)

    # cv2.imshow("Raw 1",gray1)
    # cv2.imshow("Raw 2",gray2)
    
    while True:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break