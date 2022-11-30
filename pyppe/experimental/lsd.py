
'''
Linea Feature
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
    
    raw1_gray = cv2.cvtColor(raw1, COLOR_RGB2GRAY)
    raw2_gray = cv2.cvtColor(raw2, COLOR_RGB2GRAY)

    # edge aware filtering
    raw1_gray = cv2.bilateralFilter(raw1_gray, -1, 10, 55)
    raw2_gray = cv2.bilateralFilter(raw2_gray, -1, 10, 55)

    _, raw1_binary = cv2.threshold(raw1_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, raw2_binary = cv2.threshold(raw2_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # gray1 = cv2.Canny(gray1, 10,5)
    # gray2 = cv2.Canny(gray2, 10,5)

    cv2.getStructuringElement()
int size = 3;
Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*size + 1, 2*size+1 ), Point( size, size ) );
morphologyEx( mask, mask, MORPH_CLOSE, element );

    # 선명하게
    kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])

    # 커널 적용 
    #gray1 = cv2.filter2D(raw1_gray, -1, kernel)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(raw1_gray)[0] #Position 0 of the returned tuple are the detected lines
    print(lines.shape)

    #Draw detected lines in the image
    drawn_img = lsd.drawSegments(raw1_gray,lines)

    #Show image
    cv2.imshow("LSD",drawn_img )

    # cv2.imshow("Raw 1",gray1)
    # cv2.imshow("Raw 2",gray2)
    
    while True:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break