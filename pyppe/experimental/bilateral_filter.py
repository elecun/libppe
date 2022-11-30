
'''
Bilateral Filtering for smoothing
'''

import cv2
from cv2 import COLOR_GRAY2BGR
import numpy as np
from cv2 import COLOR_RGB2GRAY
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load Images
    raw = cv2.imread("./experiments/0216_tool_1/25.jpg")
    raw_gray = cv2.cvtColor(raw, COLOR_RGB2GRAY)

    dst = cv2.bilateralFilter(raw, -1, 100, 55)

    cv2.imwrite("raw.jpg", raw)
    cv2.imwrite("dst.jpg", dst)

    cv2.imshow("Raw",raw)
    cv2.imshow("Filtered",dst)
    
    while True:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break