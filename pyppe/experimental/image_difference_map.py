
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
    raw = cv2.imread("./raw.jpg")
    dst = cv2.imread("./dst.jpg")
    raw_gray = cv2.cvtColor(raw, COLOR_RGB2GRAY)
    dst_gray = cv2.cvtColor(dst, COLOR_RGB2GRAY)

    difference = cv2.subtract(raw, dst)

    cv2.imshow("Diff",difference)
    cv2.imwrite("diff.jpg", difference)
    
    
    while True:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break