'''
Fork-tip key point detection with Mask R-CNN 
'''
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

patch = cv2.imread("./data/fork_patch.png")
patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
print(patch_gray.shape)

#patch_gray = cv2.GaussianBlur(patch_gray, (5, 5), 0)

#patch_gray = cv2.medianBlur(patch_gray, 5)
# patch_gray = cv2.bilateralFilter(patch_gray, -1, 1, 5)

patch_canny = cv2.Canny(patch_gray, 10, 150)
#patch_laplacian = cv2.Laplacian(patch_gray, cv2.CV_8UC1, ksize=5) #laplacian edge
#patch_sobel = cv2.Sobel(patch_gray,cv2.CV_8UC1, 2, 2)
while True:
    cv2.imshow("test", patch_canny)
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyAllWindows()
        break