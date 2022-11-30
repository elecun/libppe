
'''
Wafer Pose Estimation Test
'''

import cv2
from cv2 import COLOR_GRAY2BGR
import numpy as np
from cv2 import COLOR_RGB2GRAY


if __name__ == "__main__":

    # use camera device
    device = cv2.VideoCapture(0)
    device.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    device.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    device.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    device.set(cv2.CAP_PROP_AUTO_WB, 0)
    success, raw = device.read()
    if success is False:
        print("Cannot read data from camera device")
        exit()
    else:
        h, w, _ = raw.shape
        mtx = np.matrix([[2517.792, 0., 814.045],[0., 2514.767, 567.330],[0., 0., 1.]])
        dist = np.matrix([[-0.361044, 0.154482, 0.000808, 0.000033, 0.]])
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))

    while True:

        success, raw = device.read()
        raw_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) # convert to grayscale
        #raw_gray = cv2.bilateralFilter(raw_gray, -1, 100, 55) # apply bilateral filter
        #_, raw_gray = cv2.threshold(raw_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)    # binarization

        raw_gray = cv2.undistort(raw_gray, mtx, dist, None, newcameramtx)
        
        cv2.imshow("Capture",raw_gray)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            device.release()
            break
        else:
            cv2.imwrite("{}.png".format(key), raw_gray)
