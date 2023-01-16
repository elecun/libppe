'''
Reference Wafer Marker Pose Accuracy Test 
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import COLOR_RGB2GRAY
import warnings
warnings.filterwarnings('ignore')

mtx = np.matrix([[2517.792, 0., 814.045],[0., 2514.767, 567.330],[0., 0., 1.]])
dist = np.matrix([[-0.361044, 0.154482, 0.000808, 0.000033, 0.]])

video = cv2.VideoCapture('./video/140_1600.avi')
w  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print("> high resolution video info : ({},{}@{}), {} frames".format(w, h, fps, frames))
newcameramtx, roi_high = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

markerdict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
markerparams = cv2.aruco.DetectorParameters_create()

# for video-1 (1280_960 resolution)
if video.isOpened():
    while True:
        ret, frame = video.read()
        if ret == True:
            frame_undist = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            corners, ids, rejected = cv2.aruco.detectMarkers(gray, markerdict, parameters=markerparams)

            if len(corners) > 0:
                for i in range(0, len(ids)):
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.04, mtx, dist)
                    
                    if ids[i] == 102:
                        print("{}\tX : {}\tY : {}\tZ : {}".format(ids[i], tvec.reshape(-1)[0]*100, tvec.reshape(-1)[1]*100, tvec.reshape(-1)[2]*100))
                        print(rvec)
                        break

                    (topLeft, topRight, bottomRight, bottomLeft) = corners[i].reshape((4,2))
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(frame_undist, (cX, cY), 4, (0, 0, 255), -1)

                    cv2.aruco.drawDetectedMarkers(frame_undist, corners) 
                    #cv2.aruco.drawFrameAxes(frame_undist, mtx, dist, rvec, tvec, 0.01) 
                    #cv2.putText(frame_undist, str(ids[i]),(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            

            # cv2.imshow("Detected Marker",frame_undist)
            # key = cv2.waitKey(1)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     break
            
        else:
            break
    
video.release()
