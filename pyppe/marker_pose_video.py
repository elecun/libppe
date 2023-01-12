'''
Reference Wafer Marker Pose Accuracy Test 
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import COLOR_RGB2GRAY
import warnings
warnings.filterwarnings('ignore')

video_low = cv2.VideoCapture('./video/100_1280_960.avi')
video_high = cv2.VideoCapture('./video/100_1600_1200.avi')

w_low  = int(video_low.get(cv2.CAP_PROP_FRAME_WIDTH))
h_low = int(video_low.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_low = video_low.get(cv2.CAP_PROP_FPS)
frames_low = int(video_low.get(cv2.CAP_PROP_FRAME_COUNT))
print("> low resolution video info : ({},{}@{}), {} frames".format(w_low, h_low, fps_low, frames_low))

w_high  = int(video_high.get(cv2.CAP_PROP_FRAME_WIDTH))
h_high = int(video_high.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_high = video_high.get(cv2.CAP_PROP_FPS)
frames_high = int(video_high.get(cv2.CAP_PROP_FRAME_COUNT))
print("> high resolution video info : ({},{}@{}), {} frames".format(w_high, h_high, fps_high, frames_high))


mtx = np.matrix([[2517.792, 0., 814.045],[0., 2514.767, 567.330],[0., 0., 1.]])
dist = np.matrix([[-0.361044, 0.154482, 0.000808, 0.000033, 0.]])
newcameramtx_low, roi_low = cv2.getOptimalNewCameraMatrix(mtx, dist, (w_low,h_low), 0, (w_low,h_low))
newcameramtx_high, roi_high = cv2.getOptimalNewCameraMatrix(mtx, dist, (w_low,h_high), 0, (w_low,h_high))

markerdict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
markerparams = cv2.aruco.DetectorParameters_create()

# for video-1 (1280_960 resolution)
if video_low.isOpened():
    while True:
        ret, frame = video_low.read()
        if ret == True:
            frame_undist = cv2.undistort(frame, mtx, dist, None, newcameramtx_low)
            gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = cv2.aruco.detectMarkers(gray, markerdict, parameters=markerparams)

            if len(corners) > 0:
                for i in range(0, len(ids)):
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.04, mtx, dist)
                    
                    if ids[i] == 32:
                        print("{}\tX : {}\tY : {}\tZ : {}".format(ids[i], tvec.reshape(-1)[0]*100, tvec.reshape(-1)[1]*100, tvec.reshape(-1)[2]*100))

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
            

            cv2.imshow("Detected Marker",frame_undist)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
            
        else:
            break
    
video_low.release()
