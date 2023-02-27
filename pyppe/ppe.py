'''
Reference Wafer Marker Pose Accuracy Test 
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import COLOR_RGB2GRAY
import warnings
warnings.filterwarnings('ignore')
import argparse
from abc import *
import os
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from skspatial.objects import Sphere, Cylinder

class estimator(metaclass=ABCMeta):

    source_file = ""

    # camera parameters by camera calibration
    mtx = np.matrix([[2517.792, 0., 814.045],[0., 2514.767, 567.330],[0., 0., 1.]])
    dist = np.matrix([[-0.361044, 0.154482, 0.000808, 0.000033, 0.]])

    @abstractmethod
    def estimate(self):
        pass


class wafer_estimator(estimator):

    def __init__(self, map, source) -> None:
        self.map = map
        self.source_file = source
        #self.markerdict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250) # for < opencv 4.6
        #self.markerparams = cv2.aruco.DetectorParameters_create() # for < opencv 4.6
        self.markerdict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250) # for opencv 4.7.x (API was changed)
        self.markerparams = cv2.aruco.DetectorParameters() # for opencv 4.7.x
        self.markerparams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX #marker detection refinement
        self.markerdetector = cv2.aruco.ArucoDetector(self.markerdict, self.markerparams)

    def __str__(self) -> str:
        return "wafer estimator"

    def estimate(self):
        # show wafer position in 3D 
        # fig = plt.figure(figsize=(10, 10))
        # ax = plt.axes(projection='3d')
        # fig.clf()
        # fig.show()
        _path, _ext = os.path.splitext(self.source_file)
        if _ext == '.avi': # for video
            _video = cv2.VideoCapture(self.source_file)
            _width  = int(_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            _height = int(_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _fps = _video.get(cv2.CAP_PROP_FPS)
            _frames = int(_video.get(cv2.CAP_PROP_FRAME_COUNT))
            print("> Video source info. : ({},{}@{}), {} frames".format(_width, _height, _fps, _frames))
            _newmtx, _roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (_width,_height), 0, (_width,_height))

            if _video.isOpened():
                while True:
                    ret, frame = _video.read()
                    if ret == True:
                        _undist_frame = cv2.undistort(frame, self.mtx, self.dist, None, _newmtx)
                        _gray_frame = cv2.cvtColor(_undist_frame, cv2.COLOR_BGR2GRAY)
                        _gray_inv = cv2.bitwise_not(_gray_frame)
                        ret, _binary = cv2.threshold(_gray_inv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        markerCorners, markerIds, rejectedCandidates = self.markerdetector.detectMarkers(_binary)
                        print(markerCorners[0].tolist())

                        if len(markerCorners) > 2:
                            for i in range(0, len(markerIds)):
                                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners[i], 0.04, self.mtx, self.dist)
                                
                                if markerIds[i] == 9:
                                    print("{}\tX : {}\tY : {}\tZ : {}".format(markerIds[i], tvec.reshape(-1)[0]*100, tvec.reshape(-1)[1]*100, tvec.reshape(-1)[2]*100))
                                    print(rvec)
                                    break

                                (topLeft, topRight, bottomRight, bottomLeft) = markerCorners[i].reshape((4,2))
                                topRight = (int(topRight[0]), int(topRight[1]))
                                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                                topLeft = (int(topLeft[0]), int(topLeft[1]))

                                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                                cv2.circle(_undist_frame, (cX, cY), 4, (0, 0, 255), -1)

                                cv2.aruco.drawDetectedMarkers(_undist_frame, markerCorners) 
                                #cv2.aruco.drawFrameAxes(frame_undist, mtx, dist, rvec, tvec, 0.01) 
                                #cv2.putText(frame_undist, str(ids[i]),(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        
                        # draw corner points
                        cv2.imshow("result", _undist_frame)
                    key = cv2.waitKey(100)
                    if key == 27:
                        cv2.destroyAllWindows()
                        break
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', nargs='?', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--source', nargs='?', required=True, help="input image or video file to estimate the pose")
    args = parser.parse_args()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0, 200) # z limit : 200mm
    cylinder = Cylinder([0, 0, 0], [0, 0, 0.775], 150) # center, normal vector(z-775um), radius(150mm)
    cylinder.plot_3d(ax, alpha=0.8)
    cylinder.point.plot_3d(ax, s=50)
    fig.show()


    if args.target == "fork" or args.target == "effector":
        pass
    elif args.target == "wafer":
        estimator = wafer_estimator(args.map, args.source)
        result = estimator.estimate()
    elif args.target == "camera":
        pass