# -*- coding: utf-8 -*-

import sys
import json
import cv2
import numpy as np
import os.path
#import torch
#import torchvision
import math
import argparse
import csv
        

'''
Object 3D coordinates convert to Image Pixel 2D coordinates
* input : np.array([[x,y,z]], dtype=np.double)
'''
def obj_coord2pixel_coord(obj_coord, rvec, tvec, camera_matrix, dist_coeff, verbose=False):
    in_coord = np.array(obj_coord, dtype=float)
    if in_coord.size!=3:
        raise ValueError("Object Coordinates must be 3 dimensional numpy array")
    image_pts, jacobian = cv2.projectPoints(in_coord, rvec, tvec, cameraMatrix=camera_matrix, distCoeffs=dist_coeff) #3D to 2D
    if verbose:
        print("Coordinates 3D to 2D : ", in_coord.squeeze(), np.array(image_pts).squeeze())
    return image_pts

'''
Image pixel 2d coordinates convert to object 3d coordinates
* input : np.array()
'''
def pixel_coord2obj_coord(pixel_coord, rvec, tvec, camera_matrix, dist_coeff, verbose=False):
    R, jacobian = cv2.Rodrigues(rvec) # rotation vector to matrix
    point_2D_homogeneous = np.array([pixel_coord[0], pixel_coord[1], 0]).reshape(3, 1)

    # Compute the 3D point in camera coordinates
    point_3D_camera_homogeneous = np.dot(np.linalg.inv(camera_matrix), point_2D_homogeneous)
    point_3D_camera_homogeneous = np.vstack((point_3D_camera_homogeneous, [1]))  # Add homogeneous coordinate
    point_3D_camera = np.dot(np.hstack((R, tvec.reshape(3, 1))), point_3D_camera_homogeneous)
    point_3D_camera = point_3D_camera[:3, 0]  # Convert homogeneous coordinate to 3D point
    print("point_3D_camera", point_3D_camera)
    return point_3D_camera_homogeneous
        

def undistort_unproject_pts(pts_uv, camera_matrix, dist_coefs):
    """
    This function converts a set of 2D image coordinates to vectors in pinhole camera space.
    Hereby the intrinsics of the camera are taken into account.
    UV is converted to normalized image space (think frustum with image plane at z=1) then undistored
    adding a z_coordinate of 1 yield vectors pointing from 0,0,0 to the undistored image pixel.
    @return: ndarray with shape=(n, 3)

    """
    pts_uv = np.array(pts_uv)
    num_pts = pts_uv.size / 2

    pts_uv.shape = (int(num_pts), 1, 2)
    pts_uv = cv2.undistortPoints(pts_uv, camera_matrix, dist_coefs)
    pts_3d = cv2.convertPointsToHomogeneous(np.float32(pts_uv))
    pts_3d.shape = (int(num_pts),3)
    return pts_3d

'''
 Undefined Parameter key Exception
'''
class UndefinedParamError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
    
'''
Compute image quality using Sobel
'''
def compute_image_quality(json_camera_param, json_job_desc):
    result_dic = {} # estimated results (dictionary type for converting json)
    '''
    load job & parameters
    '''
    # load job & parameters from string arguments
    try:
        process_job = json.loads(json_job_desc)
        process_param = json.loads(json_camera_param)
    except json.decoder.JSONDecodeError:
        print("Job & Parameters decoding error is occured")
        
    '''
     getting developer options
    '''
    if "verbose" in process_job:
        _verbose = int(process_job["verbose"])
    else:
        _verbose = 0
    if "use_camera" in process_job:
        _use_camera = int(process_job["use_camera"])
    else:
        _use_camera = 0
    if "save_result" in process_job:
        _save_result = int(process_job["save_result"])
    else:
        _save_result = 0
        
    '''
     getting system & library check
    '''
    # get python version for different API functions prototype
    _python_version = list(map(int, cv2.__version__.split(".")))
    if _verbose:
        print("Installed Python version :", cv2.__version__)
    
    '''
    main code below
    ''' 
    try:
        
        # read image resolution
        if "resolution" in process_param:
            _w, _h = process_param["resolution"]
        else:
            raise UndefinedParamError("Image resultion configurations are not defined")
        
        # read options
        if "target_marker" in process_job:
            _target_marker = process_job["target_marker"]
        else:
            raise UndefinedParamError("Target Marker is not defined")
        
        # read roi bound option
        if "roi_bound" in process_job:
            _roi_bound = process_job["roi_bound"]
        else:
            raise UndefinedParamError("ROI Bound is not defined")
        
        # read camera parameters
        if "fx" not in process_param or "fy" not in process_param or "cx" not in process_param or "cy" not in process_param or "coeff_k1" not in process_param or "coeff_k2" not in process_param or "coeff_p1" not in process_param or "coeff_p2" not in process_param:
            raise UndefinedParamError("Some camera parameter(s) is missing")
        else:
            _fx = float(process_param["fx"])
            _fy = float(process_param["fy"])
            _cx = float(process_param["cx"])
            _cy = float(process_param["cy"])
            _k1 = float(process_param["coeff_k1"])
            _k2 = float(process_param["coeff_k2"])
            _p1 = float(process_param["coeff_p1"])
            _p2 = float(process_param["coeff_p2"])
            
        if "wafer" not in process_param:
            raise UndefinedParamError("wafer is not defined")
        else:
            if "diameter" not in process_param["wafer"]:
                raise UndefinedParamError("wafer diameter is not defined")
        _wafer_diameter = float(process_param["wafer"]["diameter"])
        
        # read reference wafer position as dictionary
        if "marker" not in process_param:
            raise UndefinedParamError("marker is not defined")
        else:
            if "coord" not in process_param["marker"]:
                raise UndefinedParamError("Marker coordinates are not defined")
        _wafer_marker_pos = json.loads(json.dumps(process_param["marker"]["coord"]))
        _wafer_marker_pos = {int(k):[int(i)+_wafer_diameter/2 for i in v] for k,v in _wafer_marker_pos.items()}
        
        # set camera parameter
        intrinsic_mtx = np.matrix([[_fx, 0.000000, _cx], [0.000000, _fy, _cy], [0.000000, 0.000000, 1.000000]])
        distorsion_mtx = np.matrix([_k1, _k2, _p1, _p2, 0.])
        newcamera_mtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic_mtx, distorsion_mtx,(_w, _h), 1)
        newcamera_mtx = np.matrix(newcamera_mtx, dtype=float)
        
        # preparation for marker detection
        if _python_version[0]==4: # < 4.7.x
            if _python_version[1]<7:
                markerdict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
                markerparams = cv2.aruco.DetectorParameters_create()
                markerparams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            else: # >= 4.7.x
                markerdict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
                markerparams = cv2.aruco.DetectorParameters()
                markerparams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                markerdetector = cv2.aruco.ArucoDetector(markerdict, markerparams)
        else:
            raise ValueError("Your python version is not supported")
        
        # image files and path prefix check
        if "files" not in process_job or "path" not in process_job:
            raise UndefinedParamError("files and path are not defined")
        _video_files = np.array(process_job["files"])
        _path_prefix = process_job["path"]
        video_files_path = np.array([_path_prefix+f for f in _video_files])
        
        # file existance check
        for src_video in video_files_path:
            if not os.path.isfile(src_video):
                raise FileNotFoundError("%s file does not exist"%src_video)
            
        
        # processing
        for filename in _video_files:
            src_video = _path_prefix+filename # video full path

            if _verbose:
                print("%s is now processing..."%src_video)
                
            _video = cv2.VideoCapture(src_video)
            _width  = int(_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            _height = int(_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _fps = _video.get(cv2.CAP_PROP_FPS)
            _frames = int(_video.get(cv2.CAP_PROP_FRAME_COUNT))
            if _verbose:
                print("> Video source info. : ({},{}@{}), {} frames".format(_width, _height, _fps, _frames))
                
            # create directory to store temporary results
            temp_result_dir = _path_prefix+filename[:-4]+"/"
            try:
                if not os.path.exists(temp_result_dir):
                    os.makedirs(temp_result_dir)
            except OSError:
                print ('Error: Creating directory. ' +  temp_result_dir)
            
            if _video.isOpened():
                measured_q = []
                measured_p = []
                measured_w = []
                measured_h = []
                frame_count = 0
                while True:
                    ret, frame = _video.read()
                    if ret == True:
                        raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        undist_raw_gray = cv2.undistort(raw_gray, intrinsic_mtx, distorsion_mtx, None, newcamera_mtx) # undistortion by camera parameters
                        undist_raw_gray = cv2.bitwise_not(undist_raw_gray) # color inversion
                        
                        # find markers printed on reference wafer
                        if _python_version[0]==4 and _python_version[1]<7:
                            corners, ids, rejected = cv2.aruco.detectMarkers(undist_raw_gray, markerdict, parameters=markerparams)
                        else:
                            corners, ids, rejected = markerdetector.detectMarkers(undist_raw_gray)
                        
                        if type(ids) == np.ndarray and ids.size>1:
                            for idx, marker_id in enumerate(ids.squeeze()):
                                corner = corners[idx].squeeze()
                                if marker_id == _target_marker:       
                                    measured_p.append(np.mean(corner, axis=0, dtype=float)[0])
                                    # extract ROI
                                    mean = np.mean(corner, axis=0, dtype=float)
                                    width = corner[1][0]-corner[0][0]
                                    height = corner[2][1]-corner[1][1]
                                    s1 = np.round(corner[0][1]-_roi_bound).astype(int)
                                    s2 = np.round(corner[0][0]-_roi_bound).astype(int)
                                    e1 = np.round(corner[2][1]+_roi_bound).astype(int)
                                    e2 = np.round(corner[2][0]+_roi_bound).astype(int)
                                    measured_w.append(corner[2][0]-corner[0][0])
                                    measured_h.append(corner[2][1]-corner[0][1])
                                    roi = undist_raw_gray[s1:e1, s2:e2].copy()
                                    
                                    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                                    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                                    scaled_sobel_x = cv2.convertScaleAbs(sobel_x)
                                    scaled_sobel_y = cv2.convertScaleAbs(sobel_y)
                                    
                                    out = cv2.addWeighted(scaled_sobel_x, 1, scaled_sobel_y, 1, 0)
                                    _mean, _std = cv2.meanStdDev(out)
                                    mean = _mean.squeeze()
                                    std = _std.squeeze()
                                    measured_q.append(std*std)
        
                                    if _save_result:
                                        cv2.imwrite(temp_result_dir+"roi_"+str(frame_count)+"_"+filename+".png", roi)     
                            frame_count += 1                
                    else:
                        # capture done
                        break
            
                # after while
                measured_q_mean, measured_q_std = cv2.meanStdDev(np.array(measured_q, dtype=float))
                measured_p_mean, measured_p_std = cv2.meanStdDev(np.array(measured_p, dtype=float))
                measured_w_mean, measured_w_std = cv2.meanStdDev(np.array(measured_w, dtype=float))
                measured_h_mean, measured_h_std = cv2.meanStdDev(np.array(measured_h, dtype=float))
                if _verbose:
                    print("Quality Mean : ", measured_q_mean)
                    print("Quality Std. Dev : ", measured_q_std)
                    print("Position Mean : ", measured_p_mean)
                    print("Position Std. Dev. : ", measured_p_std)
                    print("ROI Width Mean : ", measured_w_mean)
                    print("ROI Width Std. Dev. : ", measured_w_std)
                    print("ROI Height Mean : ", measured_h_mean)
                    print("ROI Height Std. Dev. : ", measured_h_std)
                
            
    except json.decoder.JSONDecodeError :
        print("Error : Decoding Job Description has failed")
    except (ValueError, UndefinedParamError) as e:
        print("Error : ",e)

    json_result = json.dumps(result_dic)
    return json_result

    
'''
 estimation processing with already saved image frame (from job description)
'''
def estimate(json_camera_param, json_job_desc):
    result_dic = {} # estimated results (dictionary type for converting json)

    '''
    load job & parameters
    '''
    # load job & parameters from string arguments
    try:
        process_job = json.loads(json_job_desc)
        process_param = json.loads(json_camera_param)
    except json.decoder.JSONDecodeError:
        print("Job & Parameters decoding error is occured")
             
    '''
     getting developer options
    '''
    if "verbose" in process_job:
        _verbose = int(process_job["verbose"])
    else:
        _verbose = 0
    if "use_camera" in process_job:
        _use_camera = int(process_job["use_camera"])
    else:
        _use_camera = 0
    if "save_result" in process_job:
        _save_result = int(process_job["save_result"])
    else:
        _save_result = 0
    if "yaw_gt_initial" in process_job:
        _yaw_gt_initial = process_job["yaw_gt_initial"]
    else:
        _yaw_gt_initial = 0.0
    if "yaw_gt" in process_job:
        _yaw_gt = np.array(process_job["yaw_gt"], dtype=float)
        _yaw_gt += _yaw_gt_initial
    else:
        _yaw_gt = np.array([], dtype=float)
        
    '''
     getting system & library check
    '''
    # get python version for different API functions prototype
    _python_version = list(map(int, cv2.__version__.split(".")))
    if _verbose:
        print("Installed Python version :", cv2.__version__)
        
        
    '''
    main code below
    ''' 
    try:
        
        # read image resolution
        if "resolution" in process_param:
            _w, _h = process_param["resolution"]
        else:
            raise UndefinedParamError("Image resultion configurations are not defined")

        
        # read camera parameters
        if "fx" not in process_param or "fy" not in process_param or "cx" not in process_param or "cy" not in process_param or "coeff_k1" not in process_param or "coeff_k2" not in process_param or "coeff_p1" not in process_param or "coeff_p2" not in process_param:
            raise UndefinedParamError("Some camera parameter(s) is missing")
        else:
            _fx = float(process_param["fx"])
            _fy = float(process_param["fy"])
            _cx = float(process_param["cx"])
            _cy = float(process_param["cy"])
            _k1 = float(process_param["coeff_k1"])
            _k2 = float(process_param["coeff_k2"])
            _p1 = float(process_param["coeff_p1"])
            _p2 = float(process_param["coeff_p2"])
            
        if "wafer" not in process_param:
            raise UndefinedParamError("wafer is not defined")
        else:
            if "diameter" not in process_param["wafer"]:
                raise UndefinedParamError("wafer diameter is not defined")
        _wafer_diameter = float(process_param["wafer"]["diameter"])
        
        # read reference wafer position as dictionary
        if "marker" not in process_param:
            raise UndefinedParamError("marker is not defined")
        else:
            if "coord" not in process_param["marker"]:
                raise UndefinedParamError("Marker coordinates are not defined")
        _wafer_marker_pos = json.loads(json.dumps(process_param["marker"]["coord"]))
        _wafer_marker_pos = {int(k):[int(i)+_wafer_diameter/2 for i in v] for k,v in _wafer_marker_pos.items()}
        
        # set camera parameter
        intrinsic_mtx = np.matrix([[_fx, 0.000000, _cx], [0.000000, _fy, _cy], [0.000000, 0.000000, 1.000000]])
        distorsion_mtx = np.matrix([_k1, _k2, _p1, _p2, 0.])
        newcamera_mtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic_mtx, distorsion_mtx,(_w, _h), 1)
        newcamera_mtx = np.matrix(newcamera_mtx, dtype=float)
        
        # preparation for marker detection
        if _python_version[0]==4: # < 4.7.x
            if _python_version[1]<7:
                markerdict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
                markerparams = cv2.aruco.DetectorParameters_create()
                markerparams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            else: # >= 4.7.x
                markerdict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
                markerparams = cv2.aruco.DetectorParameters()
                markerparams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                markerdetector = cv2.aruco.ArucoDetector(markerdict, markerparams)
        else:
            raise ValueError("Your python version is not supported")
        
        # image files and path prefix check
        if "files" not in process_job or "path" not in process_job:
            raise UndefinedParamError("files and path are not defined")
        _image_files = np.array(process_job["files"])
        _path_prefix = process_job["path"]
        image_files_path = np.array([_path_prefix+f for f in _image_files])
        
        # measured distance check
        if "laser_wafer_distance" not in process_job:
            raise UndefinedParamError("laser_wafer_distance is not defined")
        _laser_distance = np.array(process_job["laser_wafer_distance"])
        
        # file existance check
        for src_image in image_files_path:
            if not os.path.isfile(src_image):
                raise FileNotFoundError("%s file does not exist"%src_image)
            
        # size of files and laser distance sould be same
        if _laser_distance.size != _image_files.size:
            raise ValueError("Laser Distance data and Image file size should be same.")
            
        
        # processing
        estimated_yaw_deg = []
        for filename in _image_files:
            src_image = _path_prefix+filename # image full path

            if _verbose:
                print("%s is now processing..."%src_image)
            
            raw_image = cv2.imread(src_image, cv2.IMREAD_UNCHANGED)
            if raw_image.shape[2]==3: # if color image
                raw_color = raw_image.copy()
                raw_gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
            else:
                raw_gray = raw_image.copy()
                raw_color = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
                
            undist_raw_gray = cv2.undistort(raw_gray, intrinsic_mtx, distorsion_mtx, None, newcamera_mtx) # undistortion by camera parameters
            undist_raw_gray = cv2.bitwise_not(undist_raw_gray) # color inversion
            undist_raw_color = cv2.cvtColor(undist_raw_gray, cv2.COLOR_GRAY2BGR)
            
            # find markers printed on reference wafer
            if _python_version[0]==4 and _python_version[1]<7:
                corners, ids, rejected = cv2.aruco.detectMarkers(undist_raw_gray, markerdict, parameters=markerparams)
            else:
                corners, ids, rejected = markerdetector.detectMarkers(undist_raw_gray)
            
            # detected marker preprocessing
            marker_centroids_on_image = []
            marker_centroids_on_wafer = []
            
            for idx, marker_id in enumerate(ids.squeeze()):
                corner = corners[idx].squeeze()                
                marker_centroids_on_image.append(np.mean(corner, axis=0, dtype=float))
                marker_centroids_on_wafer.append(_wafer_marker_pos[marker_id])
            marker_centroids_on_image = np.array(marker_centroids_on_image)
            marker_centroids_on_wafer = np.array(marker_centroids_on_wafer)
            marker_centroids_on_wafer = np.append(marker_centroids_on_wafer, np.zeros(shape=(np.size(marker_centroids_on_wafer, axis=0), 1), dtype=np.double),axis=1) # column add
            
            if marker_centroids_on_image.shape[0] != marker_centroids_on_wafer.shape[0]:
                raise ValueError("Marker pointset dimension is not same")
            
            # save detected image (draw point on marker center point)
            if _save_result:
                for idx, pts in enumerate(marker_centroids_on_image):
                    p = tuple(pts.round().astype(int))
                    
                    str_image_pos = "on image : [%d] x=%2.2f,y=%2.2f"%(ids[idx], pts[0], pts[1])
                    str_world_pos = "on wafer : x=%2.2f,y=%2.2f"%(marker_centroids_on_wafer[idx][0], marker_centroids_on_wafer[idx][1])
                    if _verbose:
                        print("marker :",str_image_pos, str_world_pos)
                    cv2.putText(undist_raw_color, str_image_pos,(p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(undist_raw_color, str_world_pos,(p[0]+10, p[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.line(undist_raw_color, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,255,0), 1, cv2.LINE_AA)
                    cv2.line(undist_raw_color, (p[0],p[1]-10), (p[0],p[1]+10), (0,255,0), 1, cv2.LINE_AA)
                cv2.imwrite(_path_prefix+"markers_"+filename, undist_raw_color)
                
            if ids.size>3:
                # compute 2D-3D corredpondence with Perspective N Point'
                # note] use undistorted image points to solve, then appply the distortion coefficient and camera matrix as pin hole camera model
                _, rVec, tVec = cv2.solvePnP(marker_centroids_on_wafer, marker_centroids_on_image, cameraMatrix=newcamera_mtx, distCoeffs=distorsion_mtx, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP)
                _, prVec, ptVec = cv2.solvePnP(marker_centroids_on_wafer, marker_centroids_on_image, cameraMatrix=intrinsic_mtx, distCoeffs=None, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP)
                R, jacobian = cv2.Rodrigues(rVec) # rotation vector to matrix
                
                # save yaw angles
                estimated_yaw_deg.append(np.rad2deg(math.atan2(-R[1][0], R[0][0])))

                if _verbose:
                    print("Rotation Angle", -1*np.rad2deg(math.atan2(-R[1][0], R[0][0])))
                
                # testing for 3D to 2D
                #np.array([[130.0, 210.0, 0.0]], dtype=float)
                image_pts = obj_coord2pixel_coord([130.0, 210.0, 0.0], rVec, tVec, newcamera_mtx, distorsion_mtx, verbose=_verbose)
                if _save_result:
                    image_pts = image_pts.squeeze()
                    str_pos = "x=%2.2f,y=%2.2f"%(image_pts[0], image_pts[1])
                    image_ptsi = (image_pts.round().astype(int))
                    cv2.putText(undist_raw_color, str_pos,(image_ptsi[0]+10, image_ptsi[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.line(undist_raw_color, (image_ptsi[0]-10,image_ptsi[1]), (image_ptsi[0]+10,image_ptsi[1]), (0,255,255), 1, cv2.LINE_AA)
                    cv2.line(undist_raw_color, (image_ptsi[0],image_ptsi[1]-10), (image_ptsi[0],image_ptsi[1]+10), (0,255,255), 1, cv2.LINE_AA)
                    cv2.imwrite(_path_prefix+"temp_"+filename, undist_raw_color)
                

                

                
                #point_2D = (437.06, 511.31)
                # point_2D = (418.06, 513.01)

                # # Convert the point to homogeneous coordinates
                # point_2D_homogeneous = np.array([419.0, 514.0, 0]).reshape(3, 1)

                # # Define the intrinsic camera matrix and extrinsic parameters (rotation and translation)
                # R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Rotation matrix
                # #T = np.array([0, 0, 0])  # Translation vector

                # # Compute the 3D point in camera coordinates
                # point_3D_camera_homogeneous = np.dot(np.linalg.inv(newcamera_mtx), point_2D_homogeneous)
                # point_3D_camera_homogeneous = np.vstack((point_3D_camera_homogeneous, [1]))  # Add homogeneous coordinate
                # point_3D_camera = np.dot(np.hstack((R, tVec.reshape(3, 1))), point_3D_camera_homogeneous)
                # point_3D_camera = point_3D_camera[:3, 0]  # Convert homogeneous coordinate to 3D point
                # print("point_3D_camera", point_3D_camera)

                
                # testing for coordinate conversion (2D to 3D)
                #image_pts = np.array([[275.66, 411.34]], dtype=float) # world = (130, 210)
                #world_pts = cv2.undistortPoints(np.expand_dims(image_pts, axis=1), cameraMatrix=newcamera_mtx, distCoeffs=distorsion_mtx, R=None, P=None)
                #world_pts = world_pts.squeeze()
                #print("world point",world_pts)
                
                #print(undistort_unproject_pts(image_pts, newcamera_mtx, distorsion_mtx))
                
                # ptsOut = np.array(corners, dtype='float32')
                # ptsTemp = np.array([], dtype='float32')
                # rtemp = ttemp = np.array([0,0,0], dtype='float32')
                # image_pts = cv2.undistortPoints(image_pts, newcamera_mtx, None)
                # ptsTemp = cv2.convertPointsToHomogeneous(image_pts)
                # output = cv2.projectPoints( ptsTemp, rtemp, ttemp, newcamera_mtx, distorsion_mtx, ptsOut )\
                
                
                
                p_dic = {}
                p_dic["wafer_x"] = 0.0
                p_dic["wafer_x"] = 0.0
                p_dic["wafer_y"] = 0.0
                p_dic["wafer_z"] = 0.0
                p_dic["wafer_r"] = 0.0
                p_dic["wafer_p"] = 0.0
                p_dic["wafer_w"] = 0.0
                p_dic["effector_x"] = 0.0
                p_dic["effector_y"] = 0.0
                p_dic["effector_z"] = 0.0
                p_dic["effector_r"] = 0.0
                p_dic["effector_p"] = 0.0
                p_dic["effector_w"] = 0.0
                p_dic["distance"] = 0.0
                result_dic[filename] = p_dic
            else:
                print("Not enough markers are detected")
        
        # for report (yaw angle)
        if _yaw_gt.size>0:
            estimated_yaw_deg = np.abs(np.array(estimated_yaw_deg))
            yaw_rmse = np.sqrt(np.mean((estimated_yaw_deg-_yaw_gt)**2))
            print("* RMSE Rotation(Yaw deg)", yaw_rmse)
            if _save_result:
                with open('yaw_rotation.csv', 'w') as f:
                    rot_out_file = csv.writer(f)
                    rot_out_file.writerow(["index", "Estimated", "Ground Truth"])
                    for idx, deg in enumerate(_yaw_gt):
                        rot_out_file.writerow([idx, estimated_yaw_deg[idx], _yaw_gt[idx]])
        
                
        # dumps into json result        
        json_result = json.dumps(result_dic)
        
    except json.decoder.JSONDecodeError :
        print("Error : Decoding Job Description has failed")
    except (ValueError, UndefinedParamError) as e:
        print("Error : ",e)

    json_result = json.dumps(result_dic)
    return json_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', required=True, help="Configuration file")
    parser.add_argument('--job', nargs='?', required=True, help="Job file")
    parser.add_argument('--quality', required=False, help="Compute Image Quality", action='store_true')
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = json.load(file)
    with open(args.job, 'r') as file:
        job = json.load(file)
    
    if args.quality is False:
        # call estimate function
        estimate(json.dumps(config), json.dumps(job))
    else:
        compute_image_quality(json.dumps(config), json.dumps(job))