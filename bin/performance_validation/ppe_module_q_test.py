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
     set developer options
    '''
    _verbose = 0 if "verbose" not in process_job else int(process_job["verbose"])
    _use_camera = 0 if "use_camera" not in process_job else int(process_job["use_camera"])
    _save_result = 0 if "save_result" not in process_job else int(process_job["save_result"])
    
        
    '''
    set program parameters
    '''
    _target_marker = 1 if "target_marker" not in process_job else process_job["target_marker"]
    _roi_bound = 0 if "roi_bound" not in process_job else process_job["roi_bound"]
    
    '''
     set system & library
    '''
    # get python version for different API functions prototype
    _python_version = list(map(int, cv2.__version__.split(".")))
    print("* Installed Python version :", cv2.__version__) if _verbose else None
        
        
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
        if all(key in process_param for key in ("fx","fy", "cx", "cy", "coeff_k1", "coeff_k2", "coeff_p1", "coeff_p2")):
            _fx = float(process_param["fx"])
            _fy = float(process_param["fy"])
            _cx = float(process_param["cx"])
            _cy = float(process_param["cy"])
            _k1 = float(process_param["coeff_k1"])
            _k2 = float(process_param["coeff_k2"])
            _p1 = float(process_param["coeff_p1"])
            _p2 = float(process_param["coeff_p2"])
        else:
            raise UndefinedParamError("Some camera parameter(s) is missing")
        
        # read reference wafer parameters    
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
        _source_files = np.array(process_job["files"])
        _path_prefix = process_job["path"]
        _source_full_path = np.array([_path_prefix+f for f in _source_files])
        
        # measured distance check
        if "laser_wafer_distance" not in process_job:
            raise UndefinedParamError("laser_wafer_distance is not defined")
        _laser_distance = np.array(process_job["laser_wafer_distance"])
        
        # file existance check
        for file in _source_full_path:
            if not os.path.isfile(file):
                raise FileNotFoundError("%s file does not exist"%file)
            
        # size of files and laser distance sould be same
        if _laser_distance.size != _source_files.size:
            raise ValueError("Laser Distance data and Image file size should be same.")
            
        # processing
        # data container for processing results
        estimated_yaw_deg = []
        real_yaw_deg = []
        
        
        for fid, filename in enumerate(_source_files):
            src_file = _path_prefix+filename # video full path
            
            print("%s is now processing..."%src_file) if _verbose else None
            
            _video = cv2.VideoCapture(src_file)
            _width  = int(_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            _height = int(_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _fps = _video.get(cv2.CAP_PROP_FPS)
            _frames = int(_video.get(cv2.CAP_PROP_FRAME_COUNT))
            print("> Video source info. : ({},{}@{}), {} frames".format(_width, _height, _fps, _frames)) if _verbose else None
            
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
                        
                        if ids is not None and ids.size>1:
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
                                    
                                    # sobel method
                                    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                                    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                                    scaled_sobel_x = cv2.convertScaleAbs(sobel_x)
                                    scaled_sobel_y = cv2.convertScaleAbs(sobel_y)
                                    
                                    out = cv2.addWeighted(scaled_sobel_x, 1, scaled_sobel_y, 1, 0)
                                    _mean, _std = cv2.meanStdDev(out)
                                    mean = _mean.squeeze()
                                    std = _std.squeeze()
                                    measured_q.append(std*std)
        
                                    cv2.imwrite(temp_result_dir+"roi_"+str(frame_count)+"_"+filename+".png", roi) if _save_result else None
                            frame_count += 1                
                    else:
                        # capture done
                        break
            
                # after while
                measured_q_mean, measured_q_std = cv2.meanStdDev(np.array(measured_q, dtype=float))
                measured_p_mean, measured_p_std = cv2.meanStdDev(np.array(measured_p, dtype=float))
                measured_w_mean, measured_w_std = cv2.meanStdDev(np.array(measured_w, dtype=float))
                measured_h_mean, measured_h_std = cv2.meanStdDev(np.array(measured_h, dtype=float))
                print("* ------<Wafer Transfer Validity Test Result>------")
                print("Quality Mean : ", measured_q_mean)
                print("Quality Std. Dev : ", measured_q_std)
                print("Position Mean : ", measured_p_mean)
                print("Position Std. Dev. : ", measured_p_std)
                print("ROI Width Mean : ", measured_w_mean)
                print("ROI Width Std. Dev. : ", measured_w_std)
                print("ROI Height Mean : ", measured_h_mean)
                print("ROI Height Std. Dev. : ", measured_h_std)
                print("* ------------------------------------------------")
                
            else:
                print("%s video file cannot be opened"%filename)
        
    except json.decoder.JSONDecodeError :
        print("Error : Decoding Job Description has failed")
    except (ValueError, UndefinedParamError) as e:
        print("Error : ",e)

    json_result = json.dumps(result_dic)
    print(result_dic) if _verbose else None
    
    return json_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', required=True, help="Configuration file")
    parser.add_argument('--job', nargs='?', required=True, help="Job file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = json.load(file)
    with open(args.job, 'r') as file:
        job = json.load(file)

    # do estimate
    estimate(json.dumps(config), json.dumps(job))