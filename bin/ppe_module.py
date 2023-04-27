# -*- coding: utf-8 -*-

import sys
import json
import cv2
import numpy as np
import os.path
import torch
import torchvision
import math
import argparse

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
 estimation processing with already saved image frame (from job description)
'''
def estimate(json_camera_param, json_job_desc):
    
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
            raise ValueError("Image resultion configurations are not defined")
        
        # read camera parameters
        if "fx" not in process_param or "fy" not in process_param or "cx" not in process_param or "cy" not in process_param or "coeff_k1" not in process_param or "coeff_k2" not in process_param or "coeff_p1" not in process_param or "coeff_p2" not in process_param:
            raise ValueError("Some camera parameter(s) is missing")
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
            raise ValueError("wafer is not defined")
        else:
            if "diameter" not in process_param["wafer"]:
                raise ValueError("wafer diameter is not defined")
        _wafer_diameter = float(process_param["wafer"]["diameter"])
        
        # read reference wafer position as dictionary
        if "marker" not in process_param:
            raise ValueError("marker is not defined")
        else:
            if "coord" not in process_param["marker"]:
                raise ValueError("Marker coordinates are not defined")
        _wafer_marker_pos = json.loads(json.dumps(process_param["marker"]["coord"]))
        _wafer_marker_pos = {int(k):[int(i)+_wafer_diameter for i in v] for k,v in _wafer_marker_pos.items()}
        print(_wafer_marker_pos)
        
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
            raise ValueError("files and path are not defined")
        _image_files = np.array(process_job["files"])
        _path_prefix = process_job["path"]
        image_files_path = np.array([_path_prefix+f for f in _image_files])
        
        # measured distance check
        if "laser_wafer_distance" not in process_job:
            raise ValueError("laser_wafer_distance is not defined")
        _laser_distance = np.array(process_job["laser_wafer_distance"])
        
        # file existance check
        for src_image in image_files_path:
            if not os.path.isfile(src_image):
                raise ValueError("%s file does not exist"%src_image)
            
        
        # processing
        result_dic = {} # estimated results (dictionary type for converting json)
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
            
            # save detected image (draw point on marker center point)
            if _save_result:
                for pts in marker_centroids_on_image:
                    p = tuple(pts.round().astype(int))
                    cv2.line(undist_raw_color, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,255,0), 1, cv2.LINE_AA)
                    cv2.line(undist_raw_color, (p[0],p[1]-10), (p[0],p[1]+10), (0,255,0), 1, cv2.LINE_AA)
                cv2.imwrite(_path_prefix+"markers_"+filename, undist_raw_color)
                
            if ids.size>3:
                pass
                # compute 2D-3D corredpondence with Perspective N Point
                #_, rVec, tVec = cv2.solvePnP(wafer_pts_vec, image_pts_vec, newcamera_mtx, distorsion_mtx, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP)
                #R, jacobian = cv2.Rodrigues(rVec) # rotation vector to matrix
                
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
                
        # dumps into json result        
        json_result = json.dumps(result_dic)
        
    except json.decoder.JSONDecodeError :
        print("Decoding Job Description has failed")
        json_result = json.dumps(result_dic)
        return json_result
    except ValueError as e:
        print(e)
        json_result = json.dumps(result_dic)
        return json_result
    
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
    
    # call estimate function
    estimate(json.dumps(config), json.dumps(job))