# -*- coding: utf-8 -*-

import sys
import json
import cv2
import numpy as np
import os.path
import torch
import torchvision
import math


'''
 job processing with camera parameters
'''
def estimate(json_camera_param, json_job_desc):
    
    try:
        # load user parameters
        job = json.loads(json_job_desc)
        param = json.loads(json_camera_param)
        
        # read pre-defined image info
        if "resolution" in param:
            _w, _h = param["resolution"]
        else:
            raise ValueError("Image resultion configurations are not defined")
        
        # set camera parameter
        intrinsic_mtx = np.matrix([[float(param['fx']), 0.000000, float(param['cx'])], [0.000000, float(param['fy']), float(param['cy'])], [0.000000, 0.000000, 1.000000]])
        distorsion_mtx = np.matrix([[float(param['coeff_k1']), float(param['coeff_k2']), float(param['coeff_p1']), float(param['coeff_p2']), 0.]])
        newcamera_mtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic_mtx, distorsion_mtx,(_w, _h),0,(_w, _h))
        
        # marker
        markerdict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        markerparams = cv2.aruco.DetectorParameters_create()
        markerparams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        # camera coord. / transformation matrix
        coord = np.matrix([[1,0,0,float(param['coord'][0])],[0,1,0,float(param['coord'][1])],[0,0,1,float(param['coord'][2])],[0,0,0,1]])
        
        # file existance check
        result_dic = {}
        if "files" in job and "path" in job:
            for job_file in job["files"]:
                job_file_path = job["path"]+job_file
                
                # estimation processing
                if os.path.isfile(job_file_path): # if file exist
                    
                    # getting image profile
                    raw_image = cv2.imread(job_file_path, cv2.IMREAD_UNCHANGED)
                    raw_image_gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
                    _raw_h, _raw_w, _raw_c = raw_image.shape
                    if _raw_h!=_h and _raw_w!=_w:
                        raise ValueError("Image shaoe is different from your configurations")
                    
                    # undistorsion
                    ud_image_gray = cv2.undistort(raw_image_gray, intrinsic_mtx, None, newcamera_mtx)
                    
                    # preprocessing (binarization)
                    ud_image_gray = cv2.bitwise_not(ud_image_gray)
                    _, ud_image_binary = cv2.threshold(ud_image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    
                    
                    
                    # find markers
                    corners, ids, rejected = cv2.aruco.detectMarkers(ud_image_binary, markerdict, parameters=markerparams)
                    print("marker detected")
                    
                    
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
                    
                    result_dic[job_file] = p_dic
    
            # output to return
            json_result = json.dumps(result_dic)
        
    except json.decoder.JSONDecodeError :
        print("Decoding Job Description has failed")
    except ValueError as e:
        print(e)
    
    return json_result