# -*- coding: utf-8 -*-

import sys
import json
import cv2
import numpy as np


'''
 job processing with camera parameters
'''
def estimate(json_camera_param, json_job_desc):
    
    try:
        # load parameters
        desc = json.loads(json_job_desc)
        param = json.loads(json_camera_param)
        
        print(param.x)
        
        # parameterset
        # intrinsic_mtx = (
        #     [[param.fx]]
        # )
        
        #mtx = np.matrix([[2104.980135, 0.000000, 685.611350],[0.000000, 2105.382253, 481.061709],[0., 0., 1.]])
        
        # output to return
        result_dic = {}
        result_dic["wafer_x"] = 0.0
        result_dic["wafer_y"] = 0.0
        result_dic["wafer_z"] = 0.0
        result_dic["wafer_r"] = 0.0
        result_dic["wafer_p"] = 0.0
        result_dic["wafer_w"] = 0.0
        result_dic["effector_x"] = 0.0
        result_dic["effector_y"] = 0.0
        result_dic["effector_z"] = 0.0
        result_dic["effector_r"] = 0.0
        result_dic["effector_p"] = 0.0
        result_dic["effector_w"] = 0.0
        result_dic["distance"] = 0.0
        json_result = json.dumps(result_dic)
        
    except json.decoder.JSONDecodeError :
        print("Decoding Job Description has failed")
    
    return json_result