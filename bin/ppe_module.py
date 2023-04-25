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
        newcamera_mtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic_mtx, distorsion_mtx,(_w, _h), 1, (_w, _h))
        
        # marker
        markerdict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        markerparams = cv2.aruco.DetectorParameters_create()
        markerparams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        # camera coord. / transformation matrix
        coord = np.matrix([[1,0,0,float(param['coord'][0])],[0,1,0,float(param['coord'][1])],[0,0,1,float(param['coord'][2])],[0,0,0,1]])
        
        # file existance check
        result_dic = {}
        if "files" in job and "path" in job:
            distance = []
            for d in job["laser_wafer_distance"]:
                distance.append(d)
                
            for job_file in job["files"]:
                job_file_path = job["path"]+job_file
                
                # estimation processing
                if os.path.isfile(job_file_path): # if file exist
                    print(job_file, "is processing...")
                    
                    # image read and color conversion
                    raw_image = cv2.imread(job_file_path, cv2.IMREAD_UNCHANGED)
                    
                    # image undistorsion by calibration parameters
                    ud_image = cv2.undistort(raw_image, intrinsic_mtx, distorsion_mtx, None, newcamera_mtx)
                    ud_image_color = ud_image.copy()
                    
                    # color conversion
                    ud_image_gray = cv2.cvtColor(ud_image, cv2.COLOR_BGR2GRAY)
                    _raw_h, _raw_w, _raw_c = raw_image.shape
                    if _raw_h!=_h and _raw_w!=_w:
                        raise ValueError("Image shaoe is different from your configurations")
                    
                    # preprocessing (invert, binarization)
                    ud_image_gray = cv2.bitwise_not(ud_image_gray)
                    #ud_image_gray = cv2.bilateralFilter(ud_image_gray, -1, 10, 5)
                    #_, ud_image_binary = cv2.threshold(ud_image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    
                    # find markers
                    corners, ids, rejected = cv2.aruco.detectMarkers(ud_image_gray, markerdict, parameters=markerparams)
                    
                    # find matching wafer points
                    marker_centroid_pointset = []
                    wafer_centroid_pointset = []
                    for idx in range(0, len(ids)):
                        corner_points = corners[idx].reshape(4,2)
                        center_on_marker = np.mean(corner_points, axis=0, dtype=float)
                        center_on_wafer = np.array(param["marker"]["coord"][str(ids[idx,0])])
                        if center_on_marker.shape != center_on_wafer.shape:
                            raise ValueError("Geometric pointset dimension is not same")
                            
                        marker_centroid_pointset.append(center_on_marker)
                        wafer_centroid_pointset.append(center_on_wafer)
                        
                        # write to image
                        cv2.circle(ud_image_color, center_on_marker.round().astype(int), 1, (0, 0, 255), 2)
                        
                        #optical center (cx, cy)
                        cx = round(newcamera_mtx[0,2])
                        cy = round(newcamera_mtx[1,2])
                        cv2.line(ud_image_color, (cx-100,cy), (cx+100,cy), (0,0,255), 1, cv2.LINE_AA)
                        cv2.line(ud_image_color, (cx,cy-100), (cx,cy+100), (0,0,255), 1, cv2.LINE_AA)
                        
                        # image center
                        cv2.line(ud_image_color, (round(_raw_w/2)-100,round(_raw_h/2)), (round(_raw_w/2)+100,round(_raw_h/2)), (0,255,0), 1, cv2.LINE_AA)
                        cv2.line(ud_image_color, (round(_raw_w/2),round(_raw_h/2)-100), (round(_raw_w/2),round(_raw_h/2)+100), (0,255,0), 1, cv2.LINE_AA)
                        
                        str_pos = "[%d] x=%2.2f,y=%2.2f"%(ids[idx], center_on_wafer[0], center_on_wafer[1])
                        cv2.putText(ud_image_color, str_pos,(int(center_on_marker[0]), int(center_on_marker[1]) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # camera pose
                    image_pts_vec = np.array(marker_centroid_pointset, dtype=np.double)
                    wafer_pts_vec = np.array(wafer_centroid_pointset, dtype=np.double)
                    wafer_pts_vec = np.append(wafer_pts_vec, np.zeros(shape=(np.size(wafer_pts_vec, axis=0), 1), dtype=np.double),axis=1) # append Z column with 0
                    
                    # temporary calc : optical center
                    _, rVec, tVec = cv2.solvePnP(wafer_pts_vec, image_pts_vec, newcamera_mtx, distorsion_mtx, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP)
                    R, jacobian = cv2.Rodrigues(rVec) # rotation vector to matrix
                    
                    world_pts = np.array([[0.0, 60.0, 0.0]], dtype=np.double)
                    print(wafer_pts_vec)
                    imagePoints, jacobian = cv2.projectPoints(world_pts, rVec, tVec, newcamera_mtx, distorsion_mtx) # world to image coord
                    print(imagePoints)
                    for pts in imagePoints.reshape(-1,2):
                        cv2.circle(ud_image_color, pts.round().astype(int), 1, (0, 255, 0), 2)
                    
                    theta = np.linalg.norm(rVec) # rotation angle(radian)
                    #print(np.rad2deg(theta))
                    r_vec = np.array(rVec/theta).reshape(-1) # rotation unit vector
                    
                    # camera position
                    #print("t vector : ", tVec)
                    #print("r matrix : ", R.T)
                    camera_position = np.matrix(-R.T)*(np.matrix(tVec))
                    #print("camera pos : ", camera_position)
                    
                    
                    #print("rotation matrix transpose : ", R.T)
                    #print("rotation matrix inverse : ", np.linalg.inv(R))
                    
                    axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
                    imgpts, jac = cv2.projectPoints(axis, rVec, tVec, newcamera_mtx, distorsion_mtx)
                    #print(imgpts)
                    corner = tuple(corners[0].ravel())
                    #print((round(newcamera_mtx[0,2]),round(newcamera_mtx[1,2])))
                    #print(tuple(round(c) for c in imgpts[0].ravel()))
                    
                    #print(imgpts)
                    
                    cv2.imwrite("marker_centroid_"+job_file, ud_image_color)
                    
                    
                    #print(R.T)
                    #R = Rt.transpose()
                    pos = -R * tVec.reshape(-1)
                    #print("pos", pos)
                    #print(pos)
                    roll = math.atan2(-R[2][1], R[2][2])
                    pitch = math.asin(R[2][0])
                    yaw = math.atan2(-R[1][0], R[0][0])
                    #print("yaw : ",yaw*180/3.14)
                    
                    
                    
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