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
    
    python_version = list(map(int, cv2.__version__.split(".")))
    
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
        newcamera_mtx = np.matrix(newcamera_mtx, dtype=float)
        #print(np.matrix(newcamera_mtx, dtype=float))
        
        # marker
        if python_version[0]==4:
            if python_version[1]<7:
                markerdict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
                markerparams = cv2.aruco.DetectorParameters_create()
                markerparams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            else:
                markerdict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
                markerparams = cv2.aruco.DetectorParameters()
                markerparams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                markerdetector = cv2.aruco.ArucoDetector(markerdict, markerparams)
        else:
            raise ValueError("Your python version is not supported")
        
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
                    if python_version[0]==4 and python_version[1]<7:
                        corners, ids, rejected = cv2.aruco.detectMarkers(ud_image_gray, markerdict, parameters=markerparams)
                    else:
                        corners, ids, rejected = markerdetector.detectMarkers(ud_image_gray)
                    
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
                        cv2.putText(ud_image_color, str_pos,(int(center_on_marker[0]), int(center_on_marker[1]) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # camera pose
                    image_pts_vec = np.array(marker_centroid_pointset, dtype=np.double)
                    wafer_pts_vec = np.array(wafer_centroid_pointset, dtype=np.double)
                    wafer_pts_vec = np.append(wafer_pts_vec, np.zeros(shape=(np.size(wafer_pts_vec, axis=0), 1), dtype=np.double),axis=1) # append Z column with 0
                    
                    # temporary calc : optical center
                    _, rVec, tVec = cv2.solvePnP(wafer_pts_vec, image_pts_vec, newcamera_mtx, distorsion_mtx, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP)
                    R, jacobian = cv2.Rodrigues(rVec) # rotation vector to matrix
                    
                    # testing for coordinate conversion (world to image) --- ok
                    test_world_pts = np.array([[20.0, 60.0, 0.0],[-20.0, 60.0, 0.0]], dtype=np.double)
                    test_image_pts, jacobian = cv2.projectPoints(test_world_pts, rVec, tVec, newcamera_mtx, distorsion_mtx) # world to image coord (3D to 2D)
                    print("test image points : ", test_image_pts)
                    for pts in test_image_pts.reshape(-1,2):
                        p = pts.round().astype(int)
                        cv2.line(ud_image_color, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,255,0), 1, cv2.LINE_AA)
                        cv2.line(ud_image_color, (p[0],p[1]-10), (p[0],p[1]+10), (0,255,0), 1, cv2.LINE_AA)
                        cv2.circle(ud_image_color, pts.round().astype(int), 1, (0, 0, 255), 2)
                        
                    
                    # testing for coordinate conversion (image to world)
                    uv = np.array([[999, 520, 1]], dtype=int).T
                    print("test uv", uv.ravel())
                    cv2.line(ud_image_color, (uv.ravel()[0]-20,uv.ravel()[1]), (uv.ravel()[0]+20,uv.ravel()[1]), (255,0,0), 1, cv2.LINE_AA)
                    cv2.line(ud_image_color, (uv.ravel()[0],uv.ravel()[1]-20), (uv.ravel()[0],uv.ravel()[1]+20), (255,0,0), 1, cv2.LINE_AA)
                    
                    RMu = np.linalg.inv(R)*np.linalg.inv(newcamera_mtx)*uv
                    print(type(tVec), tVec.shape)
                    Rt = np.linalg.inv(newcamera_mtx)*uv
                    print(Rt)
                    
                    #xyz_c = np.linalg.inv(newcamera_mtx).dot(uv)
                    #xyz_c = xyz_c - tVec
                    #XYZ = np.linalg.inv(R).dot(xyz_c)
                    #print("world coord : ", XYZ)
                    
                    
    #                 invR_x_invM_x_uv1=rotationMatrix.inv()*cameraMatrix.inv()*screenCoordinates;
	# invR_x_tvec      =rotationMatrix.inv()*translationVector;
	# wcPoint=(Z+invR_x_tvec.at<double>(2, 0))/invR_x_invM_x_uv1.at<double>(2, 0)*invR_x_invM_x_uv1-invR_x_tvec;
	# cv::Point3f worldCoordinates(wcPoint.at<double>(0, 0), wcPoint.at<double>(1, 0), wcPoint.at<double>(2, 0));
	# std::cerr << "World Coordinates" << worldCoordinates << std::endl << std::endl;
	# std::cout 	<< screenCoordinates.at<double>(0, 0) << ","
	# 		<< screenCoordinates.at<double>(1, 0) << ","
	# 		<< worldCoordinates.x << ","
	# 		<< worldCoordinates.y << std::endl;
                    
                    
                    theta = np.linalg.norm(rVec) # rotation angle(radian)
                    #print(np.rad2deg(theta))
                    r_vec = np.array(rVec/theta).reshape(-1) # rotation unit vector
                    
                    # camera position
                    #print("t vector : ", tVec)
                    #print("r matrix : ", R.T)
                    camera_position = np.matrix(-R.T)*(np.matrix(tVec))
                    print("camera pos : ", camera_position)
                    # for p in image_pts.reshape(-1,2):
                    #     cv2.circle(ud_image_color, p.round().astype(int), 1, (0, 0, 255), 2)
                    
                    
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