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
        
        # read reference wafer position as dictionary
        if "marker" not in process_param:
            raise ValueError("marker is not defined")
        else:
            if "coord" not in process_param["marker"]:
                raise ValueError("Marker coordinates are not defined")
        _wafer_marker_pos = json.loads(json.dumps(process_param["marker"]["coord"]))
        _wafer_marker_pos = {int(k):[int(i) for i in v] for k,v in _wafer_marker_pos.items()}
        
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
           
        
        result_dic = {} # estimated results (dictionary type for converting json)
        
        # file existance check
        for src_image in image_files_path:
            if not os.path.isfile(src_image):
                raise ValueError("%s file does not exist"%src_image)
            
        
        # processing
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
            
            
        if "files" in job and "path" in job:
            
            for job_file in job["files"]:
                job_file_path = job["path"]+job_file #image file to be processed
                
                # estimation processing
                if os.path.isfile(job_file_path): # if file exist
                    print(job_file, "is now processing...")
                    
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
                    
                    # preprocessing
                    ud_image_gray = cv2.bitwise_not(ud_image_gray)
                    #ud_image_gray = cv2.bilateralFilter(ud_image_gray, -1, 10, 5) #bilateral filter for edge enhancement
                    #_, ud_image_binary = cv2.threshold(ud_image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # binarization (marker loss occured)
                    
                    # find markers (API depends on the installed python version)
                    if python_version[0]==4 and python_version[1]<7:
                        corners, ids, rejected = cv2.aruco.detectMarkers(ud_image_gray, markerdict, parameters=markerparams)
                    else:
                        corners, ids, rejected = markerdetector.detectMarkers(ud_image_gray)
                    
                    # find matching wafer points
                    marker_centroid_pointset = [] #2d points
                    wafer_centroid_pointset = [] #3d points (z=0)
                    
                    for idx in range(0, len(ids)):
                        corner_points = corners[idx].reshape(4,2)
                        center_on_marker = np.mean(corner_points, axis=0, dtype=float)
                        center_on_wafer = np.array(param["marker"]["coord"][str(ids[idx,0])])
                        print("marker-wafer ", center_on_marker, center_on_wafer)
                        if center_on_marker.shape != center_on_wafer.shape:
                            raise ValueError("Geometric pointset dimension is not same")
                            
                        marker_centroid_pointset.append(center_on_marker)
                        wafer_centroid_pointset.append(center_on_wafer)
                        
                        # write to image
                        #cv2.circle(ud_image_color, center_on_marker.round().astype(int), 1, (0, 0, 255), 2)
                        cv2.circle(ud_image_color, tuple(center_on_marker.round().astype(int).reshape(1,-1)[0]), 1, (0, 0, 255), 2)
                        
                        #optical center (cx, cy)
                        cx = round(newcamera_mtx[0,2])
                        cy = round(newcamera_mtx[1,2])
                        cv2.line(ud_image_color, (cx-100,cy), (cx+100,cy), (0,0,255), 1, cv2.LINE_AA)
                        cv2.line(ud_image_color, (cx,cy-100), (cx,cy+100), (0,0,255), 1, cv2.LINE_AA)
                        
                        # image center
                        cv2.line(ud_image_color, (round(_raw_w/2)-100,round(_raw_h/2)), (round(_raw_w/2)+100,round(_raw_h/2)), (0,255,0), 1, cv2.LINE_AA)
                        cv2.line(ud_image_color, (round(_raw_w/2),round(_raw_h/2)-100), (round(_raw_w/2),round(_raw_h/2)+100), (0,255,0), 1, cv2.LINE_AA)
                        
                        #print("diff_x", cx-round(_raw_w/2))
                        #print("diff_y", cy-round(_raw_h/2))
                        
                        str_pos = "[%d] x=%2.2f,y=%2.2f"%(ids[idx], center_on_wafer[0], center_on_wafer[1])
                        cv2.putText(ud_image_color, str_pos,(int(center_on_marker[0]), int(center_on_marker[1]) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if len(ids)>3: # if marker found
                        # camera pose
                        image_pts_vec = np.array(marker_centroid_pointset, dtype=np.double)
                        wafer_pts_vec = np.array(wafer_centroid_pointset, dtype=np.double)
                        wafer_pts_vec = np.append(wafer_pts_vec, np.zeros(shape=(np.size(wafer_pts_vec, axis=0), 1), dtype=np.double),axis=1) # append Z column with 0
                        
                        # calc 2D-3D correnspondance
                        _, rVec, tVec = cv2.solvePnP(wafer_pts_vec, image_pts_vec, newcamera_mtx, distorsion_mtx, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP)
                        R, jacobian = cv2.Rodrigues(rVec) # rotation vector to matrix
                        
                        # testing for coordinate conversion (3D to 2D) --- ok
                        test_world_pts = np.array([[20.0, 60.0, 0.0]], dtype=np.double)
                        test_image_pts, jacobian = cv2.projectPoints(test_world_pts, rVec, tVec, newcamera_mtx, distorsion_mtx) # world to image coord (3D to 2D)
                        print("test image points : ", test_image_pts)
                        for pts in test_image_pts.reshape(-1,2):
                            p = pts.round().astype(int)
                            cv2.line(ud_image_color, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,255,0), 1, cv2.LINE_AA)
                            cv2.line(ud_image_color, (p[0],p[1]-10), (p[0],p[1]+10), (0,255,0), 1, cv2.LINE_AA)
                            cv2.circle(ud_image_color, tuple(pts.round().astype(int).reshape(1,-1)[0]), 1, (0, 0, 255), 2)
                            
                        # testing for 2D to 3D
                        Z = np.array([[0]])
                        points_undistorted = np.array([])
                        points = np.asmatrix(np.array([[1095.29546489-17, 411.07699445-22]]))
                        distorsion = np.matrix([[0.0, 0.0, 0.0, 0.0]])
                        points_undistorted = cv2.undistortPoints(points, intrinsic_mtx, distorsion_mtx)
                        points_undistorted = np.squeeze(points_undistorted, axis=1)
                        points_undistorted = np.append(points_undistorted, [0])
                        print(points_undistorted)
                        print("undistored",newcamera_mtx*np.asmatrix(points_undistorted).T)
                        #if len(points) > 0:
                            #points_undistorted = cv2.undistortPoints(np.expand_dims(points, axis=1), newcamera_mtx, distorsion_mtx, P=distorsion_mtx)
                            #points_undistorted = cv2.undistortPoints(points, newcamera_mtx, distorsion_mtx)
                        #points_undistorted = np.squeeze(points_undistorted, axis=1)
                        
                        result = []
                        for idxx in range(points_undistorted.shape[0]):
                            z = Z[0] if len(Z) == 1 else Z[idxx]
                            print("z",z)
                            x = (points_undistorted[idxx, 0] ) / float(param['fx']) * z
                            y = (points_undistorted[idxx, 1] ) / float(param['fy']) * z
                            result.append([x, y, z])
                        print("result",result)
                        
                        a = undistort_unproject_pts(points, newcamera_mtx, distorsion_mtx)
                        print("a", a)
                            
                        
                        # testing for coordinate conversion (image to world)
                        #print("optical center position(pixel) : ", cx, cy)
                        uv = np.array([[1097, 615, 1]], dtype=int).T
                        cv2.line(ud_image_color, (uv.ravel()[0]-20,uv.ravel()[1]), (uv.ravel()[0]+20,uv.ravel()[1]), (255,0,0), 1, cv2.LINE_AA)
                        cv2.line(ud_image_color, (uv.ravel()[0],uv.ravel()[1]-20), (uv.ravel()[0],uv.ravel()[1]+20), (255,0,0), 1, cv2.LINE_AA)
                        
                        R_1t = np.asmatrix(np.linalg.inv(R)*np.asmatrix(tVec))
                        R_1M_1 = np.asmatrix(np.linalg.inv(R))*np.asmatrix(np.linalg.inv(newcamera_mtx))
                        R_1M_1uv = R_1M_1*np.asmatrix(uv)
                        s = 100 + R_1t[2,0]/R_1M_1uv[2,0]
                        
                        P = np.asmatrix(np.linalg.inv(R))*(s*np.asmatrix(np.linalg.inv(newcamera_mtx))*np.asmatrix(uv)-np.asmatrix(tVec))
                        print("P",P)
                        
                        # testing for coordinate conversion (world to image) --- ok
                        '''
                        test_world_pts = np.array([[20.0, 60.0, 0.0],[-20.0, 60.0, 0.0]], dtype=np.double)
                        test_image_pts, jacobian = cv2.projectPoints(test_world_pts, rVec, tVec, newcamera_mtx, distorsion_mtx) # world to image coord (3D to 2D)
                        print("test image points : ", test_image_pts)
                        for pts in test_image_pts.reshape(-1,2):
                            p = pts.round().astype(int)
                            cv2.line(ud_image_color, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,255,0), 1, cv2.LINE_AA)
                            cv2.line(ud_image_color, (p[0],p[1]-10), (p[0],p[1]+10), (0,255,0), 1, cv2.LINE_AA)
                            cv2.circle(ud_image_color, tuple(pts.round().astype(int).reshape(1,-1)[0]), 1, (0, 0, 255), 2)
                        '''
                        
                        
                        '''
                        A = np.linalg.inv(R)*np.linalg.inv(newcamera_mtx)*uv
                        print("A", A[0,0])
                        print("tvec",tVec)
                        Rt = np.linalg.inv(newcamera_mtx)*tVec
                        print("Rt", Rt[0,0])
                        Rt_list = np.array(Rt).reshape(-1,).tolist()
                        print("rt", np.array(Rt).reshape(-1,).tolist())
                        
                        wcPoint=(0+Rt[2,0])/A[2,0]*A-Rt
                        print("world",wcPoint)
                        '''
                        
                        #print(Rt, Rt.shape)
                        
                        #xyz_c = np.linalg.inv(newcamera_mtx).dot(uv)
                        #xyz_c = xyz_c - tVec
                        #XYZ = np.linalg.inv(R).dot(xyz_c)
                        #print("world coord : ", XYZ)
                        
                        cv2.imwrite("marker_centroid_"+job_file, ud_image_color)
                    else:
                        print("No markers found")
                    
                
                        
                        
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
                        
                        #cv2.imwrite("marker_centroid_"+job_file, raw_image)
                    
                    
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
                else:
                    ValueError("No image file in the path")
            
        else:
            raise ValueError("files and path configuration are not defined")
        
        # output to return
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