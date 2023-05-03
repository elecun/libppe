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
    _yaw_gt_initial = 0.0 if "yaw_gt_initial" not in process_job else process_job["yaw_gt_initial"]
    _yaw_gt = np.array([], dtype=float) if "yaw_gt" not in process_job else np.array(process_job["yaw_gt"], dtype=float)+_yaw_gt_initial
    _yaw_direction = 1.0 if "yaw_direction" not in process_job else process_job["yaw_direction"]
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
            src_file = _path_prefix+filename # image full path
            
            print("%s is now processing..."%src_file) if _verbose else None
            
            raw_image = cv2.imread(src_file, cv2.IMREAD_UNCHANGED)
            
            if raw_image is not None:
                if raw_image.shape[2]==3: # if color image
                    raw_color = raw_image.copy()
                    raw_gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
                else:
                    raw_gray = raw_image.copy()
                    raw_color = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError("Image is empty")
                
            undist_raw_gray = cv2.undistort(raw_gray, intrinsic_mtx, distorsion_mtx, None, newcamera_mtx) # undistortion by camera parameters
            undist_raw_gray = cv2.bitwise_not(undist_raw_gray) # color inversion
            undist_raw_color = cv2.cvtColor(undist_raw_gray, cv2.COLOR_GRAY2BGR)
            
            # find markers printed on reference wafer
            if _python_version[0]==4 and _python_version[1]<7: # < opencv 4.7
                corners, ids, rejected = cv2.aruco.detectMarkers(undist_raw_gray, markerdict, parameters=markerparams)
            else:
                corners, ids, rejected = markerdetector.detectMarkers(undist_raw_gray)
            
            # found markers
            if ids is not None and ids.size>3:
                
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
                        #print("marker :",str_image_pos, str_world_pos) if _verbose else None
                        
                        cv2.putText(undist_raw_color, str_image_pos,(p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(undist_raw_color, str_world_pos,(p[0]+10, p[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.line(undist_raw_color, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,255,0), 1, cv2.LINE_AA)
                        cv2.line(undist_raw_color, (p[0],p[1]-10), (p[0],p[1]+10), (0,255,0), 1, cv2.LINE_AA)
                    cv2.imwrite(_path_prefix+"markers_"+filename, undist_raw_color)
                
                # compute 2D-3D corredpondence with Perspective N Point'
                # note] use undistorted image points to solve, then appply the distortion coefficient and camera matrix as pin hole camera model
                _, rVec, tVec = cv2.solvePnP(marker_centroids_on_wafer, marker_centroids_on_image, cameraMatrix=newcamera_mtx, distCoeffs=distorsion_mtx, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP)
                _, prVec, ptVec = cv2.solvePnP(marker_centroids_on_wafer, marker_centroids_on_image, cameraMatrix=intrinsic_mtx, distCoeffs=None, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP)
                R, jacobian = cv2.Rodrigues(rVec) # rotation vector to matrix
                
                # save yaw angles
                yaw_deg = _yaw_direction*np.rad2deg(math.atan2(-R[1][0], R[0][0]))
                gt_deg = _yaw_gt[fid]
                estimated_yaw_deg.append(yaw_deg)
                real_yaw_deg.append(gt_deg)

                #print("* Estimated rotation Angle(deg)", yaw_deg) if _verbose else None
                #print("* Ground Truth rotation Angle(deg)", gt_deg) if _verbose else None
                
                p_dic = {}
                p_dic["wafer_x"] = 0.0
                p_dic["wafer_x"] = 0.0
                p_dic["wafer_y"] = 0.0
                p_dic["wafer_z"] = 0.0
                p_dic["wafer_r"] = 0.0
                p_dic["wafer_p"] = 0.0
                p_dic["wafer_w"] = yaw_deg
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
        if estimated_yaw_deg is not None and len(estimated_yaw_deg)>0:
            estimated_yaw_deg = np.abs(np.array(estimated_yaw_deg))
            yaw_rmse = np.sqrt(np.mean((estimated_yaw_deg-real_yaw_deg)**2))
            yaw_mae = np.mean(np.abs(estimated_yaw_deg - real_yaw_deg))
            print("* ------<Rotation Error Test Result>------")
            print("* RMSE Rotation(Yaw deg) : ", yaw_rmse) # root mean square error
            print("* MAE Rotation(Yaw deg) : ", yaw_mae) # mean average error
            print("* ----------------------------------------")
            if _save_result:
                csv_filename = "yaw_rotation.csv"
                with open(csv_filename, 'w') as f:
                    rot_out_file = csv.writer(f)
                    rot_out_file.writerow(["index", "Estimated", "Ground Truth"])
                    for idx, deg in enumerate(real_yaw_deg):
                        rot_out_file.writerow([idx, estimated_yaw_deg[idx], real_yaw_deg[idx]])
                print("saved results in %s file"%(csv_filename))
        
                
        # dumps into json result        
        json_result = json.dumps(result_dic)
        
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