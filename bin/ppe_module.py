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
import csv
import pathlib
from torchvision.models import detection
from models.experimental import attempt_load
from utils.general import non_max_suppression

# define working path
WORKING_PATH = pathlib.Path(__file__).parent
        

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
 [Note] if it has a problem of performance, try code below into c++
'''
def estimate(process_param, process_job):
    result_dic = {} # estimated results (dictionary type for converting json)
             
    '''
     getting developer options
    '''
    _verbose = 0 if "verbose" not in process_job else int(process_job["verbose"])
    _use_camera = 0 if "use_camera" not in process_job else int(process_job["use_camera"])
    _save_result = 0 if "save_result" not in process_job else int(process_job["save_result"])
    _x_direction = 1.0 if "x_direction" not in process_job else process_job["x_direction"]
    _y_direction = 1.0 if "y_direction" not in process_job else process_job["y_direction"]
    _yaw_direction = 1.0 if "yaw_direction" not in process_job else process_job["yaw_direction"]
    _forktip_model = WORKING_PATH.joinpath("forktip_type1.pt") if "forktip_model" not in process_job else WORKING_PATH.joinpath(process_job["forktip_model"])
    _working_path = WORKING_PATH if "path" not in process_job else WORKING_PATH.joinpath(process_job["path"])

    '''
    forktip detection model load
    '''
    # fork_detection_model = torch.load(_forktip_model)
    fork_detection_model = attempt_load(_forktip_model, device='cpu')
    fork_detection_model.eval()

        
    '''
     getting system & library check
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
        image_files_path = np.array([_working_path / pathlib.Path(f) for f in _image_files])
        
        
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
        estimated_x_pos = []
        estimated_y_pos = []
        
        for fid, filename in enumerate(_image_files):
            src_image = str(_working_path / pathlib.Path(filename))

            if _verbose:
                print("%s is now processing..."%src_image)
            
            raw_image = cv2.imread(src_image, cv2.IMREAD_UNCHANGED)
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
            undist_raw_color = cv2.cvtColor(undist_raw_gray, cv2.COLOR_GRAY2BGR) # for draw results
            undist_color_result = cv2.cvtColor(undist_raw_gray, cv2.COLOR_GRAY2BGR) # for draw results
            
            
            # find markers printed on reference wafer
            if _python_version[0]==4 and _python_version[1]<7:
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
                        
                        str_image_pos = "on image : [%d] x=%2.2f,y=%2.2f"%(ids[idx].item(), pts[0], pts[1])
                        str_world_pos = "on wafer : x=%2.2f,y=%2.2f"%(marker_centroids_on_wafer[idx][0], marker_centroids_on_wafer[idx][1])
                        if _verbose:
                            print("marker :",str_image_pos, str_world_pos)
                        cv2.putText(undist_color_result, str_image_pos,(p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(undist_color_result, str_world_pos,(p[0]+10, p[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.line(undist_color_result, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,255,0), 1, cv2.LINE_AA)
                        cv2.line(undist_color_result, (p[0],p[1]-10), (p[0],p[1]+10), (0,255,0), 1, cv2.LINE_AA)
                
            
                # compute 2D-3D corredpondence with Perspective N Point'
                # note] use undistorted image points to solve, then appply the distortion coefficient and camera matrix as pin hole camera model
                _, rVec, tVec = cv2.solvePnP(marker_centroids_on_wafer, marker_centroids_on_image, cameraMatrix=newcamera_mtx, distCoeffs=distorsion_mtx, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP)
                _, prVec, ptVec = cv2.solvePnP(marker_centroids_on_wafer, marker_centroids_on_image, cameraMatrix=intrinsic_mtx, distCoeffs=None, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP)
                R, jacobian = cv2.Rodrigues(rVec) # rotation vector to matrix
                
                # save estimated results
                estimated_x = (_x_direction*ptVec[0])[0]
                estimated_y = (_y_direction*ptVec[1])[0]
                yaw_deg = _yaw_direction*np.rad2deg(math.atan2(-R[1][0], R[0][0]))
                
                estimated_x_pos.append(estimated_x)
                estimated_y_pos.append(estimated_y)
                estimated_yaw_deg.append(yaw_deg)
                
                estimated_yaw_deg.append(np.rad2deg(math.atan2(-R[1][0], R[0][0])))

                if _verbose:
                    print("Rotation Angle", -1*np.rad2deg(math.atan2(-R[1][0], R[0][0])))
                    
                if _save_result:
                    for idx, pts in enumerate(marker_centroids_on_image):
                        p = tuple(pts.round().astype(int))
                        
                        str_image_pos = "on image : [%d] x=%2.2f,y=%2.2f"%(ids[idx].item(), pts[0], pts[1])
                        str_world_pos = "on wafer : x=%2.2f,y=%2.2f"%(marker_centroids_on_wafer[idx][0], marker_centroids_on_wafer[idx][1])
                        #print("marker :",str_image_pos, str_world_pos) if _verbose else None
                        
                        cv2.putText(undist_color_result, str_image_pos,(p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        cv2.putText(undist_color_result, str_world_pos,(p[0]+10, p[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        cv2.line(undist_color_result, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,255,0), 1, cv2.LINE_AA)
                        cv2.line(undist_color_result, (p[0],p[1]-10), (p[0],p[1]+10), (0,255,0), 1, cv2.LINE_AA)
                        
                        op_x = np.round(_cx).astype(int)
                        op_y = np.round(_cy).astype(int)
                        str_image_op_pos = "OC_img : x=%2.2f,y=%2.2f"%(_cx, _cy)
                        str_world_op_pos = "OC_real : x=%2.2f,y=%2.2f"%(_x_direction*ptVec[0].item(), _y_direction*ptVec[1].item())
                        cv2.putText(undist_color_result, str_image_op_pos,(op_x+10, op_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.putText(undist_color_result, str_world_op_pos,(op_x+10, op_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.line(undist_color_result, (op_x-10,op_y), (op_x+10,op_y), (0,0,255), 1, cv2.LINE_AA) # optical center
                        cv2.line(undist_color_result, (op_x,op_y-10), (op_x,op_y+10), (0,0,255), 1, cv2.LINE_AA)
                
                # testing for 3D to 2D
                #np.array([[130.0, 210.0, 0.0]], dtype=float)
                image_pts = obj_coord2pixel_coord([130.0, 210.0, 0.0], rVec, tVec, newcamera_mtx, distorsion_mtx, verbose=_verbose)
                if _save_result:
                    image_pts = image_pts.squeeze()
                    str_pos = "x=%2.2f,y=%2.2f"%(image_pts[0], image_pts[1])
                    image_ptsi = (image_pts.round().astype(int))
                    cv2.putText(undist_color_result, str_pos,(image_ptsi[0]+10, image_ptsi[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.line(undist_color_result, (image_ptsi[0]-10,image_ptsi[1]), (image_ptsi[0]+10,image_ptsi[1]), (0,255,255), 1, cv2.LINE_AA)
                    cv2.line(undist_color_result, (image_ptsi[0],image_ptsi[1]-10), (image_ptsi[0],image_ptsi[1]+10), (0,255,255), 1, cv2.LINE_AA)
                    

                # finally save image
                if _save_result:
                    cv2.imwrite(str(_working_path / pathlib.Path("out_"+filename)), undist_color_result)
                
                
                # final outputs
                p_dic = {}
                p_dic["wafer_x"] = estimated_x
                p_dic["wafer_y"] = estimated_y
                p_dic["wafer_z"] = _laser_distance[fid]
                p_dic["wafer_r"] = 0.0
                p_dic["wafer_p"] = 0.0
                p_dic["wafer_w"] = estimated_yaw_deg
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

            # fork-tip detection
            try:
                with torch.no_grad():
                    print("detecting forktip...")
                    img = undist_raw_color[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to("cpu")
                    img = img.float()
                    img /= 255.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = fork_detection_model(img, augment=False)[0]
                    print('pred shape:', pred.shape)
                    pred = non_max_suppression(pred, 0.25, 0.45, classes=['tip'], agnostic=False)
                    det = pred[0]
                    print('det shape:', det.shape)
                    print(det)
                    
            except TypeError as e:
                print("Error : ", e)
        
    except json.decoder.JSONDecodeError :
        print("Error : Decoding Job Description has failed")
    except (ValueError, UndefinedParamError) as e:
        print("Error : ",e)
        
                
    # dumps into json result        
    json_result = json.dumps(result_dic)
    return json_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', required=True, help="Configuration file")
    parser.add_argument('--job', nargs='?', required=True, help="Job file")
    args = parser.parse_args()
    
    config_path = WORKING_PATH / pathlib.Path(args.config)
    job_path = WORKING_PATH / pathlib.Path(args.job)

    try :
        with open(config_path, 'r') as file:
            config = json.load(file)
        with open(job_path, 'r') as file:
            job = json.load(file)
    except json.decoder.JSONDecodeError:
        print("Job & Parameters decoding error is occured")
    except FileNotFoundError:
        print("File does not exist")
    else:
        #estimate(json.dumps(config), json.dumps(job))
        estimate(config, job)