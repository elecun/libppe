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
import platform
import gc
from torchvision.models import detection
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.plots import colors, save_one_box
from math import atan2, cos, sin, sqrt, pi

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# define working path
WORKING_PATH = pathlib.Path(__file__).parent
        

'''
Object 3D coordinates convert to Image Pixel 2D coordinates
* input : np.array([[x,y,z]], dtype=np.double)
ex)
# for testing 3d-2d
pos_3d = np.array([80.0, -120.0, 0], dtype=np.double).reshape(1,-1)
pos_2d, _ = obj_coord2pixel_coord(pos_3d, rVec, tVec, newcamera_mtx, distorsion_mtx)
if _save_result:
    print(pos_2d)
    p = pos_2d.round().astype(int)
    cv2.line(undist_raw_color, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,255,255), 1, cv2.LINE_AA)
    cv2.line(undist_raw_color, (p[0],p[1]-10), (p[0],p[1]+10), (0,255,255), 1, cv2.LINE_AA)
'''
def obj_coord2pixel_coord(obj_coord, rvec, tvec, camera_matrix, dist_coeff, verbose=False):
    if type(obj_coord) is not np.ndarray:
        raise Exception("Input position vector should be type of numpy ndarray.")
    if obj_coord.shape[1] !=3:
        raise Exception("Shape must be (1,3)")
    
    image_pts, jacobian = cv2.projectPoints(obj_coord, rvec, tvec, cameraMatrix=camera_matrix, distCoeffs=dist_coeff) #3D to 2D
    if verbose:
        print("Coordinates 3D to 2D : ", obj_coord.squeeze(), np.array(image_pts).squeeze())
    return image_pts.squeeze(), jacobian

'''
Image pixel 2d coordinates convert to object 3d coordinates
* input : np.array([pixel_x, pixel_y])
'''
def pixel_coord2obj_coord(pixel_coord, rvec, tvec, camera_matrix, dist_coeff, verbose=False):
    if type(pixel_coord) is not np.ndarray:
        raise Exception("Input position vector should be type of numpy ndarray.")
    if pixel_coord.shape[1] !=2:
        raise Exception("Shape must be (1,2)")
    
    R, jacobian = cv2.Rodrigues(rvec) # rotation vector to matrix
    point_2D_homogeneous = np.append(pixel_coord, [[1]], axis=1).reshape(-1, 1) # homogeneous form

    point_3D_camera_homogeneous = np.dot(np.linalg.inv(camera_matrix), point_2D_homogeneous)
    point_3D_camera_homogeneous = np.vstack((point_3D_camera_homogeneous, [1]))  # Add homogeneous coordinate
    point_3D_camera = np.dot(np.hstack((R, tvec.reshape(3, 1))), point_3D_camera_homogeneous)
    point_3D_camera = point_3D_camera[:3, 0]  # Convert homogeneous coordinate to 3D point
    return point_3D_camera.squeeze()
    

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


def canny_auto(image, sigma=0.1):
    v = np.mean(image)
    lower = int(max(0, (1.0-sigma)*v))
    upper = int(min(255, (1.0+sigma)*v))
    edged = cv2.Canny(image, lower, upper)
    return edged


'''
Forktip detection model execution by YOLOv5s model (only single processing)
'''
@smart_inference_mode()
def detect_forktip(
    weights=WORKING_PATH / 'forktip_type1.pt',  # model path or triton URL
    source=WORKING_PATH / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
    data=WORKING_PATH / 'data/forktip.yaml',  # dataset.yaml path
    imgsz=(1280, 960),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=WORKING_PATH / 'runs/detect',  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)

    device = torch.device('cpu')
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    batch_size = 1
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # inference
    model.warmup(imgsz=(1 if pt or model.triton else batch_size, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    detected = []
    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255  # normalize
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            pred = model(im, augment=augment)

        # apply non maximum supression
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        # process prediction
        for i, det in enumerate(pred):
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            if len(pred):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                detected.append(det.numpy().ravel())
    return detected

'''
 Undefined Parameter key Exception
'''
class UndefinedParamError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
    

'''
find intersect point between two line segments
'''
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(round(x)), int(round(y))


'''
 estimation processing with already saved image frame (from job description)
 [Note] if it has a problem of performance, try code below into c++
'''
def estimate(process_param, process_job):
             
    try:
        # read parameters from configuration and job description file
        _verbose = 0 if "verbose" not in process_job else int(process_job["verbose"])
        _use_camera = 0 if "use_camera" not in process_job else int(process_job["use_camera"])
        _save_result = 0 if "save_result" not in process_job else int(process_job["save_result"])
        _x_direction = 1.0 if "x_direction" not in process_job else process_job["x_direction"]
        _y_direction = 1.0 if "y_direction" not in process_job else process_job["y_direction"]
        _yaw_direction = 1.0 if "yaw_direction" not in process_job else process_job["yaw_direction"]
        _forktip_roi_confidence = 0.5 if "forktip_roi_confidence" not in process_job else process_job["forktip_roi_confidence"]
        _forktip_model = WORKING_PATH.joinpath("forktip_type1.pt") if "forktip_model" not in process_job else WORKING_PATH.joinpath(process_job["forktip_model"])
        _forktip_roi_padding = [0,0,0,0] if "forktip_roi_padding" not in process_job else process_job["forktip_roi_padding"] # top, right, bottom, left
        _forktip_line_select = 100 if "forktip_line_select" not in process_job else process_job["forktip_line_select"] # minimum line length in pixel
        _working_path = WORKING_PATH if "path" not in process_job else WORKING_PATH.joinpath(process_job["path"])
        _coord_offset = [0,0,0] if "coord" not in process_param else process_param["coord"]
        _resolution = [1280, 960] if "resolution" not in process_param else process_param["resolution"]
        _w, _h = _resolution
        _camera_parameter = [float(process_param[key]) for key in ("fx","fy", "cx", "cy", "coeff_k1", "coeff_k2", "coeff_p1", "coeff_p2")]
        _fx, _fy, _cx, _cy, _k1, _k2, _p1, _p2 = _camera_parameter
        _wafer_diameter = 300.0 if "diameter" not in process_param["wafer"] else float(process_param["wafer"]["diameter"])
        if "coord" in process_param["marker"]:
            _wafer_marker_pos = json.loads(json.dumps(process_param["marker"]["coord"]))
            _wafer_marker_pos = {int(k):[int(i) for i in v] for k,v in _wafer_marker_pos.items()} # changed origin
            
        if "files" in process_job and "path" in process_job:
            _image_filenames = np.array(process_job["files"])
            _image_files_path = np.array([_working_path / pathlib.Path(f) for f in _image_filenames])
        else:
            raise ValueError("image file path and files are missing..")
        
        if "laser_wafer_distance" in process_job:
            _laser_distance = np.array(process_job["laser_wafer_distance"])
        else:
            raise ValueError("laser distance values are missing..")
    
    except:
        print("Error occurred while read some parameters..")
        
    
    estimate_result = {}
    try:
        # set camera parameters
        intrinsic_mtx = np.matrix([[_fx, 0.000000, _cx], [0.000000, _fy, _cy], [0.000000, 0.000000, 1.000000]])
        distorsion_mtx = np.matrix([_k1, _k2, _p1, _p2, 0.])
        newcamera_mtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic_mtx, distorsion_mtx,(_w, _h), 1)
        newcamera_mtx = np.matrix(newcamera_mtx, dtype=float)
        
        # ready for marker detection depending on OpenCV Version
        _python_version = list(map(int, cv2.__version__.split(".")))
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
            raise RuntimeError("Not supported python version :", _python_version)
    
        # pre-check source data : file existance check
        for src_image in _image_files_path:
            if not os.path.isfile(src_image):
                raise FileNotFoundError("%s file does not exist"%src_image)
            
        # size of files and laser distance sould be same
        if _laser_distance.size != _image_filenames.size:
            raise ValueError("Laser Distance data and Image file size should be same.")        
        
        # processing
        estimated_yaw_deg = []
        estimated_x_pos = []
        estimated_y_pos = []
        
        # processing for each image file
        for fid, image_path in enumerate(_image_files_path):
            print("({}) {} is now processing...".format(fid, image_path)) if _verbose else None
            
            # read image and ready grayscale and color image both
            raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if raw_image.shape[2]==3: # if color image
                raw_color = raw_image.copy()
                raw_gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
            else:
                raw_gray = raw_image.copy()
                raw_color = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
                
            # apply undistortion with camera parameters
            undist_raw_gray = cv2.undistort(raw_gray, intrinsic_mtx, distorsion_mtx, None, newcamera_mtx)
            undist_raw_color = cv2.undistort(raw_color, intrinsic_mtx, distorsion_mtx, None, newcamera_mtx) # for output visualization
            undist_raw_gray_inv = cv2.bitwise_not(undist_raw_gray) # color inversion, because of the light fork and dark wafer background
            
            # save undist image for forktip detection
            export_undist_image = os.path.splitext(image_path)[0]+"_undist.png"
            cv2.imwrite(export_undist_image, undist_raw_gray)
                
             # marker detection   
            if _python_version[0]==4 and _python_version[1]<7:
                corners, ids, rejected = cv2.aruco.detectMarkers(undist_raw_gray_inv, markerdict, parameters=markerparams)
            else:
                corners, ids, rejected = markerdetector.detectMarkers(undist_raw_gray_inv)
                
            # coordination calculation with detected marker position
            if ids is not None and ids.size>3:
                marker_centroids_on_image = []
                marker_centroids_on_wafer = []
                
                for idx, marker_id in enumerate(ids.squeeze()):
                    corner = corners[idx].squeeze()    
                    centroid = np.mean(corner, axis=0, dtype=float)            
                    marker_centroids_on_image.append(centroid)
                    marker_centroids_on_wafer.append(_wafer_marker_pos[marker_id])
                    if _verbose:
                        print("Detected marker centroid position on wafer :", marker_id, _wafer_marker_pos[marker_id])
                        
                    # detected marker result display    
                    if _save_result:
                        p = tuple(centroid.round().astype(int))
                        str_image_pos = "image: [%d] %2.2f,%2.2f"%(ids[idx].item(), centroid[0], centroid[1])
                        str_world_pos = "wafer: %2.2f,%2.2f"%(marker_centroids_on_wafer[idx][0], marker_centroids_on_wafer[idx][1])
                        
                        cv2.putText(undist_raw_color, str_image_pos,(p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(undist_raw_color, str_world_pos,(p[0]+10, p[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.line(undist_raw_color, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,255,0), 1, cv2.LINE_AA)
                        cv2.line(undist_raw_color, (p[0],p[1]-10), (p[0],p[1]+10), (0,255,0), 1, cv2.LINE_AA)
                    
                marker_centroids_on_image = np.array(marker_centroids_on_image)
                marker_centroids_on_wafer = np.array(marker_centroids_on_wafer)
                marker_centroids_on_wafer = np.append(marker_centroids_on_wafer, np.zeros(shape=(np.size(marker_centroids_on_wafer, axis=0), 1), dtype=np.double),axis=1) # column add
                
                
                if marker_centroids_on_image.shape[0] != marker_centroids_on_wafer.shape[0]:
                    raise ValueError("Marker pointset dimension is not same")
                
                # compute 2D-3D corredpondence with Perspective N Point'
                # note] use undistorted image points to solve, then appply the distortion coefficient and camera matrix as pin hole camera model
                _, rVec, tVec = cv2.solvePnP(marker_centroids_on_wafer, marker_centroids_on_image, cameraMatrix=newcamera_mtx, distCoeffs=distorsion_mtx, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP)
                _, prVec, ptVec = cv2.solvePnP(marker_centroids_on_wafer, marker_centroids_on_image, cameraMatrix=intrinsic_mtx, distCoeffs=None, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP) #pinhole camera model
                R, jacobian = cv2.Rodrigues(rVec)
                pR, pjacobian = cv2.Rodrigues(prVec)
                
                
                # markerset pose estimation results
                estimated_x = (_x_direction*ptVec[0])[0]
                estimated_y = (_y_direction*ptVec[1])[0]
                estimated_z = (_laser_distance[fid])
                yaw_deg = _yaw_direction*np.rad2deg(math.atan2(-R[1][0], R[0][0]))
                print("Reference Wafer 6D Pose Estimation Result(for wafer coord. system):", estimated_x, estimated_y, _laser_distance[fid], 0, 0, yaw_deg) # 웨이퍼기준 공간좌표계에서 카메라의 광축이 바라보는 지점의 좌표
                
                # coord offset translation
                estimated_x_cam = estimated_x+_coord_offset[0]
                estimated_y_cam = estimated_y+_coord_offset[1]
                estimated_z_cam = estimated_z+_coord_offset[2]
                print("Reference Wafer 6D Pose Estimation Result(for camera coord. system):", estimated_x_cam, estimated_y_cam, estimated_z_cam, 0, 0, yaw_deg) # 웨이퍼기준 공간좌표계에서 카메라의 광축이 바라보는 지점의 좌표
                
                # optical center position result display
                if _save_result:
                    op_x = np.round(_cx).astype(int)
                    op_y = np.round(_cy).astype(int)
                    str_image_op_pos = "OC(image): %2.2f,%2.2f"%(_cx, _cy)
                    str_world_op_pos = "OC(world): %2.2f,%2.2f"%(_x_direction*ptVec[0].item(), _y_direction*ptVec[1].item())
                    cv2.putText(undist_raw_color, str_image_op_pos,(op_x+10, op_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(undist_raw_color, str_world_op_pos,(op_x+10, op_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.line(undist_raw_color, (op_x-10,op_y), (op_x+10,op_y), (255,0,255), 1, cv2.LINE_AA) # optical center
                    cv2.line(undist_raw_color, (op_x,op_y-10), (op_x,op_y+10), (255,0,255), 1, cv2.LINE_AA)
                
                # forktip detection
                fork_roi = detect_forktip(weights=_forktip_model, source=export_undist_image, imgsz=_resolution, conf_thres=_forktip_roi_confidence)
                if _save_result:
                    for box in fork_roi:
                        cv2.rectangle(undist_raw_color, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])), (0,255,255), 1)
                        cv2.putText(undist_raw_color, "End-Effector ROI",(int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                '''
                post process for tip point calculation
                '''        
                # padding by custom
                for i, box in enumerate(fork_roi):
                    fork_roi[i][1] = fork_roi[i][1] + _forktip_roi_padding[0] # top
                    fork_roi[i][2] = fork_roi[i][2] - _forktip_roi_padding[1] # right
                    fork_roi[i][3] = fork_roi[i][3] - _forktip_roi_padding[2] # bottom
                    fork_roi[i][0] = fork_roi[i][0] + _forktip_roi_padding[3] # left
                    roi_image = undist_raw_gray[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    roi_image_color = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
                    roi_h, roi_w = roi_image.shape
                    
                    # line segment detection
                    lsd = cv2.createLineSegmentDetector(0)
                    lsd_lines = lsd.detect(roi_image)[0]
                    selected_lsd_lines = []
                    for line in lsd_lines:
                        line_in = line.ravel()
                        p = line_in[0:2]
                        q = line_in[2:4]
                        if math.dist(p, q)>_forktip_line_select:
                            selected_lsd_lines.append(line_in)
                    best_lines = np.array(selected_lsd_lines).reshape((-1, 1, 4))
                    
                    sx, sy = int(round(box[0])), int(round(box[1]))
                    if _save_result:
                        for line in best_lines:
                            ll = [round(pt) for pt in line.ravel()]
                            p1 = (ll[0]+sx, ll[1]+sy)
                            p2 = (ll[2]+sx, ll[3]+sy)
                            cv2.line(undist_raw_color, p1, p2, (0,0,255), 2, cv2.LINE_AA)
                            
                    # selected line feature extension to find intersection point
                    best_lines_extension = []
                    distance = math.dist((0,0), (_w, _h))
                    for line in best_lines:
                        ll = [round(pt) for pt in line.ravel()]
                        p1 = (ll[0]+sx, ll[1]+sy)
                        p2 = (ll[2]+sx, ll[3]+sy)
                        diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
                        new_p1 = (int(p1[0] + distance*np.cos(diff)), int(p1[1] + distance*np.sin(diff)))
                        new_p2 = (int(p1[0] - distance*np.cos(diff)), int(p1[1] - distance*np.sin(diff)))
                        best_lines_extension.append([new_p1, new_p2])
                        
                        if _save_result:
                            cv2.line(undist_raw_color, p1, p2, (0,0,255), 2, cv2.LINE_AA)
                            cv2.line(undist_raw_color, new_p1, new_p2, (255,0,255), 1, cv2.LINE_AA)
                            
                    selected_intersect_pt = []
                    for idx, nl in enumerate(best_lines_extension):
                        step = 1
                        while len(best_lines_extension)-(idx+step)>0:
                            intersect_pt = line_intersection([best_lines_extension[idx][0], best_lines_extension[idx][1]], [best_lines_extension[idx+step][0], best_lines_extension[idx+step][1]])
                            if (intersect_pt[0]>=0 and intersect_pt[0]<=_w) and (intersect_pt[1]>=0 and intersect_pt[1]<=_h):
                                selected_intersect_pt.append(intersect_pt)
                            step +=1
                    
                    # estimate object coord.
                    for p in selected_intersect_pt:
                        pc = np.array([p[0], p[1], 1], dtype=np.float64)
                        wc = np.dot(np.linalg.inv(intrinsic_mtx), pc)
                        xyz_coords = cv2.convertPointsFromHomogeneous(wc.reshape(1, 1, 3))
                        wx = _x_direction*(ptVec[0] - xyz_coords[0, 0, 0]*_laser_distance[fid])
                        wy = _y_direction*(ptVec[1] - xyz_coords[0, 0, 1]*_laser_distance[fid])
                        p_wc = np.array([wx,wy], dtype=np.double).squeeze()
                    
                        if _save_result:
                            cv2.circle(undist_raw_color, p, radius=2, color=(0,0,255), thickness=7)
                            str_image_p_pos = "P(image):%d, %d)"%(p[0], p[1])
                            str_world_u_pos = "P(world):(%2.2f, %2.2f)"%(p_wc[0], p_wc[1])
                            cv2.putText(undist_raw_color, str_image_p_pos,(p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.putText(undist_raw_color, str_world_u_pos,(p[0]+10, p[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            p_dist = math.dist([0,0], p_wc)
                            print("- Distance : ", p_dist)
                            print("P(wc)", p_wc)
                            print("P(px)", p)
                            
                            # center line
                            org_3d = np.array([0.0, 0.0, 0.0], dtype=np.double).reshape(1,-1)
                            org_2d, _ = obj_coord2pixel_coord(org_3d, rVec, tVec, newcamera_mtx, distorsion_mtx)
                            p_center = org_2d.round().astype(int)
                            #cv2.line(undist_raw_color, p, p_center, (255,0,255), 1, cv2.LINE_AA)
                            
                            # outer line
                            if p_dist>_wafer_diameter/2:
                                th_rad = math.atan(p_wc[1]/p_wc[0])
                                print("theta", p_wc, math.degrees(th_rad))
                                p_outer_wc = np.array([_wafer_diameter/2*math.cos(th_rad), _wafer_diameter/2*sin(th_rad), 0], dtype=np.double).reshape(1,-1)
                                p_outer_px, _ = obj_coord2pixel_coord(p_outer_wc, rVec, tVec, newcamera_mtx, distorsion_mtx)
                                p_outer_pts = p_outer_px.round().astype(int)
                                # cv2.line(undist_raw_color, p, p_outer_pts, (0,255,0), 2, cv2.LINE_AA)
                        
    
                # finally save image
                if _save_result:
                    cv2.imwrite(os.path.splitext(image_path)[0]+"_out.png", undist_raw_color)
                    
                    
                
                
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
                p_dic["line_features"] = []
                estimate_result[_image_filenames[fid]] = p_dic
                print(estimate_result)
            else:
                print("Not enough markers are detected")
        
    except json.decoder.JSONDecodeError :
        print("Error : Decoding Job Description has failed")
    except (ValueError, UndefinedParamError) as e:
        print("Error : ",e)
        
                
    # dumps into json result        
    json_result = json.dumps(estimate_result)
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
        estimate(config, job)