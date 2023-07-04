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

def robust_edge_detection(img):
    # Find edges
    kernel_size = 5
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # io.imagesc(blur_gray)
    #edges = cv2.Canny((blur_gray * 255).astype(np.uint8), 10, 200, apertureSize=5)
    edges = canny_auto(blur_gray, sigma=0.33)
    # io.imagesc(edges)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(edges)[0]  # Position 0 of the returned tuple are the detected lines

    long_lines = []
    for j in range(lines.shape[0]):
        x1, y1, x2, y2 = lines[j, 0, :]
        if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) > 50:
            long_lines.append(lines[j, :, :])

    lines = np.array(long_lines)
    edges = 1 * np.ones_like(img)
    drawn_img = lsd.drawSegments(edges, lines)
    edges = (drawn_img[:, :, 2] > 1).astype(np.float32)

    kernel = np.ones((7, 7), np.uint8)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    return edges 
        

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
    return point_3D_camera
        

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
Compute orientation with PCA
'''
def draw_axis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

def calc_orientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 *  eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    draw_axis(img, cntr, p1, (0, 150, 0), 1)
    draw_axis(img, cntr, p2, (200, 150, 0), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    return angle


def canny_auto(image, sigma=0.1):
    v = np.mean(image)
    lower = int(max(0, (1.0-sigma)*v))
    upper = int(min(255, (1.0+sigma)*v))
    edged = cv2.Canny(image, lower, upper)
    return edged

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
    #save_img = not nosave and not source.endswith('.txt')  # save inference images
    #is_file = pathlib.Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    #screenshot = source.lower().startswith('screen')

    # Directories
    save_dir = increment_path(pathlib.Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = torch.device('cpu')
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    result_bb = []
    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / pathlib.Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
        # process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                result_bb.append(det.numpy().ravel())
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")        
    return result_bb

'''
 Undefined Parameter key Exception
'''
class UndefinedParamError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
    


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
        _working_path = WORKING_PATH if "path" not in process_job else WORKING_PATH.joinpath(process_job["path"])
        _resolution = [1280, 960] if "resolution" not in process_param else process_param["resolution"]
        _w, _h = _resolution
        _camera_parameter = [float(process_param[key]) for key in ("fx","fy", "cx", "cy", "coeff_k1", "coeff_k2", "coeff_p1", "coeff_p2")]
        _fx, _fy, _cx, _cy, _k1, _k2, _p1, _p2 = _camera_parameter
        _wafer_diameter = 300.0 if "diameter" not in process_param["wafer"] else float(process_param["wafer"]["diameter"])
        if "coord" in process_param["marker"]:
            _wafer_marker_pos = json.loads(json.dumps(process_param["marker"]["coord"]))
            _wafer_marker_pos = {int(k):[int(i)+_wafer_diameter/2 for i in v] for k,v in _wafer_marker_pos.items()} # changed origin
            
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
            print("(%d) %s is now processing..."%fid,image_path) if _verbose else None
            
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
            
            # save result
            if _save_result:
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
                    marker_centroids_on_image.append(np.mean(corner, axis=0, dtype=float))
                    marker_centroids_on_wafer.append(_wafer_marker_pos[marker_id])
                marker_centroids_on_image = np.array(marker_centroids_on_image)
                marker_centroids_on_wafer = np.array(marker_centroids_on_wafer)
                marker_centroids_on_wafer = np.append(marker_centroids_on_wafer, np.zeros(shape=(np.size(marker_centroids_on_wafer, axis=0), 1), dtype=np.double),axis=1) # column add
                
                if marker_centroids_on_image.shape[0] != marker_centroids_on_wafer.shape[0]:
                    raise ValueError("Marker pointset dimension is not same")
                
                # compute 2D-3D corredpondence with Perspective N Point'
                # note] use undistorted image points to solve, then appply the distortion coefficient and camera matrix as pin hole camera model
                _, rVec, tVec = cv2.solvePnP(marker_centroids_on_wafer, marker_centroids_on_image, cameraMatrix=newcamera_mtx, distCoeffs=distorsion_mtx, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP)
                _, prVec, ptVec = cv2.solvePnP(marker_centroids_on_wafer, marker_centroids_on_image, cameraMatrix=intrinsic_mtx, distCoeffs=None, rvec=None, tvec=None, useExtrinsicGuess=None, flags=cv2.SOLVEPNP_SQPNP) #pinhole camera model
                R, jacobian = cv2.Rodrigues(rVec) # rotation vector to matrix
            
            
        
        
        
        
        for fid, filename in enumerate(_image_filenames):
            src_image = str(_working_path / pathlib.Path(filename))
            
            src_image_undist = os.path.splitext(src_image)[0]+"_undist.png"

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
            undist_raw_color = cv2.undistort(raw_color, intrinsic_mtx, distorsion_mtx, None, newcamera_mtx) # undistortion by camera parameters
            undist_raw_gray_inv = cv2.bitwise_not(undist_raw_gray) # color inversion
            undist_color_result = undist_raw_color.copy()# cv2.cvtColor(undist_raw_color, cv2.COLOR_GRAY2BGR) # for draw results
            
            # save undistorted image for roi detection
            cv2.imwrite(src_image_undist, undist_raw_gray)
            
            # find markers printed on reference wafer
            if _python_version[0]==4 and _python_version[1]<7:
                corners, ids, rejected = cv2.aruco.detectMarkers(undist_raw_gray_inv, markerdict, parameters=markerparams)
            else:
                corners, ids, rejected = markerdetector.detectMarkers(undist_raw_gray_inv)
                
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
                # if _save_result:
                #     for idx, pts in enumerate(marker_centroids_on_image):
                #         p = tuple(pts.round().astype(int))
                        
                #         str_image_pos = "on image : [%d] (%2.2f,%2.2f)"%(ids[idx].item(), pts[0], pts[1])
                #         str_world_pos = "on wafer : (%2.2f,%2.2f)"%(marker_centroids_on_wafer[idx][0], marker_centroids_on_wafer[idx][1])
                #         if _verbose:
                #             print("marker :",str_image_pos, str_world_pos)
                #         cv2.putText(undist_color_result, str_image_pos,(p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                #         cv2.putText(undist_color_result, str_world_pos,(p[0]+10, p[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                #         cv2.line(undist_color_result, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,255,0), 1, cv2.LINE_AA)
                #         cv2.line(undist_color_result, (p[0],p[1]-10), (p[0],p[1]+10), (0,255,0), 1, cv2.LINE_AA)
                
            
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
                        
                        cv2.putText(undist_color_result, str_image_pos,(p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(undist_color_result, str_world_pos,(p[0]+10, p[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.line(undist_color_result, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,255,0), 1, cv2.LINE_AA)
                        cv2.line(undist_color_result, (p[0],p[1]-10), (p[0],p[1]+10), (0,255,0), 1, cv2.LINE_AA)
                        
                        op_x = np.round(_cx).astype(int)
                        op_y = np.round(_cy).astype(int)
                        str_image_op_pos = "OC_img : x=%2.2f,y=%2.2f"%(_cx, _cy)
                        str_world_op_pos = "OC_real : x=%2.2f,y=%2.2f"%(_x_direction*ptVec[0].item(), _y_direction*ptVec[1].item())
                        cv2.putText(undist_color_result, str_image_op_pos,(op_x+10, op_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(undist_color_result, str_world_op_pos,(op_x+10, op_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.line(undist_color_result, (op_x-10,op_y), (op_x+10,op_y), (255,0,255), 1, cv2.LINE_AA) # optical center
                        cv2.line(undist_color_result, (op_x,op_y-10), (op_x,op_y+10), (255,0,255), 1, cv2.LINE_AA)
                    

                # forktip detection ([upper-left(x,y), bottom-right(x,y), confidence, 0])
                roi = detect_forktip(weights=_forktip_model, source=src_image_undist, imgsz=_resolution, conf_thres=_forktip_roi_confidence)
                if _save_result:
                    for bb in roi:
                        cv2.rectangle(undist_color_result, (int(bb[0]), int(bb[1])),(int(bb[2]), int(bb[3])), (0,255,0), 1)
                        cv2.putText(undist_color_result, "End-Effector ROI",(int(bb[0]), int(bb[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                
                # crop image with ROI, remove unbound area
                unbound_y = 0
                if int(bb[3])-int(bb[1])>10:
                    unbound_y = 10
                roi_image = undist_raw_gray[int(bb[1]):int(bb[3])-unbound_y, int(bb[0]):int(bb[2])]
                roi_image_color = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
                roi_h, roi_w = roi_image.shape
                
                
                # line detection
                roi_image_edge = canny_auto(roi_image, sigma=0.33)
                # roi_image_edge = cv2.Canny(roi_image, 10, 100, apertureSize=3)
                
                lsd = cv2.createLineSegmentDetector(0)
                lsd_lines = lsd.detect(roi_image)[0] #Position 0 of the returned tuple are the detected lines
                
                lsd_feature = []
                for d in lsd_lines:
                    d_in = d.ravel()
                    p = d_in[0:2]
                    q = d_in[2:4]
                    if math.dist(p, q)>100:
                        lsd_feature.append(d_in)
                
                
                t = np.array(lsd_feature).reshape((-1, 1, 4))
                
                
                new_lines = []
                if len(roi):
                    bb = roi[0]
                    sx, sy = int(round(bb[0])), int(round(bb[1]))
                    for l in t:
                        ll = [round(pt) for pt in l.ravel()]
                        p1 = (ll[0]+sx, ll[1]+sy)
                        p2 = (ll[2]+sx, ll[3]+sy)
                        cv2.line(undist_color_result, p1, p2, (0,0,255), 2, cv2.LINE_AA)
                        
                        distance = math.dist((0,0), (_w, _h))
                        diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
                        p3_x = int(p1[0] + distance*np.cos(diff))
                        p3_y = int(p1[1] + distance*np.sin(diff))
                        p4_x = int(p1[0] - distance*np.cos(diff))
                        p4_y = int(p1[1] - distance*np.sin(diff))
                        
                        new_p1 = (p3_x, p3_y) 
                        new_p2 = (p4_x, p4_y)
                        new_lines.append([new_p1, new_p2])
                        
                        # draw extended line
                        cv2.line(undist_color_result, new_p1, new_p2, (255,0,255), 1, cv2.LINE_AA)
                    
                for idx, nl in enumerate(new_lines):
                    step = 1
                    while len(new_lines)-(idx+step)>0:
                        intersect_pt = line_intersection([new_lines[idx][0], new_lines[idx][1]], [new_lines[idx+step][0], new_lines[idx+step][1]])
                        
                        # check inbound
                        if (intersect_pt[0]>=0 and intersect_pt[0]<=_w) and (intersect_pt[1]>=0 and intersect_pt[1]<=_h):
                            cv2.circle(undist_color_result, intersect_pt, radius=2, color=(0,0,255), thickness=5)
                            print(intersect_pt)
                            
                            str_image_is_pos = "IS_img : (%2.2f, %2.2f)"%(intersect_pt[0], intersect_pt[1])
                            wc = pixel_coord2obj_coord(intersect_pt, prVec, ptVec, intrinsic_mtx, None, verbose=False)
                            str_world_is_pos = "IS_real : (%2.2f, %2.2f)"%(_x_direction*wc[0], _y_direction*wc[1])
                            cv2.putText(undist_color_result, str_image_is_pos,(intersect_pt[0]+10, intersect_pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.putText(undist_color_result, str_world_is_pos,(intersect_pt[0]+10, intersect_pt[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                        
                        step += 1
                        
                        
                    
                
                
                
                
                # finally save image
                if _save_result:
                    cv2.imwrite(str(_working_path / pathlib.Path("out_"+filename)), undist_color_result)
                    
                

                
                
                cv2.imwrite("test_roi_edge.png", roi_image_edge)
                rho, theta, thresh = 1, np.pi/1800, 40
                lines = cv2.HoughLines(roi_image_edge, rho, theta, thresh)
                #lines = cv2.HoughLinesP(roi_image_edge, rho, theta, thresh, minLineLength=5, maxLineGap=10)
                print(lines.shape, type(lines))
                for line in lines:
                    r,theta = line[0]
                    if theta<100*np.pi/180:
                        tx, ty = np.cos(theta), np.sin(theta)
                        #print("R : ", r, ", Theta :", theta*180/np.pi)
                        x0, y0 = tx*r, ty*r
                        #cv2.circle(roi_image_show, (abs(x0), abs(y0)), 3, (0,0,255), -1)
                        x1, y1 = int(x0 + roi_w*(-ty)), int(y0 + roi_h * tx)
                        x2, y2 = int(x0 - roi_w*(-ty)), int(y0 - roi_h * tx)
                        cv2.line(roi_image_color, (x1, y1), (x2, y2), (0,255,0), 1)

                cv2.imwrite("test_roi.png", roi_image_color)
                        
                
                
     
                
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
                estimate_result[filename] = p_dic
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