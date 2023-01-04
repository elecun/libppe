/**
 * @file libppe.hpp
 * @author Byunghun Hwang (bh.hwang@iae.re.kr)
 * @brief Precision Pose Estimation Algorithm & Interface Library
 * @version 0.1
 * @date 2022-11-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _LIB_PPE_HPP_
#define _LIB_PPE_HPP_

#include <vector>
#include <string>
#include "json.hpp"
#include <opencv2/core/mat.hpp>

using namespace std;
using namespace nlohmann;

namespace libppe {
    /**
     * @brief type definitions
     * 
     */
    #ifndef _pos2d
    typedef struct _pos2d {
        double x,y;
        _pos2d():x(0.),y(0.){ }
        void reset(){x=y=0.; }
        _pos2d operator= (const _pos2d &arg) { x=arg.x; y=arg.y; }
    } pos2d;
    #endif

    #ifndef _pos3d
    typedef struct _pos3d {
        double x,y,z;
        _pos3d():x(0.),y(0.),z(0.){ }
        void reset(){x=y=z=0.; }
        _pos3d operator= (const _pos3d &arg) { x=arg.x; y=arg.y; z=arg.z; }
    } pos3d;
    #endif

    #ifndef _pos6d 
    typedef struct _pos6d {
        double x,y,z;
        double R,P,Y;
        _pos6d():x(0.),y(0.),z(0.),R(0.),P(0.),Y(0.){ }
        void reset(){x=y=z=R=P=Y=0.; }
        _pos6d operator= (const _pos6d &arg) { x=arg.x; y=arg.y; z=arg.z; R=arg.R; P=arg.P; Y=arg.Y; }
    } pos6d;
    #endif

    #ifndef _wafer_param
    typedef struct _wafer_param {
        double diameter_inch = 12.0;
    } wafer_parameter;
    #endif

    #ifndef _camera_param
    typedef struct _camera_param {
        double fx, fy;
        double cx, cy;
        double dist_coeff[5] = {0.0, };
    } camera_param;
    #endif

    /**
     * @brief pre-defined parameters
     * 
     */
    #define _CAM_ID_0   0   //Camera ID 0 : Wafer viewpoint
    #define _CAM_ID_1   1   //Camera ID 1 : Fork viewpoint  

    
    int set_configure(const char* config_filename);

    /**
     * @brief estimate the wafer pose from source directory
     * 
     * @param source_directory source directory path
     * @return std::vector<pair<double, pos6d>> coordinates
     */
    vector<pair<double, pos6d>> estimate_wafer_dir(const char* source_directory);

    vector<pos6d> estimate_wafer_m(const vector<string> image_files);
    vector<pos6d> estimate_wafer_m(const vector<cv::Mat> images);

    pos6d estimate_wafer_s(const char* image_file);
    pos6d estimate_wafer_s(const cv::Mat image);

    /**
     * @brief estimate the effector pose from source directory
     * 
     * @param source_series 
     * @return std::vector<pair<double, pos6d>> 
     */
    std::vector<pair<double, pos6d>> estimate_pos6d_effector(const char* source_series);

} /* namespace */

# endif