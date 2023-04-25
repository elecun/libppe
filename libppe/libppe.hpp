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
#include <opencv2/core/mat.hpp>
#include <exception>

using namespace std;

namespace libppe {
    
    // newly added interface function (23.03.29)
    /**
     * @brief estimate the position and orientation both of effector and wafer
     * @param job_desc image processing job descriptions(parameters)
     * @return string dumped from JSON
     */
    string estimate(string job_desc);
    string estimate(string job_param, vector<cv::Mat> images);

    /**
     * @brief set parameters for estimating position/orientation
     * @param parameters predefined parameters (JSON format)
     * @return true success, else false
     */
    bool set_parameters(string parameterset);

    /**
     * @brief library initialize
     */
    void initialize();

    /**
     * @brief release and terminate the dynamic objects
     * 
     */
    void finalize();

} /* namespace */

# endif