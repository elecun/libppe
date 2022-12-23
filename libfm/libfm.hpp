/**
 * @file libfm.hpp
 * @author Byunghun Hwang (bh.hwang@iae.re.kr)
 * @brief M3-FS Focus Module
 * @version 0.1
 * @date 2022-12-08
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _LIB_FM_HPP_
#define _LIB_FM_HPP_


#include <string>

using namespace std;

namespace libfm {

    /**
     * @brief position approximation from working distance
     * 
     * @param wd workind distance (distance between target object and lens)
     * @return double position value of the focus module (Newscale M3-FS)
     */
    double fm_pos_approx(double wd);

    /**
     * @brief focus quality measure with sobel-based method
     * 
     * @param image_path source image file path
     * @param roi_size size of rectangular ROI
     * @return double focus quality value
     */
    double fm_measure(const char* image_path, unsigned int roi_size=300);


} /* namespace */

# endif