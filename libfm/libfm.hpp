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


} /* namespace */

# endif