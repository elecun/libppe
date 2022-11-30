/**
 * @file libppecam.hpp
 * @author Byunghun Hwang (bh.hwang@iae.re.kr)
 * @brief Camera device interface library for libppe
 * @version 0.1
 * @date 2022-11-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _LIB_PPECAM_HPP_
#define _LIB_PPECAM_HPP_


#include <string>

using namespace std;

namespace libppecam {    

    bool cam_open();
    void cam_close();

    int set_configure(const char* config_filename);

    string cam_trigger_on(double time_limit_ms);
    void cam_trigger_off();
    bool is_cam_triggered();


} /* namespace */

# endif