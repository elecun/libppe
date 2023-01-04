/**
 * @file util.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _LIBPPECAM_UTIL_HPP_
#define _LIBPPECAM_UTIL_HPP_

#include <sys/sysinfo.h>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <sys/stat.h>
#include <unistd.h>


namespace util {
/**
 * @brief file existance check
 * 
 * @param filepath filepath to find
 * @return true if exist
 * @return false if it does not exist
 */
inline bool exist(const char* filepath) {
    struct stat buffer;   
    return (stat(filepath, &buffer) == 0); 
}

} //namespace util

#endif