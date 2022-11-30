
#include "ov2311_uc762c.hpp"
#include <core/iController.hpp>
#include <spdlog/spdlog.h>

#include <thread>
#include <time.h>
#include <iostream>
#include <istream>
#include <string>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <chrono>

using namespace std;

namespace ppe::cmos {

    ov2311_uc762c::ov2311_uc762c(int camera_id):_camera_id(camera_id) {
        camera_matrix = (cv::Mat1d(3,3) << _FX_, 0., _CX_, 0., _FY_, _CY_, 0., 0., 1.);
        distortion_coeff = (cv::Mat1d(1,5) << _K1_, _K2_, 0, _P1_, _P2_);
        this->width = 1600;
        this->height = 1200;
    }
    
    ov2311_uc762c::~ov2311_uc762c(){
        this->close();
    }

    bool ov2311_uc762c::open(){
        _camera = new cv::VideoCapture(_camera_id, cv::CAP_V4L2);
        if(_camera->isOpened()){
            _camera->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            _camera->set(cv::CAP_PROP_FRAME_WIDTH, this->width);
            _camera->set(cv::CAP_PROP_FRAME_HEIGHT, this->height);
            _camera->set(cv::CAP_PROP_FPS, 30);
            return true;
        }
        return false;
    }

    void ov2311_uc762c::close(){
        if(_camera)
            _camera->release();
    }

    bool ov2311_uc762c::is_valid(){
        return true;
    }
  

    cv::Mat ov2311_uc762c::capture(){
        if(_camera){
            if(_camera->grab()){
                cv::Mat raw;
                _camera->retrieve(raw);
                cv::cvtColor(raw, raw, cv::COLOR_BGR2GRAY);
                return raw;
            }
        }

        return cv::Mat();
    }


    

}