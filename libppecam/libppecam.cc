

#include "libppecam.hpp"
#include "json.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include "util.hpp"

#define _DEVICE_ "device"

using namespace nlohmann;
using namespace std;

namespace libppecam {

    cv::VideoCapture* _camera_w = nullptr;
    cv::VideoCapture* _camera_e = nullptr;

    //for configuration
    static string _config_filename = "";
    json _config;
    cv::Mat _camera_matrix = cv::Mat::eye(3, 3,CV_64FC1);
    cv::Mat _distortion_coeff = cv::Mat::eye(1, 5,CV_64FC1);
    int _cam_width = 0;
    int _cam_height = 0;
    int _cam_w_vid = -1;


    /**
     * @brief runner function for loop sequence
     * 
     */
    void _runnner(){

        if(_camera_w){
            if(_camera_w->grab()){
                cv::Mat raw;
                _camera_w->retrieve(raw);
            }
        }

    }

    bool cam_open(){
        // read required parameters for camera_w
        if(_config.find(_DEVICE_)!=_config.end()){
            json _camera_w_param = _config["device"]["camera-w"];
            cout << "camera-w vid : " << _camera_w_param["vid"].get<int>() << endl;

            // read parameters
            _cam_w_vid = _camera_w_param["vid"].get<int>();
            double _cam_fx = _camera_w_param["fx"].get<double>();
            double _cam_fy = _camera_w_param["fy"].get<double>();
            double _cam_cx = _camera_w_param["cx"].get<double>();
            double _cam_cy = _camera_w_param["cy"].get<double>();
            double _cam_k1 = _camera_w_param["coeff_k1"].get<double>();
            double _cam_k2 = _camera_w_param["coeff_k2"].get<double>();
            double _cam_p1 = _camera_w_param["coeff_p1"].get<double>();
            double _cam_p2 = _camera_w_param["coeff_p2"].get<double>();
            vector<int> _resolution = _camera_w_param["resolution"].get<vector<int>>();
            cout << "camera resolution : " << _resolution[0] << "," << _resolution[1] << endl;
            int _cam_fps = _camera_w_param["fps"].get<int>();
            cout << "camera-w fps set : " << _cam_fps << endl;

            //setting parameters
            _camera_matrix = (cv::Mat1d(3,3) << _cam_fx, 0., _cam_cx, 0., _cam_fy, _cam_cy, 0., 0., 1.);
            _distortion_coeff = (cv::Mat1d(1,5) << _cam_k1, _cam_k2, 0, _cam_p1, _cam_p2);
            _cam_width = _resolution[0];
            _cam_height = _resolution[1];

            //camera device open
            _camera_w = new cv::VideoCapture(_cam_w_vid, cv::CAP_V4L2);
            if(_camera_w->isOpened()){
                _camera_w->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                _camera_w->set(cv::CAP_PROP_FRAME_WIDTH, _cam_width);
                _camera_w->set(cv::CAP_PROP_FRAME_HEIGHT, _cam_height);
                _camera_w->set(cv::CAP_PROP_FPS, _cam_fps);

                cout << "camera successfully opened : " << _cam_w_vid << endl;

                // grab sample image
                for(int i=0;i<50;i++){
                    auto total_start = chrono::steady_clock::now();
                    vector<cv::Mat> raw_container;
                    raw_container.reserve(50);
                    cv::Mat raw;
                    if(_camera_w->grab()){
                        _camera_w->retrieve(raw);
                        raw_container.emplace_back(raw);
                        //cv::imwrite("./cam_w.png", raw);
                    }
                    auto total_end = chrono::steady_clock::now();
                    float total_fps = 1000.0 / chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count();

                    cout << "FPS : " << total_fps << endl;
                }
            }
            else {
                cout << "camera cannot be opened. vid : " << _cam_w_vid << endl;
            }

        }

        return true;
    }

    void cam_close(){
        if(_camera_w){
            _camera_w->release();
            cout << "camera successfully closed : " << _cam_w_vid << endl;
        }
    }

    int set_configure(const char* config_filename){
        _config.clear();
        
        try {
            if(!util::exist(config_filename)){
                cout << "Configuration file does not exist" << endl;
                return 1;
            }

            std::ifstream file(config_filename);
            file >> _config;
        }
        catch(const json::exception& e){
            cout << "Configuration file load failed : " << e.what() << endl;
            return 3;
        }
        catch(std::ifstream::failure& e){
            cout << "Configuration file load failed : " << e.what() << endl;
            return 2;
        }

        return 0;
    }

    string cam_trigger_on(double time_limit_ms){
        return string("");
    }

    void cam_trigger_off(){

    }

    bool is_cam_triggered(){
        return false;
    }

}