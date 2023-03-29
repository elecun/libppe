

#include "libppe.hpp"
#include <string>
#include "json.hpp"
#include <fstream>
#include <iostream>
#include "util.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>

#define _DEVICE_ "device"

using namespace std;
using namespace nlohmann;
using namespace cv;

namespace libppe {

    /**
     * @brief internal variables
     * 
     */
    static string _config_filename = "";
    json _config;
    cv::Mat _camera_matrix = cv::Mat::eye(3, 3,CV_64FC1);
    cv::Mat _distortion_coeff = cv::Mat::eye(1, 5,CV_64FC1);

    int set_configure(const char* config_filename){
        _config.clear();
        
        try {
            if(!util::exist(config_filename)){
                cout << "Configuration file does not exist" << endl;
                return 1;
            }

            std::ifstream file(config_filename);
            file >> _config;

            //load configurations
            if(_config.find(_DEVICE_)!=_config.end()){
                json _camera_w_param = _config["device"]["camera-w"];
                cout << "camera-w vid : " << _camera_w_param["vid"].get<int>() << endl;

                // read parameters
                double _cam_fx = _camera_w_param["fx"].get<double>();
                double _cam_fy = _camera_w_param["fy"].get<double>();
                double _cam_cx = _camera_w_param["cx"].get<double>();
                double _cam_cy = _camera_w_param["cy"].get<double>();
                double _cam_k1 = _camera_w_param["coeff_k1"].get<double>();
                double _cam_k2 = _camera_w_param["coeff_k2"].get<double>();
                double _cam_p1 = _camera_w_param["coeff_p1"].get<double>();
                double _cam_p2 = _camera_w_param["coeff_p2"].get<double>();

                //setting parameters
                _camera_matrix = (cv::Mat1d(3,3) << _cam_fx, 0., _cam_cx, 0., _cam_fy, _cam_cy, 0., 0., 1.);
                _distortion_coeff = (cv::Mat1d(1,5) << _cam_k1, _cam_k2, 0, _cam_p1, _cam_p2);
            }
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

    pos6d estimate_wafer_s(const char* image_file){
        pos6d _result;

        //1. load image file calibrated
        if(!util::exist(image_file)){
            cout << "Could not find the file " << image_file << endl;
            return _result;
        }

        cv::Mat image = cv::imread(image_file);
        cv::Mat out = image.clone();
        cv::Mat grayscale;
        if(image.channels()!=1){
            cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
        }

        cv::Ptr<cv::aruco::Dictionary> dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        vector<vector<cv::Point2f>> marker_corners, rejected_candidates;
        vector<int> marker_ids;
        cv::aruco::detectMarkers(grayscale, dict, marker_corners, marker_ids, parameters, rejected_candidates);

        if(marker_ids.size()>0) {
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(marker_corners, 0.04, _camera_matrix, _distortion_coeff, rvecs, tvecs);

            //3.2. draw markers
            for(unsigned int i=0; i<marker_ids.size(); i++){
                cv::aruco::drawAxis(out, _camera_matrix, _distortion_coeff, rvecs[i], tvecs[i], 0.01);
                cout << "id : " << marker_ids[i] << endl; 
                cout << "tvec : " << tvecs[i][0] << "\t" << tvecs[i][1] << "\t" << tvecs[i][2] << endl;
                cout << "rvec : " << rvecs[i][0] << "\t" << rvecs[i][1] << "\t" << rvecs[i][2] << endl;
            }
            //cv::imwrite("marker_out.png", out);
        }

        return _result;
    }

    pos6d estimate_wafer_s(const cv::Mat image){
        pos6d _result;
        if(image.empty()){
            cout << "image has no data" << endl;
            return _result;
        }

        cv::Mat out = image.clone();
        cv::Mat grayscale;
        if(image.channels()!=1){
            cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
        }

        cv::Ptr<cv::aruco::Dictionary> dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        vector<vector<cv::Point2f>> marker_corners, rejected_candidates;
        vector<int> marker_ids;
        cv::aruco::detectMarkers(grayscale, dict, marker_corners, marker_ids, parameters, rejected_candidates);

        if(marker_ids.size()>0) {
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(marker_corners, 0.04, _camera_matrix, _distortion_coeff, rvecs, tvecs);

            //3.2. draw markers
            for(unsigned int i=0; i<marker_ids.size(); i++){
                cv::aruco::drawAxis(out, _camera_matrix, _distortion_coeff, rvecs[i], tvecs[i], 0.01);
                cout << "id : " << marker_ids[i] << endl; 
                cout << "tvec : " << tvecs[i][0] << "\t" << tvecs[i][1] << "\t" << tvecs[i][2] << endl;
                cout << "rvec : " << rvecs[i][0] << "\t" << rvecs[i][1] << "\t" << rvecs[i][2] << endl;
            }
            //cv::imwrite("marker_out.png", out);
        }

        return _result;
    }

    vector<pos6d> estimate_wafer_m(const vector<string> image_files){
        vector<pos6d> _results;

        for(auto& image_file : image_files){
            pos6d result = estimate_wafer_s(image_file.c_str());
        }

        return _results;
    }

    vector<pos6d> estimate_wafer_m(const vector<cv::Mat> images){
        vector<pos6d> _results;
        for(auto& image : images){
            pos6d result = estimate_wafer_s(image);
        }

        return _results;
    }

    //newly added @ 23.03.29
    json _param;
    string estimate(const cahr* source_image){

    }

    bool set_parameters(string parameters){
        try {
            json param = json::parse(parameters);
        }
        catch (json::parse_error& ex){
            std::cerr << "parse error at byte " << ex.byte << std::endl;
            std::cerr << "what : " << ex.what() << std::endl;
        }
        return false;
    }

}