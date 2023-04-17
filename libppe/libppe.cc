

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

/* python bindings */
#include <python3.9/Python.h>


using namespace std;
using namespace nlohmann;
using namespace cv;


namespace libppe {
    /* internal variables */
    json _param;

    /* pose estimation interface function */
    string estimate(string job_desc){
        json _result;
        try {
            // job desc check & parse
            json desc = json::parse(job_desc);
            if(desc.contains("use_parameters") && desc.contains("path") && desc.contains("files")){
                string _param_name = desc["use_parameters"].get<string>();
                string _path = desc["path"].get<string>();
                json _files = desc["files"];
                vector<string> _image_files;

                //getting image list from job desc.
                for(json::iterator itr = _files.begin(); itr != _files.end(); ++itr){
                    if(itr.value().is_string()){
                        _image_files.push_back(itr.value());
                    }   
                }

                for(auto image_file : _image_files){
                    cv::Mat image = cv::imread(_path+image_file, 0);
                    //processing for wafer position estimation
                    //code here
                    _result[image_file]["wafer_x"] = 0.0;
                    _result[image_file]["wafer_y"] = 0.0;
                    _result[image_file]["wafer_z"] = 0.0;
                    _result[image_file]["wafer_r"] = 0.0;
                    _result[image_file]["wafer_p"] = 0.0;
                    _result[image_file]["wafer_w"] = 0.0;

                    //processing for effector position estimation
                    //code here
                    _result[image_file]["effector_x"] = 0.0;
                    _result[image_file]["effector_y"] = 0.0;
                    _result[image_file]["effector_z"] = 0.0;
                    _result[image_file]["effector_r"] = 0.0;
                    _result[image_file]["effector_p"] = 0.0;
                    _result[image_file]["effector_w"] = 0.0;

                    //processing for additional geometrical calculation
                    _result[image_file]["distance"] = 0.0;
                }

            }
            else{
                throw std::runtime_error("some parameters are missing");
            }

        }
        catch(json::parse_error& e){
            throw std::runtime_error(e.what());
        }

        return _result.dump();
    }

    string estimate(string job_param, vector<cv::Mat> images){
        json _result;
        try {

        }
        catch(json::parse_error& e){
            throw std::runtime_error(e.what());
        }

        return _result.dump();
    }

    /* setting up parameters for image processing */
    bool set_parameters(string parameterset){
        try {
            if(!_param.empty())
                _param.clear();

            _param = json::parse(parameterset);

            return true;
        }
        catch (json::parse_error& e){
            throw std::runtime_error(e.what());
        }
        return false;
    }

}