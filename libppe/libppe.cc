

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

#define _PYTHON_3_8_
#define PY_SSIZE_T_CLEAN
#if defined(_PYTHON_3_10_)
    #include <python3.10/Python.h>
#elif defined(_PYTHON_3_9_)
    #include <python3.9/Python.h>
#elif defined(_PYTHON_3_8_)
    #include <python3.8/Python.h>
#endif

#define PYTHON_MDOULE_NAME  "ppe_module"

using namespace std;
using namespace nlohmann;
using namespace cv;


namespace libppe {
    /* internal variables */
    json _param;
    string _result = "{}";

    /* pose estimation interface function */
    string estimate(string job_desc){

        _result.empty(); //clear

        try {
            // python initialize
            Py_Initialize();
            PyRun_SimpleString("import sys; sys.path.append('.')");

            //python module import
            PyObject* _py_module = PyImport_ImportModule(PYTHON_MDOULE_NAME);
            if(_py_module==nullptr){
                PyErr_Print();
                throw std::runtime_error("Python Module import failed");
            }

            //interfacing python function
            PyObject* _py_func_estimate = PyObject_GetAttrString(_py_module, "estimate");
            if(_py_func_estimate==nullptr){
                PyErr_Print();
                Py_DECREF(_py_module);
                throw std::runtime_error("It cannot be found estimate function in python module");
            }

            //json format string arguments
            PyObject* _py_func_args = PyTuple_New(1);
            PyTuple_SetItem(_py_func_args, 0, PyUnicode_FromString(job_desc.c_str()));
            PyObject* _py_result = PyObject_CallObject(_py_func_estimate, _py_func_args);

            if(_py_result==nullptr){
                PyErr_Print();
                Py_DECREF(_py_func_estimate);
                Py_DECREF(_py_module);
                Py_DECREF(_py_func_args);
            }
            else{

                //success, return string
                _result = PyUnicode_AsUTF8(_py_result);

                //termination python objects
                Py_DECREF(_py_result);
                Py_DECREF(_py_func_estimate);
                Py_DECREF(_py_module);
                Py_DECREF(_py_func_args);

                //finalized python module
                Py_Finalize();
            }

        }
        catch(json::parse_error& e){
            throw std::runtime_error(e.what());
        }

        return _result;


        // //job desc check & parse
        // json desc = json::parse(job_desc);
        // if(desc.contains("use_parameters") && desc.contains("path") && desc.contains("files")){
        //     string _param_name = desc["use_parameters"].get<string>();
        //     string _path = desc["path"].get<string>();
        //     json _files = desc["files"];
        //     vector<string> _image_files;

        //     //getting image list from job desc.
        //     for(json::iterator itr = _files.begin(); itr != _files.end(); ++itr){
        //         if(itr.value().is_string()){
        //             _image_files.push_back(itr.value());
        //         }   
        //     }

        //     for(auto image_file : _image_files){
        //         cv::Mat image = cv::imread(_path+image_file, 0);
        //         //processing for wafer position estimation
        //         //code here
        //         _result[image_file]["wafer_x"] = 0.0;
        //         _result[image_file]["wafer_y"] = 0.0;
        //         _result[image_file]["wafer_z"] = 0.0;
        //         _result[image_file]["wafer_r"] = 0.0;
        //         _result[image_file]["wafer_p"] = 0.0;
        //         _result[image_file]["wafer_w"] = 0.0;

        //         //processing for effector position estimation
        //         //code here
        //         _result[image_file]["effector_x"] = 0.0;
        //         _result[image_file]["effector_y"] = 0.0;
        //         _result[image_file]["effector_z"] = 0.0;
        //         _result[image_file]["effector_r"] = 0.0;
        //         _result[image_file]["effector_p"] = 0.0;
        //         _result[image_file]["effector_w"] = 0.0;

        //         //processing for additional geometrical calculation
        //         _result[image_file]["distance"] = 0.0;
        //     }

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