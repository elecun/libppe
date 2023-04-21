

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
#define PY_SSIZE_T_CLEAN
#if (__PYTHON_VER_MAJOR__==3)
    #if (_PYTHON_VER_MINOR__==8)
        #include <python3.8/Python.h>
    #elif (__PYTHON_VER_MINOR__==9)
        #include <python3.9/Python.h>
    #elif (__PYTHON_VER_MINOR__==10)
        #include <python3.10/Python.h>
    #endif
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

        if(_param.empty()){
            throw std::runtime_error("No camera parameters. please, set a parameter by set_parameters");
            return "{}";
        }
            

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
            PyObject* _py_func_args = PyTuple_New(2);
            string p = _param.dump();
            PyTuple_SetItem(_py_func_args, 0, PyUnicode_FromString(p.c_str()));
            PyTuple_SetItem(_py_func_args, 1, PyUnicode_FromString(job_desc.c_str()));
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