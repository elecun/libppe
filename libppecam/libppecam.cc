

#include "libppecam.hpp"
#include "json.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

using namespace nlohmann;

namespace libppecam {

    cv::VideoCapture* _camera = nullptr;

    //for configuration
    static string _config_filename = "";
    json _config;

    /**
     * @brief runner function for loop sequence
     * 
     */
    void _runnner(){

    }

    bool cam_open(){
        return true;
    }

    void cam_close(){

    }

    int set_configure(const char* config_filename){
        
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