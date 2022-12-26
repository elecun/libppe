

#include "libfm.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <csignal>
#include <ctime>
#include <cxxopts.hpp>
#include <string>
#include <sys/mman.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
namespace console = spdlog;


/* handling for signal event */
void cleanup(int sig) {
    switch(sig)
    {
        case SIGSEGV: { console::warn("Segmentation violation"); } break;
        case SIGABRT: { console::warn("Abnormal termination"); } break;
        case SIGKILL: { console::warn("Process killed"); } break;
        case SIGBUS: { console::warn("Bus Error"); } break;
        case SIGTERM: { console::warn("Termination requested"); } break;
        case SIGINT: { console::warn("interrupted"); } break;
        default:
        console::info("Cleaning up the program");
    }
        
    console::info("Successfully terminated");
    exit(EXIT_SUCCESS);
}

/* signal setting into main thread */
void signal_set(){
    const int signals[] = { SIGINT, SIGTERM, SIGBUS, SIGKILL, SIGABRT, SIGSEGV };

    for(const int& s:signals)
        signal(s, cleanup);

    //signal masking
    sigset_t sigmask;
    if(!sigfillset(&sigmask)){
        for(int signal:signals)
            sigdelset(&sigmask, signal); //delete signal from mask
    }
    else {
        console::error("Signal Handling Error");
        cleanup(0);
    }

    if(pthread_sigmask(SIG_SETMASK, &sigmask, nullptr)!=0){ // signal masking for this thread(main)
        console::error("Signal Masking Error");
        cleanup(0);
    }
}

int main(int argc, char** argv){

    cxxopts::Options options("TEST program for libfm library");
    options.add_options()
        ("w,wd", "Working Distance", cxxopts::value<double>())
        ("i,image", "Image file path", cxxopts::value<string>())
        ("d,device", "Device(Camera) ID", cxxopts::value<int>())
        ("r,roi", "ROI size for Focus Measure", cxxopts::value<int>())
        ("h,help", "Print usage");

    auto optval = options.parse(argc, argv);
    if(optval.count("help")){
        std::cout << options.help() << std::endl;
        exit(EXIT_SUCCESS);
    }

    console::stdout_color_st("console");
    signal_set();

    mlockall(MCL_CURRENT|MCL_FUTURE); //avoid memory swaping

    console::info("Ver. {}.{}.{} (built {}/{})", __MAJOR__, __MINOR__, __REV__, __DATE__, __TIME__);

    double _wd = 0.;
    int _roi_size = 0;
    string _image_file = "";
    int _camid = -1;

    if(optval.count("wd")){
        _wd = optval["wd"].as<double>();
    }
    if(optval.count("roi")){
        _roi_size = optval["roi"].as<int>();
    }
    if(optval.count("image")){
        _image_file = optval["image"].as<string>();
    }
    if(optval.count("device")){
        _camid = optval["device"].as<int>();
    }


    try{
        if(_wd!=0){
            double approx_pos = libfm::fm_pos_approx(_wd);
            console::info("Appoximated Focus Position : {}", approx_pos);
        }
        
        if(_roi_size!=0 && !_image_file.empty()){
            console::info("File path : {}", _image_file);
            console::info("ROI size : {}", _roi_size);
            if(_image_file.empty()){
                console::error("Image file does not exis to process the focus quality measurement.");
            }
            double measure = libfm::fm_measure_from_file(_image_file.c_str(), _roi_size, false);
            console::info("Focus Quality Measure : {}", measure);
        }

        if(_camid!=-1 && _roi_size!=0){
            cv::VideoCapture* _camera = nullptr;

            _camera = new cv::VideoCapture(_camid, cv::CAP_V4L2);
            if(_camera->isOpened()){
                _camera->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                _camera->set(cv::CAP_PROP_FRAME_WIDTH, 1600);
                _camera->set(cv::CAP_PROP_FRAME_HEIGHT, 1200);
                _camera->set(cv::CAP_PROP_FPS, 50);

                cv::Mat raw;
                while(1){
                    if(_camera->grab()){
                        _camera->retrieve(raw);
                        double measure = libfm::fm_measure_from_memory(raw, 300, true);
                        console::info("Focus Quality Measure : {}", measure);
                        if (cv::waitKey(1) == 27)
			                break;
                    }
                }
            }

        }
        
    }
    catch(const std::exception& e){
        console::error("Exception : {}", e.what());
    }

    return EXIT_SUCCESS;
}