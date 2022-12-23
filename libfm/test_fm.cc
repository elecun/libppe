

#include "libfm.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <csignal>
#include <ctime>
#include <cxxopts.hpp>
#include <string>
#include <sys/mman.h>

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
        ("i,im", "Image file path", cxxopts::value<string>())
        ("m,fm", "Focus Measure", cxxopts::value<int>())
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
    int _fm = 0;
    string _im = "";

    if(optval.count("wd")){
        _wd = optval["wd"].as<double>();
    }
    if(optval.count("fm")){
        _fm = optval["fm"].as<int>();
    }
    if(optval.count("im")){
        _im = optval["im"].as<string>();
    }

    try{
        if(_wd!=0){
            double approx_pos = libfm::fm_pos_approx(_wd);
            console::info("Appoximated Focus Position : {}", approx_pos);
        }
        if(_fm!=0 && !_im.empty()){
            if(_im.empty()){
                console::error("Image file does not exis to process the focus quality measurement.");
            }
            int measure = libfm::fm_measure(_im.c_str(), _fm);
        }
        else {
            console::warn("Working distance could not be zero value.");
        }
    }
    catch(const std::exception& e){
        console::error("Exception : {}", e.what());
    }

    return EXIT_SUCCESS;
}