

#include "libppecam.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <csignal>
#include <ctime>
#include <cxxopts.hpp>
#include <string>
#include <sys/mman.h>

using namespace std;
namespace console = spdlog;

/* calc fps */
int show_fps(){
    static int frame_count = 0;
	static time_t beginTime = time(NULL);
    frame_count++;
	if((time(NULL)-beginTime)>=1)
	{
		beginTime = time(NULL);
        console::info("FPS : {}", frame_count);
		frame_count = 0;
	}
    return frame_count;
}

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

    libppecam::cam_trigger_off();
    libppecam::cam_close();
        
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
    string desc = fmt::format("Ver. {}.{}.{} (built {}/{})", __MAJOR__, __MINOR__, __REV__, __DATE__, __TIME__);
    cxxopts::Options options("OpenEdge Framework Engine", desc.c_str());
    options.add_options()
        ("c,config", "Application start with configuration file(*.conf)", cxxopts::value<string>())
        ("h,help", "Print usage");

    auto optval = options.parse(argc, argv);
    if(optval.count("help")){
        std::cout << options.help() << std::endl;
        exit(EXIT_SUCCESS);
    }

    console::stdout_color_st("console");
    signal_set();

    mlockall(MCL_CURRENT|MCL_FUTURE); //avoid memory swaping

    string _config {""};
    if(optval.count("config")){
        _config = optval["config"].as<string>();
    }

    try{
        if(!_config.empty()){
            if(!libppecam::set_configure(_config.c_str())){
                if(libppecam::cam_open()){
                    string saved = libppecam::cam_trigger_on(1000);
                    console::info("images saved in {}", saved);
                }
            }
        }
    }
    catch(const std::exception& e){
        console::error("Exception : {}", e.what());
    }

    return EXIT_SUCCESS;
}