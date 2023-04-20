

#include "libppe.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <csignal>
#include <ctime>
#include <cxxopts.hpp>
#include <string>
#include <sys/mman.h>
#include "json.hpp"
#include "util.hpp"
#include <fstream>

using namespace std;
namespace console = spdlog;
using namespace nlohmann;

/* calc fps */
int show_fps(){
    static int frame_count = 0;
	static time_t beginTime = time(NULL);
    frame_count++;
	if((time(NULL)-beginTime)>=1)
	{
		beginTime = time(NULL);
        spdlog::info("FPS : {}", frame_count);
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

    cxxopts::Options options("TEST program for libppe library");
    options.add_options()
        ("c,config", "Set parameters from configuration file(*.json)", cxxopts::value<string>())
        ("j,job", "Execute pose estimation function with job file(*.json)", cxxopts::value<string>())
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

    string _config_filename {""};
    json _config;
    string _job_filename {""};
    json _job;

    if(optval.count("config")){
        _config_filename = optval["config"].as<string>();
    }
    if(optval.count("job")){
        _job_filename = optval["job"].as<string>();
    }
    else
        return EXIT_FAILURE;

    try{
        if(!util::exist(_config_filename.c_str())){
            cout << "Configuration file does not exist" << endl;
            return EXIT_FAILURE;
        }

        if(!util::exist(_job_filename.c_str())){
            cout << "Pose estimation job file does not exist" << endl;
            return EXIT_FAILURE;
        }

        std::ifstream cfile(_config_filename);
        cfile >> _config;

        std::ifstream jfile(_job_filename);
        jfile >> _config;

        if(libppe::set_parameters(_config.dump())){
            string out = libppe::estimate(_job.dump());
            console::info("{}", out);
            // json result;
            // result.parse(out);
        }        
        
    }
    catch(const std::exception& e){
        console::error("Exception : {}", e.what());
    }

    return EXIT_SUCCESS;
}