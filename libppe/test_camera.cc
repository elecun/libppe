

#include "libppe2d.hpp"
#include "camera.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <csignal>
#include <ctime>

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
    console::stdout_color_st("console");
    signal_set();

    libppe2d::camera* _camera = new libppe2d::ov2311_uc762c(0); //id 0
    if(_camera->open()){
        
    }


    console::error("Camera device cannot be opened.");
    delete _camera;

    return EXIT_SUCCESS;
}