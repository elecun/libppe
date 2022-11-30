
#include "ov2311.hpp"
#include <core/iController.hpp>


namespace ppe::cmos {

    ov2311::ov2311(const char* config):_config_file(config){

    }
    
    ov2311::~ov2311(){

    }

    void ov2311::set_bus(ppe::controller::iController* bus){
        _controller = bus;
    }

    bool ov2311::open(){
        return false;
    }

    void ov2311::close(){
        
    }

    bool ov2311::is_valid(){
        return false;
    }

    void ov2311::capture(){

    }
}