

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


#define _DEVICE_ "device"

using namespace std;
using namespace nlohmann;
using namespace cv;

namespace libppe {

    //newly added @ 23.03.29
    json _param;
    
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

                for(json::iterator itr = _files.begin(); itr != _files.end(); ++itr){
                    if(itr.value().is_string()){
                        _image_files.push_back(itr.value());
                    }
                    else {
                        cout << 
                    }
                        
                }
            }
            else{
                std::cerr << "Some parameters are missing." << std::endl;
            }

        }
        catch(json::parse_error& e){
            std::cerr << "parse error at byte " << e.byte << std::endl;
            std::cerr << "what : " << e.what() << std::endl;
        }

        return _result.dump();
    }

    bool set_parameters(string parameters){
        try {
            json param = json::parse(parameters);
        }
        catch (json::parse_error& ex){
            std::cerr << "parse error at byte " << ex.byte << std::endl;
            std::cerr << "what : " << ex.what() << std::endl;
        }
        return false;
    }

}