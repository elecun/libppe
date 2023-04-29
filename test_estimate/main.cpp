#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>

#include "../libppe/libppe.hpp"
#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

void usage()
{
}

int main(int argc, char *argv[])
{
    try
    {
        //std::string _config_fname = "/opt/sdsol/images/0/config.json";
        std::string _config_fname = "/home/libppe/bin/test_conf.json";
        //std::string _job_fname = "./job.json";
        std::string _job_fname = "/home/libppe/bin/test_job_m.json";
        std::string _config;
        std::string temp;
        std::ifstream cfp;
        cfp.open(_config_fname);
        if (cfp.is_open())
        {
            while (getline(cfp, temp))
            {
                // printf("%s\n", temp.c_str());
                _config += temp;
                _config += "\n";
            }
            cfp.close();

            //printf("%s\n", _config.c_str());
            libppe::initialize();
            if (libppe::set_parameters(_config))
            {
                printf("set parameter success\n");

                // std::string job = "{ \"path\" : \"/opt/sdsol/images/0\", \"files\":[\"zup0.png\", \"zup1.png\"] }";
                std::string _job;
                std::ifstream jfp;
                jfp.open(_job_fname);
                if (jfp.is_open())
                {
                    while (getline(jfp, temp))
                    {
                        _job += temp;
                        _job += "\n";
                    }
                    jfp.close();
                    printf("%s\n", _job.c_str());

                    std::string out = libppe::estimate(_job);
                    //puts(out.c_str());

                    //libppe::set_parameters(_config);
                    //out = libppe::estimate(_job);
                }
            }
        } else {
            printf("failed to read %s\n", _config.c_str());
        }
        libppe::finalize();
    }
    catch (const exception &e)
    {
        printf("catch exceptio0n... %s\n", e.what());
    }

    return 0;
}
