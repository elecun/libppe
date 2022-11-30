/**
 * @file ov2311.hpp
 * @author Byunghun Hwang (bh.hwang@iae.re.kr)
 * @brief OV2311 Camera Device with various interface(MIPI, USB3.0)
 * @version 0.1
 * @date 2022-01-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _MODULE_OV2311_HPP_
#define _MODULE_OV2311_HPP_

#include <string>
#include <memory>
#include <core/driver.hpp>
#include <module/ArduCamLib.h>

#include <opencv2/opencv.hpp>
#include <thread>
#include <time.h>
#include <iostream>
#include <istream>
#include <string>
#include <sys/types.h> 
#include <sys/stat.h>
#include <module/a
#include "arducam_config_parser.h"

using namespace std;

namespace ppe::controller { class iController; }
namespace ppe::cmos {

    class ov2311 : public ppe::source::driver {
        public:
            ov2311(const char* config);
            virtual ~ov2311();

            void set_bus(ppe::controller::iController* bus) override;
            bool open() override;
            void close() override;
            bool is_valid() override;
            void capture() override;

        protected:
            ppe::controller::iController* _controller = nullptr;

        private:
            string _config_file {""};

            ArduCamHandle cameraHandle;

    };

} /* end namespace */


#endif