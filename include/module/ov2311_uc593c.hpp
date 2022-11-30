/**
 * @file ov2311_uc593c.hpp
 * @author Byunghun Hwang (bh.hwang@iae.re.kr)
 * @brief OV2311 Monochrome CMOS Sensor with UC593C USB 3.0 Controller Interface
 * @version 0.1
 * @date 2022-01-17
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _MODULE_OV2311_UC593C_HPP_
#define _MODULE_OV2311_UC593C_HPP_

#include <string>
#include <memory>
#include <core/driver.hpp>
#include <thread>
#include <module/arducam_config_parser.h>
#include <module/ArduCamLib.h>

using namespace std;

#define VCOS_ALIGN_DOWN(p,n) (((ptrdiff_t)(p)) & ~((n)-1))
#define VCOS_ALIGN_UP(p,n) VCOS_ALIGN_DOWN((ptrdiff_t)(p)+(n)-1,(n))

namespace ppe::controller { class iController; }
namespace ppe::cmos {

    class ov2311_uc593c : public ppe::source::driver {
        /* Camera Intrinsic parameters */
        #define _FX_    906.666
        #define _FY_    906.611
        #define _FX_ERROR_ 0.296
        #define _FY_ERROR_ 0.290
        #define _FOCAL_LEGNTH_      _FX_, _FY_
        #define _CX_ 769.103
        #define _CY_ 689.538
        #define _CX_ERROR_ 0.170
        #define _CY_ERROR_ 0.096
        #define _PRINCIPAL_POINT_   _CX_, _CY_
        #define _K1_    0.034472    //Radial distortion
        #define _K2_    -0.046075
        #define _P1_    0.000539    //tangential distortion
        #define _P2_    0.000405

        public:
            ov2311_uc593c(const char* config, unsigned int max_fps = 5);
            virtual ~ov2311_uc593c();

            bool open() override;
            void close() override;
            bool is_valid() override;
            cv::Mat capture() override;

        private:
            cv::Mat convert(ArduCamOutData* framedata);
            void config_board(ArduCamHandle &cameraHandle, Config config);
            bool configure(const char* filename, ArduCamHandle &handle, ArduCamCfg &config);    //arducam configurations
            cv::Mat dBytesToMat(Uint8* bytes, int bit_width, int width, int height);
            cv::Mat BytestoMat(Uint8* bytes, int width, int height);

        protected:
            ppe::controller::iController* _controller = nullptr;

        private:
            string _config_file  = ""; //camera configuration file
            unsigned int _max_fps = 5;
            std::thread* _capture_thread = nullptr;


    };

} /* end namespace */


#endif