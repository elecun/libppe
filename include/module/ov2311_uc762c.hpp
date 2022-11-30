/**
 * @file ov2311_uc762c.hpp
 * @author Byunghun Hwang (bh.hwang@iae.re.kr)
 * @brief OV2311 Monochrome CMOS Sensor with UC762C USB 2.0 Controller Interface
 * @version 0.1
 * @date 2022-01-26
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _MODULE_OV2311_UC762C_HPP_
#define _MODULE_OV2311_UC762C_HPP_

#include <string>
#include <memory>
#include <core/driver.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

using namespace std;

#define VCOS_ALIGN_DOWN(p,n) (((ptrdiff_t)(p)) & ~((n)-1))
#define VCOS_ALIGN_UP(p,n) VCOS_ALIGN_DOWN((ptrdiff_t)(p)+(n)-1,(n))

namespace ppe::controller { class iController; }
namespace ppe::cmos {

    class ov2311_uc762c : public ppe::source::driver {
        /* Camera Intrinsic parameters */
        #define _FX_    2517.792
        #define _FY_    2514.767
        #define _FX_ERROR_ 21.207
        #define _FY_ERROR_ 21.231
        #define _FOCAL_LEGNTH_      _FX_, _FY_
        #define _CX_ 814.045
        #define _CY_ 567.330
        #define _CX_ERROR_ 3.114
        #define _CY_ERROR_ 0.755
        #define _PRINCIPAL_POINT_   _CX_, _CY_
        #define _K1_    -0.361044    //Radial distortion
        #define _K2_    0.154482
        #define _P1_    0.000808    //tangential distortion
        #define _P2_    0.000033

        public:
            ov2311_uc762c(int camera_id);
            virtual ~ov2311_uc762c();

            bool open() override;
            void close() override;
            bool is_valid() override;
            cv::Mat capture() override;

        protected:
            ppe::controller::iController* _controller = nullptr;

        private:
            cv::VideoCapture* _camera = nullptr;
            int _camera_id = 0;

    };

} /* end namespace */


#endif