/**
 * @file camera.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief Camera Device
 * @version 0.1
 * @date 2022-11-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _PPE2D_CAMERA_HPP_
#define _PPE2D_CAMERA_HPP_

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

namespace libppe2d {

    /**
     * @brief camera interface
     */
    class camera {
        public:
            camera() = default;
            virtual ~camera() = default;

            virtual bool open() = 0;
            virtual void close() = 0;
            virtual bool is_valid() = 0;
            virtual cv::Mat capture() = 0;
        
        protected:
            cv::Mat camera_matrix = cv::Mat::eye(3, 3,CV_64FC1);
            cv::Mat distortion_coeff = cv::Mat::eye(1, 5,CV_64FC1);
            int width = 0;
            int height = 0;
    };

    /**
     * @brief OV2311 CMOS with UC762C Controller
     */

    class ov2311_uc762c : public camera {
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

        public:
            cv::VideoCapture* _camera = nullptr;
            cv::VideoWriter* _video = nullptr;
            int _camera_id = 0;
    };
}

#endif