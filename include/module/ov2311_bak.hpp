

/**
 * @file    ov2311.hpp
 * @brief   OV2311 Camera module (arducam ov2311 with MIPI interface)
 * @author  Byunghun Hwang<bh.hwang@iae.re.kr>
 */

#ifndef _INCLUDE_MODULE_OV2311_HPP_
#define _INCLUDE_MODULE_OV2311_HPP_

#include <core/icamera.hpp>
#include <linux/v4l2-controls.h>
#include <module/arducam_mipicamera.h>
#include <include/spdlog/spdlog.h>

#include<opencv2/imgproc/imgproc.hpp>


/* Camera Intrinsic parameters */
#define _FX_    789.733
#define _FY_    790.183
#define _FX_ERROR_ 0.531
#define _FY_ERROR_ 0.537
#define _FOCAL_LEGNTH_      _FX_, _FY_
#define _CX_ 799.696
#define _CY_ 620.672
#define _CX_ERROR_ 0.267
#define _CY_ERROR_ 0.297
#define _PRINCIPAL_POINT_   _CX_, _CY_
#define _K1_    -0.280639   //Radial distortion
#define _K2_    0.055221
#define _P1_    0.000597    //tangential distortion
#define _P2_    -0.000566
const double _DISTORTION_[] = {-0.280639, 0.055221, 0.000597, -0.000566};
const double _CAMERA_MATRIX_[3][3] = {
    {789.733,0,799.696}, 
    {0,790.183,620.672}, 
    {0,0,1}
    };

#define _RESOLUTION_W_  1600
#define _RESOLUTION_H_  1300
#define _MAX_RESOLUTION_ _RESOLUTION_W_, _RESOLUTION_H_

#define VCOS_ALIGN_DOWN(p,n) (((ptrdiff_t)(p)) & ~((n)-1))
#define VCOS_ALIGN_UP(p,n) VCOS_ALIGN_DOWN((ptrdiff_t)(p)+(n)-1,(n))

namespace ppe {

    /**
     * @brief OV2311 CMOS sensor camera class
     * 
     */
    class ov2311 : public ppe::iCamera {

        typedef void* CAMERA_INSTANCE;

        public:
        ov2311(int id, int width_resolution, int height_resolution)
        :iCamera(id, width_resolution, height_resolution){
            _intrinsic.set(_FX_, _FY_, _CX_, _CY_, _K1_, _K2_, _P1_, _P2_);
        }
        virtual ~ov2311(){

        }

        /**
         * @brief camera device open
         * @param w width resolution
         * @param h height resolution
         */
        bool open(){
            int init_rc = arducam_init_camera(&_camera);
            spdlog::info("Camera Initialize Status = {}", init_rc);

            if(init_rc==0){
                int rc = arducam_set_resolution(_camera, &_property.resolution.width, &_property.resolution.height);
                if(!rc) { //success
                    spdlog::info("Set camera resolution : {},{}", _property.resolution.width, _property.resolution.height);

                    //calc camera matrix
                    _camera_matrix = (cv::Mat1d(3,3) << _FX_, 0., _CX_, 0., _FY_, _CY_, 0., 0., 1.);
                    _distortion_coeff = (cv::Mat1d(1,5) << _K1_, _K2_, 0, _P1_, _P2_);
                }
                return true;
            }
            return false;
        }

        /* camera capture */
        cv::Mat* capture(){
            IMAGE_FORMAT fmt = {IMAGE_ENCODING_I420, 50};
            BUFFER *buffer = arducam_capture(_camera, &fmt, 3000);
            if(!buffer)
                return nullptr;

            int width = VCOS_ALIGN_UP(_property.resolution.width, 32);
            int height = VCOS_ALIGN_UP(_property.resolution.height, 16);
            cv::Mat *image = new cv::Mat(cv::Size(width,(int)(height * 1.5)), CV_8UC1, buffer->data);
            cv::cvtColor(*image, *image, cv::COLOR_YUV2BGR_I420);
            arducam_release_buffer(buffer);

            //return image;
            return nullptr;
        }

        bool capture_n(){
            IMAGE_FORMAT fmt = {IMAGE_ENCODING_I420, 50};
            BUFFER *buffer = arducam_capture(_camera, &fmt, 3000);
            if(!buffer)
                return false;
            spdlog::info("Capture Data size : {}/{}",buffer->length, buffer->alloc_size);
            arducam_release_buffer(buffer);
            return true;
        }

        /* close camera */
        void close(){
            if(_camera){
                int res = arducam_close_camera(_camera);
                if(!res)
                    spdlog::info("Camera closed successfully");
            }
        }


        private:
        CAMERA_INSTANCE _camera;

    }; /* class */
} /* namespace */


#endif