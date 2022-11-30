
/**
 * @file    icamera.hpp
 * @brief   camera device interface class with opencv2
 * @author  Byunghun hwang<bh.hwagn@iae.re.kr>
 * 
 */


#ifndef _INCLUDE_CORE_CAMERA_HPP_
#define _INCLUDE_CORE_CAMERA_HPP_

#include <opencv2/core.hpp>


namespace ppe {

    class iCamera {
        public:

        //resolution
		enum class Screen : int { QVGA = 0, VGA, SVGA, XGA, HD720, HD1080, CUSTOM_OV2311 };
		typedef struct _Resolution {
			int width;
			int height;
			_Resolution():width(0), height(0) {}
			_Resolution(int w, int h):width(w), height(h){ }
			_Resolution(Screen& scn) {
				switch (scn) {
				case Screen::QVGA: width = 320; height = 240; break;
				case Screen::VGA: width = 640; height = 480; break;
				case Screen::SVGA: width = 800; height = 600; break;
				case Screen::XGA: width = 1024; height = 768; break;
				case Screen::HD720: width = 1280; height = 720; break;
				case Screen::HD1080: width = 1920; height = 1080; break;
                case Screen::CUSTOM_OV2311: width=1600; height=1300; break;
				}
			}
		} Resolution;

        //camera property
		typedef struct _sProperty {
			Resolution resolution;
			int fps;
		} sProperty;

		typedef struct _sIntrinsic {
			double fx, fy;	//focal length
			double cx, cy;	//principal point (offset from optical center position)
			double k1, k2;	//radial distortion parameter
            double p1, p2;  //tangential distortion parameter
			_sIntrinsic(){
				fx = fy = 600.;
				cx = cy = 0.;	
				k1 = k2 = 0.;
                p1 = p2 = 0.;
			}
			_sIntrinsic(double f_x, double f_y, double c_x, double c_y)
				:fx(f_x), fy(f_y), cx(c_x), cy(c_y), k1(0.0), k2(0.0){}
			_sIntrinsic(double f_x, double f_y, double c_x, double c_y, double k_1, double k_2)
				:fx(f_x), fy(f_y), cx(c_x), cy(c_y), k1(k_1), k2(k_2){}
            _sIntrinsic(double f_x, double f_y, double c_x, double c_y, double k_1, double k_2, double p_1, double p_2)
				:fx(f_x), fy(f_y), cx(c_x), cy(c_y), k1(k_1), k2(k_2), p1(p_1), p2(p_2){ }
            void set(double f_x, double f_y, double c_x, double c_y, double k_1, double k_2, double p_1, double p_2){
                this->fx = f_x; this->fy = f_y;
                this->cx = c_x; this->cy = c_y;
                this->k1 = k_1; this->k2 = k_2;
                this->p1 = p_1; this->p2 = p_2;
            }
				
		} sIntrinsic;

        iCamera(int id, iCamera::sIntrinsic intrinsic, int width_resolution, int height_resolution)
        :_device_id(id), _intrinsic(intrinsic){
            _property.resolution.width = width_resolution;
            _property.resolution.height = height_resolution;
            if(intrinsic.cx==0) _intrinsic.cx = width_resolution/2.;
            if(intrinsic.cy==0) _intrinsic.cy = height_resolution/2.;

            // camera parameters initialize
            _camera_matrix = cv::Mat::eye(3,3,CV_64FC1);
            _distortion_coeff = cv::Mat::zeros(1,5,CV_64FC1);
        }

        iCamera(int id, int width_resolution, int height_resolution)
        :_device_id(id){
            _property.resolution.width = width_resolution;
            _property.resolution.height = height_resolution;
            _intrinsic.cx = width_resolution/2.;
            _intrinsic.cy = height_resolution/2.;
        }

        virtual ~iCamera() {

        }

        /* support interface function */
        virtual cv::Mat* capture() = 0; //capture single frame
        virtual bool capture_n() = 0; //capture multi frame
        virtual int getID() const { return this->_device_id; }
        virtual Resolution* getResolution() { return &this->_property.resolution; }
        virtual void close() = 0;

        /**
         * @brief camera device open
         * 
         * @return true success to open
         * @return false fail to open
         */
        virtual bool open() = 0;

        /**
         * @brief parse configuration file for camera parameters
         * @param config_file_path configuration file
         * @return true success
         * @return false fail to read
         */
        virtual bool parse_config(const char* config_file_path) = 0;

        public:
        cv::Mat _camera_matrix;
        cv::Mat _distortion_coeff;

        protected:
        sIntrinsic _intrinsic;
        sProperty _property;

        private:
        int _device_id;
            
    }; //class

} /* namespace */

#endif