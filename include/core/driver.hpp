/**
 * @file driver.hpp
 * @author Byunghun Hwang (bh.hwang@iae.re.kr)
 * @brief Image Source Driver (camera|image|video)
 * @version 0.1
 * @date 2022-01-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */



#ifndef _CORE_SOURCE_DRIVER_HPP_
#define _CORE_SOURCE_DRIVER_HPP_

#include <memory>
#include <opencv2/core/mat.hpp>

using namespace std;


namespace ppe::controller { class iController; }
namespace ppe::source {

    typedef struct _resolution {
			int width;
			int height;
			_resolution():width(0), height(0) {}
			_resolution(int w, int h):width(w), height(h){ }
		} resolution;

    class driver {
        public:
            driver() = default;
            virtual ~driver() = default;


            /**
             * @brief open source device and file
             * 
             * @return true if success
             * @return false if failed
             */
            virtual bool open() = 0;

            /**
             * @brief close source device and file
             * 
             */
            virtual void close() = 0;

            /**
             * @brief driver check
             * 
             * @return true if it is valid
             * @return false if not
             */
            virtual bool is_valid() = 0;

            /**
             * @brief image capture (single take)
             * 
             */
            virtual cv::Mat capture() = 0;

        public:
            cv::Mat camera_matrix = cv::Mat::eye(3, 3,CV_64FC1);
            cv::Mat distortion_coeff = cv::Mat::eye(1, 5,CV_64FC1);
            int width = 0;
            int height = 0;

    };
    
} /* end namespace */

#endif