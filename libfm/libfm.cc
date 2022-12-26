

#include "libfm.hpp"
#include <cmath>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

namespace libfm {

   double fm_pos_approx(double wd){
        const double a = 3276.84234;
        const double b = -0.02751;
        const double min_bound = 0.0;
        const double max_bound = 1500.0;

        return clamp(a*exp(b*wd), min_bound, max_bound);
   }

   double fm_measure_from_file(const char* image_path, unsigned int roi_size, bool show){
      cv::Mat image = cv::imread(image_path);
      cv::Mat grayscale;
      cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);

      cv::Rect bounds(0,0,grayscale.cols,grayscale.rows);
      cv::Rect roi(bounds.width/2-roi_size/2, bounds.height/2-roi_size/2, roi_size, roi_size);
      cv::Mat subimage = grayscale(roi & bounds);

      cv::Mat sobel_x;
      cv::Mat sobel_y;
      cv::Sobel(subimage, sobel_x, CV_64F, 1, 0);
      cv::Sobel(subimage, sobel_y, CV_64F, 0, 1);

      cv::Mat scaled_sobel_x;
      cv::convertScaleAbs(sobel_x, scaled_sobel_x);

      cv::Mat scaled_sobel_y;
      cv::convertScaleAbs(sobel_y, scaled_sobel_y);

      cv::Mat sobel;
      cv::addWeighted(scaled_sobel_x, 1, scaled_sobel_y, 1, 0, sobel);
      
      cv::imwrite("./fm_result.jpg", sobel);
      if(show){
         cv::imwrite("./fm_result.jpg", sobel);
         cv::imshow("result", subimage);
      }

      cv::Scalar mean, std;
      cv::meanStdDev(sobel, mean, std);

      return (std.val[0]*std.val[0]);
   }

   double fm_measure_from_memory(cv::Mat image, unsigned int roi_size, bool show){
      cv::Mat grayscale;
      if(image.channels()!=1){
         cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
      }
      else
         grayscale = image.clone();

      cv::Rect bounds(0,0,grayscale.cols,grayscale.rows);
      cv::Rect roi(bounds.width/2-roi_size/2, bounds.height/2-roi_size/2, roi_size, roi_size);
      cv::Mat subimage = grayscale(roi & bounds);

      cv::Mat sobel_x;
      cv::Mat sobel_y;
      cv::Sobel(subimage, sobel_x, CV_64F, 1, 0);
      cv::Sobel(subimage, sobel_y, CV_64F, 0, 1);

      cv::Mat scaled_sobel_x;
      cv::convertScaleAbs(sobel_x, scaled_sobel_x);

      cv::Mat scaled_sobel_y;
      cv::convertScaleAbs(sobel_y, scaled_sobel_y);

      cv::Mat sobel;
      cv::addWeighted(scaled_sobel_x, 1, scaled_sobel_y, 1, 0, sobel);

      if(show){
         cv::imwrite("./fm_result.jpg", sobel);
         cv::imshow("result", subimage);
      }

      cv::Scalar mean, std;
      cv::meanStdDev(sobel, mean, std);

      return (std.val[0]*std.val[0]);
   }

   

}