

#include "libfm.hpp"
#include <cmath>
#include <algorithm>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>

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

   double fm_measure(const char* image_path, unsigned int roi_size=300){
      cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
      cv::Mat grayscale;
      cv::cvtColor(image, grayscale, CV_BGR2GRAY);

      cv::Mat sobel_x;
      cv::Mat sobel_y;
      cv::Sobel(grayscale, sobel_x, CV_64F, 1, 0);
      cv::Sobel(grayscale, sobel_y, CV_64F, 0, 1);
      cv::Mat sobel = cv::abs(sobel_x) + cv::abs(sobel_y);



      return 0.0;
   }

}