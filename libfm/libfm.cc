

#include "libfm.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

namespace libfm {

   double fm_pos_approx(double wd){
        const double a = 3276.84234;
        const double b = -0.002751;
        const double min_bound = 0.0;
        const double max_bound = 1500.0;

        return clamp(a*exp(b*wd), min_bound, max_bound);
   }

}