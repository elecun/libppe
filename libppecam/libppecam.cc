

#include "libppe.hpp"

namespace libppe {

    int set_configure(const char* config_filename);
    std::vector<pair<double, pos6d>> estimate_pos6d_wafer(const char* source_series);
    std::vector<pair<double, pos6d>> estimate_pos6d_effector(const char* source_series);

}