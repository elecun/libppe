/**
 * @file uc621b.hpp
 * @author Byunghun Hwang (bh.hwang@iae.re.kr)
 * @brief OV2311 MIPI bus controller Rev.B
 * @version 0.1
 * @date 2022-01-17
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _MODULE_UC621B_HPP_
#define _MODULE_US621B_HPP_

#include <include/core/iController.hpp>

namespace ppe::controller {

    class uc621b : public iController {
        public:
            uc621b() = default;
            virtual ~uc621b() = default;
    }; 

} /* end namespace */

#endif