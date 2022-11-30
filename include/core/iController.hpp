/**
 * @file iController.hpp
 * @author Byunghun Hwang (bh.hwang@iae.re.kr)
 * @brief prepheral controller interface
 * @version 0.1
 * @date 2022-01-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _CORE_CONTROLLER_HPP_
#define _CORE_CONTROLLER_HPP_

namespace ppe::controller {

    class iController {
        public:
            iController() = default;
            virtual ~iController() = default;
    };

} /* end namespace */


#endif