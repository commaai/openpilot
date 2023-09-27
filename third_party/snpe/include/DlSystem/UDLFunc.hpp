//==============================================================================
//
//  Copyright (c) 2015 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef _UDL_FUNC_HPP_
#define _UDL_FUNC_HPP_

#include <functional>

#include "ZdlExportDefine.hpp"
#include <DlSystem/IUDL.hpp>

namespace zdl {
    namespace DlSystem {
        class UDLContext;
    }
}

namespace zdl { namespace DlSystem {
/** @addtogroup c_plus_plus_apis C++
@{ */
/**
 * @brief .
 *
 * Definition of UDLFactoyFunc, using/typedef and default FactoryFunction
 * UDLBundle - a simple way to bundle func and cookie into one type
 */


/**
 * @brief .
 * 
 * Convenient typedef for user defined layer creation factory
 *
 * @param[out] void* Cookie - a user opaque data that was passed during SNPE's runtime's
 *        CreateInstance. SNPE's runtime is passing this back to the user.
 *
 * @param[out] DlSystem::UDLContext* - The specific Layer Description context what is passe
 *        SNPE runtime.
 *
 * @return IUDL* - a Concrete instance of IUDL derivative
 */
using UDLFactoryFunc = std::function<zdl::DlSystem::IUDL* (void*, const zdl::DlSystem::UDLContext*)>;

/**
 * @brief .
 *
 * default UDL factory implementation
 *
 * @param[out] DlSystem::UDLContext* - The specific Layer Description context what is passe
 *        SNPE runtime.
 *
 * @param[out] void* Cookie - a user opaque data that was passed during SNPE's runtime's
 *        CreateInstance. SNPE's runtime is passing this back to the user.
 * 
 * @return IUDL* - nullptr to indicate SNPE's runtime that there is no specific
 *         implementation for UDL. When SNPE's runtime sees nullptr as a return
 *         value from the factory, it will halt execution if model has an unknown layer
 *
 */
inline ZDL_EXPORT zdl::DlSystem::IUDL* DefaultUDLFunc(void*, const zdl::DlSystem::UDLContext*) { return nullptr; }

/**
 * @brief .
 * 
 * Simple struct to bundle 2 elements.
 * A user defined cookie that would be returned for each
 * IUDL call. The user can place anything there and the
 * SNPE runtime will provide it back
 */
struct ZDL_EXPORT UDLBundle {
   void          *cookie = nullptr;
   UDLFactoryFunc func   = DefaultUDLFunc;
};

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */
}}


#endif // _UDL_FUNC_HPP_
