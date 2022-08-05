// =============================================================================
//
// Copyright (c) 2018-2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
// =============================================================================

#ifndef SNPE_PLATFORMVALIDATOR_HPP
#define SNPE_PLATFORMVALIDATOR_HPP

#include "DlSystem/DlEnums.hpp"
#include "DlSystem/ZdlExportDefine.hpp"

#define DO_PRAGMA(s) _Pragma(#s)
#define NO_WARNING "-Wunused-variable"

#ifdef __clang__
#define SNPE_DISABLE_WARNINGS(clang_warning,gcc_warning) \
_Pragma("clang diagnostic push") \
DO_PRAGMA(clang diagnostic ignored clang_warning)

#define SNPE_ENABLE_WARNINGS \
_Pragma("clang diagnostic pop")

#elif defined __GNUC__
#define SNPE_DISABLE_WARNINGS(clang_warning,gcc_warning) \
_Pragma("GCC diagnostic push") \
DO_PRAGMA(GCC diagnostic ignored gcc_warning)

#define SNPE_ENABLE_WARNINGS \
_Pragma("GCC diagnostic pop")

#else
#define SNPE_DISABLE_WARNINGS(...)
#define SNPE_ENABLE_WARNINGS
#endif

SNPE_DISABLE_WARNINGS("-Wdelete-non-virtual-dtor","-Wdelete-non-virtual-dtor")
#include <string>
#include <memory>
SNPE_ENABLE_WARNINGS

namespace zdl
{
   namespace SNPE
   {
      class PlatformValidator;

      class IPlatformValidatorRuntime;
   }
}

/** @addtogroup c_plus_plus_apis C++
@{ */

/**
* The class for checking SNPE compatibility/capability of a device.
*
*/

class ZDL_EXPORT zdl::SNPE::PlatformValidator
{
public:
   /**
    * @brief Default Constructor of the PlatformValidator Class
    *
    * @return A new instance of a PlatformValidator object
    *         that can be used to check the SNPE compatibility
    *         of a device
    */
   PlatformValidator();

   ~PlatformValidator();

   /**
    * @brief Sets the runtime processor for compatibility check
    *
    * @return Void
    */
   void setRuntime(zdl::DlSystem::Runtime_t runtime);

   /**
    * @brief Checks if the Runtime prerequisites for SNPE are available.
    *
    * @return True if the Runtime prerequisites are available, else false.
    */
   bool isRuntimeAvailable();

   /**
    * @brief Returns the core version for the Runtime selected.
    *
    * @return String which contains the actual core version value
    */
   std::string getCoreVersion();

   /**
    * @brief Returns the library version for the Runtime selected.
    *
    * @return String which contains the actual lib version value
    */
   std::string getLibVersion();

   /**
    * @brief Runs a small program on the runtime and Checks if SNPE is supported for Runtime.
    *
    * @return If True, the device is ready for SNPE execution, else not.
    */

   bool runtimeCheck();

private:
    zdl::DlSystem::Runtime_t m_runtimeType;
    std::unique_ptr<IPlatformValidatorRuntime> m_platformValidatorRuntime;
};
/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif //SNPE_PLATFORMVALIDATOR_HPP
