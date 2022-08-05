//=============================================================================
//
//  Copyright (c) 2015, 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#ifndef __IDIAGLOG_HPP_
#define __IDIAGLOG_HPP_

#include <string>

#include "DiagLog/Options.hpp"
#include "DlSystem/String.hpp"
#include "DlSystem/ZdlExportDefine.hpp"

namespace zdl
{
namespace DiagLog
{

/** @addtogroup c_plus_plus_apis C++
@{ */

/// @brief .
///
/// Interface for controlling logging for zdl components.

class ZDL_EXPORT IDiagLog
{
public:

   /// @brief .
   ///
   /// Sets the options after initialization occurs.
   ///
   /// @param[in] loggingOptions The options to set up diagnostic logging.
   ///
   /// @return False if the options could not be set. Ensure logging is not started.
   virtual bool setOptions(const Options& loggingOptions) = 0;

   /// @brief .
   ///
   /// Gets the curent options for the diag logger.
   ///
   /// @return Diag log options object.
   virtual Options getOptions() = 0;

   /// @brief .
   ///
   /// Allows for setting the log mask once diag logging has started
   ///
   /// @return True if the level was set successfully, false if a failure occurred.
   virtual bool setDiagLogMask(const std::string& mask) = 0;

   /// @brief .
   ///
   /// Allows for setting the log mask once diag logging has started
   ///
   /// @return True if the level was set successfully, false if a failure occurred.
   virtual bool setDiagLogMask(const zdl::DlSystem::String& mask) = 0;

   /// @brief .
   ///
   /// Enables logging for zdl components.
   ///
   /// Logging should be started prior to the instantiation of zdl components
   /// to ensure all events are captured.
   ///
   /// @return False if diagnostic logging could not be started.
   virtual bool start(void) = 0;

   /// @brief Disables logging for zdl components.
   virtual bool stop(void) = 0;

   virtual ~IDiagLog() {};
};

} // DiagLog namespace
} // zdl namespace

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif
