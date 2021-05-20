//=============================================================================
//
//  Copyright (c) 2015, 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#ifndef __DIAGLOG_OPTIONS_HPP_
#define __DIAGLOG_OPTIONS_HPP_

#include <string>
#include "DlSystem/ZdlExportDefine.hpp"

namespace zdl
{
namespace DiagLog
{
/** @addtogroup c_plus_plus_apis C++
@{ */

/// @brief .
///
/// Options for setting up diagnostic logging for zdl components.
class ZDL_EXPORT Options
{
public:
   Options() :
      DiagLogMask(""),
      LogFileDirectory("diaglogs"),
      LogFileName("DiagLog"),
      LogFileRotateCount(20),
      LogFileReplace(true)
   {
      // Solves the empty string problem with multiple std libs
      DiagLogMask.reserve(1);
   }

   /// @brief .
   ///
   /// Enables diag logging only on the specified area mask (DNN_RUNTIME=ON | OFF)
   std::string DiagLogMask;

   /// @brief .
   ///
   /// The path to the directory where log files will be written.
   /// The path may be relative or absolute. Relative paths are interpreted
   /// from the current working directory.
   /// Default value is "diaglogs"
   std::string LogFileDirectory;

   /// @brief .
   ///
   //// The name used for log files. If this value is empty then BaseName will be
   /// used as the default file name.
   /// Default value is "DiagLog"
   std::string LogFileName;

   /// @brief .
   ///
   /// The maximum number of log files to create. If set to 0 no log rotation
   /// will be used and the log file name specified will be used each time, overwriting
   /// any existing log file that may exist.
   /// Default value is 20
   uint32_t LogFileRotateCount;

   /// @brief
   ///
   /// If the log file already exists, control whether it will be replaced
   /// (existing contents truncated), or appended.
   /// Default value is true
   bool LogFileReplace;
};
/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

} // DiagLog namespace
} // zdl namespace


#endif
