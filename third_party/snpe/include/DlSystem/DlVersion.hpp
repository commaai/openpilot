//==============================================================================
//
//  Copyright (c) 2014-2015 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================


#ifndef _DL_VERSION_HPP_
#define _DL_VERSION_HPP_

#include "ZdlExportDefine.hpp"
#include <stdint.h>
#include <string>
#include "DlSystem/String.hpp"


namespace zdl {
namespace DlSystem
{
   class Version_t;
}}


namespace zdl { namespace DlSystem
{
/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * A class that contains the different portions of a version number.
 */
class ZDL_EXPORT Version_t
{
public:
   /// Holds the major version number. Changes in this value indicate
   /// major changes that break backward compatibility.
   int32_t         Major;

   /// Holds the minor version number. Changes in this value indicate
   /// minor changes made to library that are backwards compatible
   /// (such as additions to the interface).
   int32_t         Minor;

   /// Holds the teeny version number. Changes in this value indicate
   /// changes such as bug fixes and patches made to the library that
   /// do not affect the interface.
   int32_t         Teeny;

   /// This string holds information about the build version.
   ///
   std::string     Build;

   static zdl::DlSystem::Version_t fromString(const std::string &stringValue);

   static zdl::DlSystem::Version_t fromString(const zdl::DlSystem::String &stringValue);

   /**
    * @brief Returns a string in the form Major.Minor.Teeny.Build
    *
    * @return A formatted string holding the version information.
    */
   const std::string toString() const;

   /**
    * @brief Returns a string in the form Major.Minor.Teeny.Build
    *
    * @return A formatted string holding the version information.
    */
   const zdl::DlSystem::String asString() const;
};

}}

/** @} */ /* end_addtogroup c_plus_plus_apis */

#endif
