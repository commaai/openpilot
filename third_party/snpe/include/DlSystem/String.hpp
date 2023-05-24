//=============================================================================
//
//  Copyright (c) 2017, 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef PLATFORM_STANDARD_STRING_HPP
#define PLATFORM_STANDARD_STRING_HPP

#include <cstdio>
#include <string>
#include <ostream>
#include "DlSystem/ZdlExportDefine.hpp"

namespace zdl
{
namespace DlSystem
{
/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * @brief .
 *
 * Class for wrapping char * as a really stripped down std::string replacement.
 */
class ZDL_EXPORT String final
{
public:
   String() = delete;

   /**
    * Construct a string from std::string reference.
    * @param str Reference to a std::string
    */
   explicit String(const std::string& str);

   /**
    * Construct a string from char* reference.
    * @param a char*
    */
   explicit String(const char* str);

   /**
    * move constructor.
    */
   String(String&& other) noexcept;

   /**
    * copy constructor.
    */
   String(const String& other) = delete;

   /**
    * assignment operator.
    */
   String& operator=(const String&) = delete;

   /**
    * move assignment operator.
    */
   String& operator=(String&&) = delete;

   /**
    * class comparators
    */
   bool operator<(const String& rhs) const noexcept;
   bool operator>(const String& rhs) const noexcept;
   bool operator<=(const String& rhs) const noexcept;
   bool operator>=(const String& rhs) const noexcept;
   bool operator==(const String& rhs) const noexcept;
   bool operator!=(const String& rhs) const noexcept;

   /**
    * class comparators against std::string
    */
   bool operator<(const std::string& rhs) const noexcept;
   bool operator>(const std::string& rhs) const noexcept;
   bool operator<=(const std::string& rhs) const noexcept;
   bool operator>=(const std::string& rhs) const noexcept;
   bool operator==(const std::string& rhs) const noexcept;
   bool operator!=(const std::string& rhs) const noexcept;

   const char* c_str() const noexcept;

   ~String();
private:

   char* m_string;
};

/**
 * overloaded << operator
 */
ZDL_EXPORT std::ostream& operator<<(std::ostream& os, const String& str) noexcept;

} // DlSystem namespace
} // zdl namespace

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif // PLATFORM_STANDARD_STRING_HPP
