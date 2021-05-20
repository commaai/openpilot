//=============================================================================
//
//  Copyright (c) 2016 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#include <cstdio>
#include "ZdlExportDefine.hpp"

#ifndef DL_SYSTEM_STRINGLIST_HPP
#define DL_SYSTEM_STRINGLIST_HPP

namespace zdl
{
namespace DlSystem
{
/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * @brief .
 *
 * Class for holding an order list of null-terminated ASCII strings.
 */
class ZDL_EXPORT StringList final
{
public:
   StringList() {}

   /**
    * Construct a string list with some pre-allocated memory.
    * @warning Contents of the list will be uninitialized
    * @param[in] length Number of elements for which to pre-allocate space.
    */
   explicit StringList(size_t length);

   /**
    * Append a string to the list.
    * @param[in] str Null-terminated ASCII string to append to the list.
    */
   void append(const char* str);

   /**
    * Returns the string at the indicated position,
    *  or an empty string if the positions is greater than the size
    *  of the list.
    * @param[in] idx Position in the list of the desired string
    */
   const char* at(size_t idx) const noexcept;

   /**
    * Pointer to the first string in the list.
    *  Can be used to iterate through the list.
    */
   const char** begin() const noexcept;

   /**
    * Pointer to one after the last string in the list.
    *  Can be used to iterate through the list.
    */
   const char** end() const noexcept;

   /**
    * Return the number of valid string pointers held by this list.
    */
   size_t size() const noexcept;


   /**
    * assignment operator. 
    */
   StringList& operator=(const StringList&) noexcept;

   /**
    * copy constructor.
    * @param[in] other object to copy.
    */
   StringList(const StringList& other);

   /**
    * move constructor.
    * @param[in] other object to move.    
    */
   StringList(StringList&& other) noexcept;

   ~StringList();
private:
   void copy(const StringList& other);

   void resize(size_t length);

   void clear();

   static const char* s_Empty;
   const char** m_Strings = nullptr;
   const char** m_End = nullptr;
   size_t       m_Size = 0;
};

} // DlSystem namespace
} // zdl namespace

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif // DL_SYSTEM_STRINGLIST_HPP

