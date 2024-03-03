//=============================================================================
//
//  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================


#ifndef DL_SYSTEM_IOBUFFER_DATATYPE_MAP_HPP
#define DL_SYSTEM_IOBUFFER_DATATYPE_MAP_HPP

#include <cstddef>
#include <memory>
#include "DlSystem/DlEnums.hpp"

namespace DlSystem
{
   // Forward declaration of IOBufferDataTypeMapImpl implementation.
   class IOBufferDataTypeMapImpl;
}

namespace zdl
{
namespace DlSystem
{
/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * @brief .
 *
 * The IoBufferDataTypeMap class definition
 */
class ZDL_EXPORT IOBufferDataTypeMap final
{
public:

    /**
    * @brief .
    *
    * Creates a new Buffer Data type map
    *
    */
   IOBufferDataTypeMap();

   /**
    * @brief Adds a name and the corresponding buffer data type
    *        to the map
    *
    * @param[name] name The name of the buffer
    * @param[bufferDataType] buffer Data Type of the buffer
    *
    * @note If a buffer with the same name already exists, no new
    *       buffer is added.
    */
   void add(const char* name, zdl::DlSystem::IOBufferDataType_t bufferDataType);

   /**
    * @brief Removes a buffer name from the map
    *
    * @param[name] name The name of the buffer
    *
    */
   void remove(const char* name);

   /**
    * @brief Returns the type of the named buffer
    *
    * @param[name] name The name of the buffer
    *
    * @return The type of the buffer, or UNSPECIFIED if the buffer does not exist
    *
    */
   zdl::DlSystem::IOBufferDataType_t getBufferDataType(const char* name);

   /**
    * @brief Returns the type of the first buffer
    *
    * @return The type of the first buffer, or UNSPECIFIED if the map is empty.
    *
    */
   zdl::DlSystem::IOBufferDataType_t getBufferDataType();

   /**
    * @brief Returns the size of the buffer type map.
    *
    * @return The size of the map
    *
    */
   size_t size();

   /**
    * @brief Checks the existence of the named buffer in the map
    *
    * @return True if the named buffer exists, false otherwise.
    *
    */
   bool find(const char* name);

   /**
    * @brief Resets the map
    *
    */
   void clear();

   /**
    * @brief Checks whether the map is empty
    *
    * @return True if the map is empty, false otherwise.
    *
    */
   bool empty();

   /**
    * @brief Destroys the map
    *
    */
   ~IOBufferDataTypeMap();

private:
   std::shared_ptr<::DlSystem::IOBufferDataTypeMapImpl> m_IOBufferDataTypeMapImpl;
};
}

}
#endif
