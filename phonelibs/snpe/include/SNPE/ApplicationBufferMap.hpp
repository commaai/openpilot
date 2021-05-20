//==============================================================================
//
//  Copyright (c) 2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef PSNPE_APPLICATIONBUFFERMAP_HPP
#define PSNPE_APPLICATIONBUFFERMAP_HPP
#include <vector>
#include <string>
#include <unordered_map>

#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/ZdlExportDefine.hpp"

namespace zdl
{
namespace PSNPE
{
/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * @brief .
 *
 * A class representing the UserBufferMap of Input and Output asynchronous mode.
 */

class ZDL_EXPORT ApplicationBufferMap final
{

 public:
   /**
    * @brief Adds a name and the corresponding buffer
    *        to the map
    *
    * @param[in] name The name of the UserBuffer
    * @param[in] buffer The vector of the uint8_t data
    *
    * @note If a UserBuffer with the same name already exists, the new
    *       UserBuffer pointer would be updated.
    */
   void add(const char* name, std::vector<uint8_t>& buff) noexcept;
   void add(const char* name, std::vector<float>& buff) noexcept;
   /**
    * @brief Removes a mapping of one UserBuffer and its name by its name
    *
    * @param[in] name The name of UserBuffer to be removed
    *
    * @note If no UserBuffer with the specified name is found, nothing
    *       is done.
    */
   void remove(const char* name) noexcept;

   /**
    * @brief Returns the number of UserBuffers in the map
    */
   size_t size() const noexcept;

   /**
    * @brief .
    *
    * Removes all UserBuffers from the map
    */
   void clear() noexcept;

   /**
    * @brief Returns the UserBuffer given its name.
    *
    * @param[in] name The name of the UserBuffer to get.
    *
    * @return nullptr if no UserBuffer with the specified name is
    *         found; otherwise, a valid pointer to the UserBuffer.
    */
   const std::vector<uint8_t>& getUserBuffer(const char* name) const;
   const std::vector<uint8_t>& operator[](const char* name) const;
   /**
    * @brief .
    *
    * Returns the names of all UserAsyncBufferMap
    *
    * @return A list of UserBuffer names.
    */
   zdl::DlSystem::StringList getUserBufferNames() const;
   const std::unordered_map<std::string, std::vector<uint8_t>>& getUserBuffer() const;
   explicit ApplicationBufferMap();
   ~ApplicationBufferMap();
   explicit ApplicationBufferMap(
     const std::unordered_map<std::string, std::vector<uint8_t>> buffer);

 private:
   std::unordered_map<std::string, std::vector<uint8_t>> m_UserMap;
};

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */
} // namespace PSNPE
} // namespace zdl

#endif // PSNPE_APPLICATIONBUFFERMAP_HPP
