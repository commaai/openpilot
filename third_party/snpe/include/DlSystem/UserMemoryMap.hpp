//=============================================================================
//
//  Copyright (c) 2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#include <memory>
#include "ZdlExportDefine.hpp"
#include "StringList.hpp"

#ifndef DL_SYSTEM_USER_MEMORY_MAP_HPP
#define DL_SYSTEM_USER_MEMORY_MAP_HPP

namespace DlSystem
{
    // Forward declaration of UserMemory map implementation.
    class UserMemoryMapImpl;
}

namespace zdl
{
namespace DlSystem
{
class IUserBuffer;

/** @addtogroup c_plus_plus_apis C++
@{ */

/**
  * @brief .
  *
  * A class representing the map of UserMemory.
  */
class ZDL_EXPORT UserMemoryMap final
{
public:

    /**
     * @brief .
     *
     * Creates a new empty UserMemory map
     */
    UserMemoryMap();

    /**
     * copy constructor.
     * @param[in] other object to copy.
     */
    UserMemoryMap(const UserMemoryMap& other);

    /**
      * assignment operator.
      */
    UserMemoryMap& operator=(const UserMemoryMap& other);

    /**
     * @brief Adds a name and the corresponding buffer address
     *        to the map
     *
     * @param[in] name The name of the UserMemory
     * @param[in] address The pointer to the Buffer Memory
     *
     * @note If a UserBuffer with the same name already exists, the new
     *       address would be updated.
     */
    void add(const char *name, void *address);

    /**
     * @brief Removes a mapping of one Buffer address and its name by its name
     *
     * @param[in] name The name of Memory address to be removed
     *
     * @note If no UserBuffer with the specified name is found, nothing
     *       is done.
     */
    void remove(const char *name) noexcept;

    /**
     * @brief Returns the number of User Memory addresses in the map
     */
    size_t size() const noexcept;

    /**
     * @brief .
     *
     * Removes all User Memory from the map
     */
    void clear() noexcept;

    /**
     * @brief .
     *
     * Returns the names of all User Memory
     *
     * @return A list of Buffer names.
     */
    zdl::DlSystem::StringList getUserBufferNames() const;

    /**
     * @brief Returns the no of UserMemory addresses mapped to the buffer
     *
     * @param[in] name The name of the UserMemory
     *
     */
    size_t getUserMemoryAddressCount(const char *name) const noexcept;

    /**
     * @brief Returns address at a specified index corresponding to a UserMemory buffer name
     *
     * @param[in] name The name of the buffer
     * @param[in] index The index in the list of addresses
     *
     */
    void* getUserMemoryAddressAtIndex(const char *name, uint32_t index) const noexcept;

    ~UserMemoryMap();
private:
    void swap(const UserMemoryMap &other);
    std::unique_ptr<::DlSystem::UserMemoryMapImpl> m_UserMemoryMapImpl;
};
/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

} // DlSystem namespace
} // zdl namespace


#endif // DL_SYSTEM_TENSOR_MAP_HPP

