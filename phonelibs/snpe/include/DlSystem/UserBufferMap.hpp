//=============================================================================
//
//  Copyright (c) 2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#include <memory>
#include "ZdlExportDefine.hpp"
#include "StringList.hpp"

#ifndef DL_SYSTEM_USER_BUFFER_MAP_HPP
#define DL_SYSTEM_USER_BUFFER_MAP_HPP

namespace DlSystem
{
    // Forward declaration of UserBuffer map implementation.
    class UserBufferMapImpl;
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
  * A class representing the map of UserBuffer.
  */
class ZDL_EXPORT UserBufferMap final
{
public:

    /**
     * @brief .
     *
     * Creates a new empty UserBuffer map
     */
    UserBufferMap();

    /**
     * copy constructor.
     * @param[in] other object to copy.
     */
    UserBufferMap(const UserBufferMap& other);

    /**
      * assignment operator.
      */
    UserBufferMap& operator=(const UserBufferMap& other);

    /**
     * @brief Adds a name and the corresponding UserBuffer pointer
     *        to the map
     *
     * @param[in] name The name of the UserBuffer
     * @param[in] userBuffer The pointer to the UserBuffer
     *
     * @note If a UserBuffer with the same name already exists, the new
     *       UserBuffer pointer would be updated.
     */
    void add(const char *name, zdl::DlSystem::IUserBuffer *buffer);

    /**
     * @brief Removes a mapping of one UserBuffer and its name by its name
     *
     * @param[in] name The name of UserBuffer to be removed
     *
     * @note If no UserBuffer with the specified name is found, nothing
     *       is done.
     */
    void remove(const char *name) noexcept;

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
    zdl::DlSystem::IUserBuffer* getUserBuffer(const char *name) const noexcept;

    /**
     * @brief .
     *
     * Returns the names of all UserBuffers
     *
     * @return A list of UserBuffer names.
     */
    zdl::DlSystem::StringList getUserBufferNames() const;

    ~UserBufferMap();
private:
    void swap(const UserBufferMap &other);
    std::unique_ptr<::DlSystem::UserBufferMapImpl> m_UserBufferMapImpl;
};
/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

} // DlSystem namespace
} // zdl namespace


#endif // DL_SYSTEM_TENSOR_MAP_HPP

