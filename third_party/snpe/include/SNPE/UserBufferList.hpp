//==============================================================================
//
//  Copyright (c) 2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#ifndef PSNPE_USERBUFFERLIST_HPP
#define PSNPE_USERBUFFERLIST_HPP

#include <vector>
#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/ZdlExportDefine.hpp"

namespace zdl {
namespace PSNPE
{

/** @addtogroup c_plus_plus_apis C++
@{ */
/**
* @brief .
*
* The class for creating a UserBufferMap container.
*
*/
class ZDL_EXPORT UserBufferList final
{
public:
   UserBufferList();
   UserBufferList(const size_t size);
   void push_back(const zdl::DlSystem::UserBufferMap &userBufferMap);
   zdl::DlSystem::UserBufferMap& operator[](const size_t index);
   UserBufferList& operator =(const UserBufferList &other);
   size_t size() const noexcept;
   size_t capacity() const noexcept;
   void clear() noexcept;
   ~UserBufferList() = default;

private:
   void swap(const UserBufferList &other);
   std::vector<zdl::DlSystem::UserBufferMap> m_userBufferMaps;

};
/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

} // namespace PSNPE
} // namespace zdl
#endif //PSNPE_USERBUFFERLIST_HPP
