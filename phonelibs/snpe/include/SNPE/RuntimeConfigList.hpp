//==============================================================================
//
//  Copyright (c) 2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#ifndef PSNPE_RUNTIMECONFIGLIST_HPP
#define PSNPE_RUNTIMECONFIGLIST_HPP

#include <iostream>
#include "DlSystem/DlEnums.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/ZdlExportDefine.hpp"
#include "DlSystem/RuntimeList.hpp"

namespace zdl {
namespace PSNPE
{

/** @addtogroup c_plus_plus_apis C++
@{ */

/**
  * @brief .
  *
  * The structure for configuring a BulkSNPE runtime
  *
  */
struct ZDL_EXPORT RuntimeConfig final {
   zdl::DlSystem::Runtime_t runtime;
   zdl::DlSystem::RuntimeList runtimeList;
   zdl::DlSystem::PerformanceProfile_t perfProfile;
   bool enableCPUFallback;
   RuntimeConfig(): runtime{zdl::DlSystem::Runtime_t::CPU_FLOAT32},
                    perfProfile{zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE},
                    enableCPUFallback{false}
   {}
   RuntimeConfig(const RuntimeConfig& other)
   {
       runtime           = other.runtime;
       runtimeList       = other.runtimeList;
       perfProfile       = other.perfProfile;
       enableCPUFallback = other.enableCPUFallback;
   }

   RuntimeConfig& operator=(const RuntimeConfig &other)
   {
       this->runtimeList       = other.runtimeList;
       this->runtime           = other.runtime;
       this->perfProfile       = other.perfProfile;
       this->enableCPUFallback = other.enableCPUFallback;
       return *this;
   }

   ~RuntimeConfig() {}

};

/**
* @brief .
*
* The class for creating a RuntimeConfig container.
*
*/
class ZDL_EXPORT RuntimeConfigList final
{
public:
   RuntimeConfigList();
   RuntimeConfigList(const size_t size);
   void push_back(const RuntimeConfig &runtimeConfig);
   RuntimeConfig& operator[](const size_t index);
   RuntimeConfigList& operator =(const RuntimeConfigList &other);
   size_t size() const noexcept;
   size_t capacity() const noexcept;
   void clear() noexcept;
   ~RuntimeConfigList() = default;

private:
   void swap(const RuntimeConfigList &other);
   std::vector<RuntimeConfig> m_runtimeConfigs;

};
/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

} // namespace PSNPE
} // namespace zdl
#endif //PSNPE_RUNTIMECONFIGLIST_HPP
