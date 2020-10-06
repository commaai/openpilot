//=============================================================================
//
//  Copyright (c) 2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "ZdlExportDefine.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/StringList.hpp"
#include <cstddef>
#include <memory>

#ifndef DL_SYSTEM_RUNTIME_LIST_HPP
#define DL_SYSTEM_RUNTIME_LIST_HPP

namespace DlSystem
{
   // Forward declaration of Runtime List implementation.
   class RuntimeListImpl;
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
  * A class representing list of runtimes
  */
class ZDL_EXPORT RuntimeList final
{
public:

    /**
    * @brief .
    *
    * Creates a new runtime list
    *
    */
   RuntimeList();

   /**
   * @brief .
   *
   * copy constructor.
   * @param[in] other object to copy.
   */
   RuntimeList(const RuntimeList& other);

   /**
   * @brief .
   *
   * constructor with single Runtime_t object
   * @param[in] Runtime_t object
   */
   RuntimeList(const zdl::DlSystem::Runtime_t& runtime);

   /**
    * @brief .
    *
    * assignment operator.
    */
   RuntimeList& operator=(const RuntimeList& other);

   /**
    * @brief .
    *
    * subscript operator.
    */
   Runtime_t& operator[](size_t index);

   /**
    * @brief Adds runtime to the end of the runtime list
    *        order of precedence is former followed by latter entry
    *
    * @param[in] runtime to add
    *
    * Ruturns false If the runtime already exists
    */
   bool add(const zdl::DlSystem::Runtime_t& runtime);

   /**
    * @brief Removes the runtime from the list
    *
    * @param[in] runtime to be removed
    *
    * @note If the runtime is not found, nothing is done.
    */
   void remove(const zdl::DlSystem::Runtime_t runtime) noexcept;

   /**
    * @brief Returns the number of runtimes in the list
    */
   size_t size() const noexcept;

   /**
    * @brief Returns true if the list is empty
    */
   bool empty() const noexcept;

   /**
    * @brief .
    *
    * Removes all runtime from the list
    */
   void clear() noexcept;

   /**
    * @brief .
    *
    * Returns a StringList of names from the runtime list in
    * order of precedence
    */
   zdl::DlSystem::StringList getRuntimeListNames() const;

   /**
    * @brief .
    *
    * @param[in] runtime string
    * Returns a Runtime enum corresponding to the in param string
    *
    */
   static zdl::DlSystem::Runtime_t stringToRuntime(const char* runtimeStr);

   /**
    * @brief .
    *
    * @param[in] runtime
    * Returns a string corresponding to the in param runtime enum
    *
    */
   static const char* runtimeToString(const zdl::DlSystem::Runtime_t runtime);

   ~RuntimeList();

private:
   void deepCopy(const RuntimeList &other);
   std::unique_ptr<::DlSystem::RuntimeListImpl> m_RuntimeListImpl;
};

} // DlSystem namespace
} // zdl namespace

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif // DL_SYSTEM_RUNTIME_LIST_HPP

