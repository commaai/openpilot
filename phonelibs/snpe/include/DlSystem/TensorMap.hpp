//=============================================================================
//
//  Copyright (c) 2016 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#include <memory>
#include "ZdlExportDefine.hpp"
#include "ITensor.hpp"
#include "StringList.hpp"

#ifndef DL_SYSTEM_TENSOR_MAP_HPP
#define DL_SYSTEM_TENSOR_MAP_HPP

namespace DlSystem
{
   // Forward declaration of tensor map implementation.
   class TensorMapImpl;
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
  * A class representing the map of tensor.
  */
class ZDL_EXPORT TensorMap final
{
public:

  /**
   * @brief .
   *
   * Creates a new empty tensor map
   */
   TensorMap();

  /**
   * copy constructor.
   * @param[in] other object to copy. 
   */
   TensorMap(const TensorMap& other);

  /**
    * assignment operator. 
    */
   TensorMap& operator=(const TensorMap& other);

   /**
    * @brief Adds a name and the corresponding tensor pointer
    *        to the map
    *
    * @param[in] name The name of the tensor
    * @param[out] tensor The pointer to the tensor
    *
    * @note If a tensor with the same name already exists, the
    *       tensor is replaced with the existing tensor.
    */
   void add(const char *name, zdl::DlSystem::ITensor *tensor);

   /**
    * @brief Removes a mapping of tensor and its name by its name
    *
    * @param[in] name The name of tensor to be removed
    *
    * @note If no tensor with the specified name is found, nothing
    *       is done.
    */
   void remove(const char *name) noexcept;

   /**
    * @brief Returns the number of tensors in the map
    */
   size_t size() const noexcept;

   /**
    * @brief .
    *
    * Removes all tensors from the map
    */
   void clear() noexcept;

   /**
    * @brief Returns the tensor given its name.
    *  
    * @param[in] name The name of the tensor to get. 
    *  
    * @return nullptr if no tensor with the specified name is 
    *         found; otherwise, a valid pointer to the tensor.
    */
   zdl::DlSystem::ITensor* getTensor(const char *name) const noexcept;

   /**
    * @brief .
    *
    * Returns the names of all tensors
    */
   zdl::DlSystem::StringList getTensorNames() const;

   ~TensorMap();
private:
   void swap(const TensorMap &other);
   std::unique_ptr<::DlSystem::TensorMapImpl> m_TensorMapImpl;
};

} // DlSystem namespace
} // zdl namespace

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif // DL_SYSTEM_TENSOR_MAP_HPP

