//=============================================================================
//
//  Copyright (c) 2016 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#include <initializer_list>
#include <cstdio>
#include <memory>
#include <vector>
#include "ZdlExportDefine.hpp"

#ifndef DL_SYSTEM_TENSOR_SHAPE_HPP
#define DL_SYSTEM_TENSOR_SHAPE_HPP

namespace DlSystem
{
   // Forward declaration of tensor shape implementation.
   class TensorShapeImpl;
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
 * Convenient typedef to represent dimension
 */
using Dimension = size_t;

/**
  * @brief .
  *
  * A class representing the shape of tensor. It is used at the
  * time of creation of tensor.
  */
class ZDL_EXPORT TensorShape final
{
public:

    /**
    * @brief .
    *
    * Creates a new shape with a list of dims specified in
    * initializer list fashion.
    *
    * @param[in] dims The dimensions are specified in which the last
    * element of the vector represents the fastest varying
    * dimension and the zeroth element represents the slowest
    * varying, etc.
    *
    */
   TensorShape(std::initializer_list<Dimension> dims);

   /**
    * @brief .
    *
    * Creates a new shape with a list of dims specified in array
    *
    * @param[in] dims The dimensions are specified in which the last
    * element of the vector represents the fastest varying
    * dimension and the zeroth element represents the slowest
    * varying, etc.
    *
    * @param[in] size Size of the array.
    *
    */
   TensorShape(const Dimension *dims, size_t size);

    /**
    * @brief .
    *
    * Creates a new shape with a vector of dims specified in
    * vector fashion.
    *
    * @param[in] dims The dimensions are specified in which the last
    * element of the vector represents the fastest varying
    * dimension and the zeroth element represents the slowest
    * varying, etc.
    * 
    */   
   TensorShape(std::vector<Dimension> dims);

   /**
   * @brief .
   *   
   * copy constructor.
   * @param[in] other object to copy. 
   */   
   TensorShape(const TensorShape& other);

   /**
    * @brief .
    *  
    * assignment operator. 
    */   
   TensorShape& operator=(const TensorShape& other);

    /**
    * @brief .
    *
    * Creates a new shape with no dims. It can be extended later
    * by invoking concatenate.
    */
   TensorShape();

  /**
    * @brief .
    *
    * Concatenates additional dimensions specified in 
    * initializer list fashion to the existing dimensions. 
    *
    * @param[in] dims The dimensions are specified in which the last
    * element of the vector represents the fastest varying
    * dimension and the zeroth element represents the slowest
    * varying, etc.
    *
   */
   void concatenate(std::initializer_list<Dimension> dims);

   /**
    * @brief .
    *
    * Concatenates additional dimensions specified in 
    * the array to the existing dimensions. 
    *
    * @param[in] dims The dimensions are specified in which the last
    * element of the vector represents the fastest varying
    * dimension and the zeroth element represents the slowest
    * varying, etc.
    * 
    * @param[in] size Size of the array.
    *
   */
   void concatenate(const Dimension *dims, size_t size);

  /**
    * @brief .
    *
    * Concatenates an additional dimension to the existing
    * dimensions.
    *
    * @param[in] dim The dimensions are specified in which the last element
    * of the vector represents the fastest varying dimension and the
    * zeroth element represents the slowest varying, etc.
    *
   */
   void concatenate(const Dimension &dim);

  /**
    * @brief .
    *
    * Retrieves a single dimension, based on its index.
    *
    * @return The value of dimension
    *
    * @throws std::out_of_range if the index is >= the number of
    * dimensions (or rank).
    */
   Dimension& operator[](size_t index);
   Dimension& operator[](size_t index) const;

  /**
    * @brief .
    *
    * Retrieves the rank i.e. number of dimensions.
    *
    * @return The rank
    */
   size_t rank() const;

  /**
    * @brief .
    *
    * Retrieves a pointer to the first dimension of shape
    *
    * @return nullptr if no dimension exists; otherwise, points to
    * the first dimension. 
    *
    */
   const Dimension* getDimensions() const;

   ~TensorShape();

private:
   void swap(const TensorShape &other);
   std::unique_ptr<::DlSystem::TensorShapeImpl> m_TensorShapeImpl;
};

} // DlSystem namespace
} // zdl namespace

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif // DL_SYSTEM_TENSOR_SHAPE_HPP

