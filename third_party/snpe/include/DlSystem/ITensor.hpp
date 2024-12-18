//=============================================================================
//
//  Copyright (c) 2015-2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef _ITENSOR_HPP_
#define _ITENSOR_HPP_

#include "ITensorItr.hpp"
#include "ITensorItrImpl.hpp"
#include "TensorShape.hpp"
#include "ZdlExportDefine.hpp"
#include <memory>
#include <ostream>
#include <cmath>

namespace zdl {
namespace DlSystem
{
   class ITensor;
}}

namespace zdl { namespace DlSystem
{
/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * Represents a tensor which holds n-dimensional data. It is important to
 * understand how the tensor data is represented in memory 
 * relative to the tensor dimensions. Tensors store data in 
 * memory in row-major order (i.e. the last tensor dimension is 
 * the fastest varying one). For example, if you have a two 
 * dimensional tensor with 3 rows and 2 columns (i.e. the tensor 
 * dimensions are 3,2 as returned in tensor dimension vectors)
 * with the following data in terms rows and columns: 
 *  
 * | 1 2 | <br/>
 * | 3 4 | <br/>
 * | 5 6 | <br/>
 *  
 * This data would be stored in memory as 1,2,3,4,5,6. 
 */
class ZDL_EXPORT ITensor 
{
public:

   typedef zdl::DlSystem::ITensorItr<false> iterator;
   typedef zdl::DlSystem::ITensorItr<true> const_iterator;

   virtual ~ITensor() {}

   /**
    * Returns a tensor iterator pointing to the beginning 
    * of the data in the tensor.
    * 
    * @return A tensor iterator that points to the first data
    *         element in the tensor.
    */
   virtual iterator begin() = 0;

   /**
    * Returns the const version of a tensor iterator 
    * pointing to the beginning of the data in the tensor.
    * 
    * @return A tensor const iterator that points to the first data
    *         element in the tensor.
    */
   virtual const_iterator cbegin() const = 0;

   /**
    * Returns a tensor iterator pointing to the end of the
    * data in the tensor. This tensor should not be
    * dereferenced.
    * 
    * @return A tensor iterator that points to the end of the data 
    *         (one past the last element) in the tensor.
    */
   virtual iterator end() = 0;

   /**
    * Returns the const version of a tensor iterator 
    * pointing to the end of the data in the tensor. This
    * tensor should not be dereferenced.
    * 
    * @return A tensor const iterator that points to the end of the
    *         data (one past the last element) in the tensor.
    */
   virtual const_iterator cend() const = 0;

   /**
    * @brief Gets the shape of this tensor.
    *  
    * The last element of the vector represents the fastest varying
    * dimension and the zeroth element represents the slowest 
    * varying dimension, etc. 
    * 
    * @return A shape class holding the tensor dimensions.
    */
   virtual TensorShape getShape() const = 0;

   /**
    * Returns the element size of the data in the tensor 
    * (discounting strides). This is how big a buffer would
    * need to be to hold the tensor data contiguously in
    * memory.
    *  
    * @return The size of the tensor (in elements).
    */
   virtual size_t getSize() const = 0;

   /**
    * @brief Serializes the tensor to an output stream.
    * 
    * @param[in] output The output stream to which to write the tensor 
    *  
    * @throw std::runtime_error If the stream is ever in a bad 
    *        state before the tensor is fully serialized.
    */
   virtual void serialize(std::ostream &output) const = 0;

   friend iterator;
   friend const_iterator;

   virtual bool isQuantized() {return false;}
   virtual float GetDelta() {return NAN;};
   virtual float GetOffset() {return NAN;};

protected:

   /**
    * Returns the tensor iterator implementation.
    * 
    * @return A pointer to the tensor iterator implementation.
    */
   virtual std::unique_ptr<::DlSystem::ITensorItrImpl> getItrImpl() const = 0;
};

}}

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif
