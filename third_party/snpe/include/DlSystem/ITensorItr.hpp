//=============================================================================
//
//  Copyright (c) 2015 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef _ITENSOR_ITR_HPP_
#define _ITENSOR_ITR_HPP_

#include "ZdlExportDefine.hpp"
#include "ITensorItrImpl.hpp"

#include <memory>
#include <iterator>
#include <iostream>

namespace zdl {
namespace DlSystem
{
   template<bool IS_CONST> class ITensorItr;
   class ITensor;
   void ZDL_EXPORT fill(ITensorItr<false> first, ITensorItr<false> end, float val);
   template<class InItr, class OutItr> OutItr ZDL_EXPORT copy(InItr first, InItr last, OutItr result)
   {
      return std::copy(first, last, result);
   }
}}
namespace DlSystem
{
   class ITensorItrImpl;
}

namespace zdl { namespace DlSystem
{

/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * A bidirectional iterator (with limited random access 
 * capabilities) for the zdl::DlSystem::ITensor class.
 *  
 * This is a standard bidrectional iterator and is compatible
 * with standard algorithm functions that operate on bidirectional 
 * access iterators (e.g., std::copy, std::fill, etc.). It uses a
 * template parameter to create const and non-const iterators 
 * from the same code. Iterators are easiest to declare via the 
 * typedefs iterator and const_iterator in the ITensor class 
 * (e.g., zdl::DlSystem::ITensor::iterator). 
 *  
 * Note that if the tensor the iterator is traversing was 
 * created with nondefault (i.e., nontrivial) strides, the 
 * iterator will obey the strides when traversing the tensor 
 * data. 
 *  
 * Also note that nontrivial strides dramatically affect the
 * performance of the iterator (on the order of 20x slower). 
 */ 
template<bool IS_CONST=true>
class ZDL_EXPORT ITensorItr : public std::iterator<std::bidirectional_iterator_tag, float>
{
public:

   typedef typename std::conditional<IS_CONST, const float&, float&>::type VALUE_REF;

   ITensorItr() = delete;
   virtual ~ITensorItr() {}

   ITensorItr(std::unique_ptr<::DlSystem::ITensorItrImpl> impl, 
              bool isTrivial = false, 
              float* data = nullptr)
      : m_Impl(impl->clone())
      , m_IsTrivial(isTrivial)
      , m_Data(data)
      , m_DataStart(data) {}

   ITensorItr(const ITensorItr<IS_CONST>& itr)
      : m_Impl(itr.m_Impl->clone()),
        m_IsTrivial(itr.m_IsTrivial),
        m_Data(itr.m_Data),
        m_DataStart(itr.m_DataStart) {}

   zdl::DlSystem::ITensorItr<IS_CONST>& operator=(const ITensorItr<IS_CONST>& other)
   {
      if (this == &other) return *this;
      m_Impl = std::move(other.m_Impl->clone());
      m_IsTrivial = other.m_IsTrivial;
      m_Data = other.m_Data;
      m_DataStart = other.m_DataStart;
      return *this;
   }

   inline zdl::DlSystem::ITensorItr<IS_CONST>& operator++()
   {
      if (m_IsTrivial) m_Data++; else m_Impl->increment();
      return *this;
   }
   inline zdl::DlSystem::ITensorItr<IS_CONST> operator++(int)
   {
      ITensorItr tmp(*this);
      operator++();
      return tmp;
   }
   inline zdl::DlSystem::ITensorItr<IS_CONST>& operator--()
   {
      if (m_IsTrivial) m_Data--; else m_Impl->decrement();
      return *this;
   }
   inline zdl::DlSystem::ITensorItr<IS_CONST> operator--(int)
   {
      ITensorItr tmp(*this);
      operator--();
      return tmp;
   }
   inline zdl::DlSystem::ITensorItr<IS_CONST>& operator+=(int rhs)
   {
      if (m_IsTrivial) m_Data += rhs; else m_Impl->increment(rhs);
      return *this;
   }
   inline friend zdl::DlSystem::ITensorItr<IS_CONST> operator+(zdl::DlSystem::ITensorItr<IS_CONST> lhs, int rhs)
      { lhs += rhs; return lhs; }
   inline zdl::DlSystem::ITensorItr<IS_CONST>& operator-=(int rhs)
   {
      if (m_IsTrivial) m_Data -= rhs; else m_Impl->decrement(rhs);
      return *this;
   }
   inline friend zdl::DlSystem::ITensorItr<IS_CONST> operator-(zdl::DlSystem::ITensorItr<IS_CONST> lhs, int rhs)
      { lhs -= rhs; return lhs; }

   inline size_t operator-(const zdl::DlSystem::ITensorItr<IS_CONST>& rhs)
   {
      if (m_IsTrivial) return (m_Data - m_DataStart) - (rhs.m_Data - rhs.m_DataStart);
      return m_Impl->getPosition() - rhs.m_Impl->getPosition();
   }

   inline friend bool operator<(const ITensorItr<IS_CONST>& lhs, const ITensorItr<IS_CONST>& rhs)
   {
      if (lhs.m_IsTrivial) return lhs.m_Data < rhs.m_Data;
      return lhs.m_Impl->dataPointer() < rhs.m_Impl->dataPointer();
   }
   inline friend bool operator>(const ITensorItr<IS_CONST>& lhs, const ITensorItr<IS_CONST>& rhs)
      { return rhs < lhs; }
   inline friend bool operator<=(const ITensorItr<IS_CONST>& lhs, const ITensorItr<IS_CONST>& rhs)
      { return !(lhs > rhs); }
   inline friend bool operator>=(const ITensorItr<IS_CONST>& lhs, const ITensorItr<IS_CONST>& rhs)
      { return !(lhs < rhs); }

   inline bool operator==(const ITensorItr<IS_CONST>& rhs) const
   {
      if (m_IsTrivial) return m_Data == rhs.m_Data;
      return m_Impl->dataPointer() == rhs.m_Impl->dataPointer();
   }
   inline bool operator!=(const ITensorItr<IS_CONST>& rhs) const
      { return !operator==(rhs); }

   inline VALUE_REF operator[](size_t idx)
   {
      if (m_IsTrivial) return *(m_DataStart + idx);
      return m_Impl->getReferenceAt(idx);
   }
   inline VALUE_REF operator*()
      { if (m_IsTrivial) return *m_Data; else return m_Impl->getReference(); }
   inline VALUE_REF operator->()
      { return *(*this); }
   inline float* dataPointer() const
      { if (m_IsTrivial) return m_Data; else return m_Impl->dataPointer(); }


protected:
   std::unique_ptr<::DlSystem::ITensorItrImpl> m_Impl;
   bool m_IsTrivial = false;
   float* m_Data = nullptr;
   float* m_DataStart = nullptr;
};

}}

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif
