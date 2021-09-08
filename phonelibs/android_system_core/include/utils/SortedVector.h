/*
 * Copyright (C) 2005 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_SORTED_VECTOR_H
#define ANDROID_SORTED_VECTOR_H

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>

#include <cutils/log.h>

#include <utils/Vector.h>
#include <utils/VectorImpl.h>
#include <utils/TypeHelpers.h>

// ---------------------------------------------------------------------------

namespace android {

template <class TYPE>
class SortedVector : private SortedVectorImpl
{
    friend class Vector<TYPE>;

public:
            typedef TYPE    value_type;
    
    /*! 
     * Constructors and destructors
     */
    
                            SortedVector();
                            SortedVector(const SortedVector<TYPE>& rhs);
    virtual                 ~SortedVector();

    /*! copy operator */
    const SortedVector<TYPE>&   operator = (const SortedVector<TYPE>& rhs) const;    
    SortedVector<TYPE>&         operator = (const SortedVector<TYPE>& rhs);    

    /*
     * empty the vector
     */

    inline  void            clear()             { VectorImpl::clear(); }

    /*! 
     * vector stats
     */

    //! returns number of items in the vector
    inline  size_t          size() const                { return VectorImpl::size(); }
    //! returns whether or not the vector is empty
    inline  bool            isEmpty() const             { return VectorImpl::isEmpty(); }
    //! returns how many items can be stored without reallocating the backing store
    inline  size_t          capacity() const            { return VectorImpl::capacity(); }
    //! sets the capacity. capacity can never be reduced less than size()
    inline  ssize_t         setCapacity(size_t size)    { return VectorImpl::setCapacity(size); }

    /*! 
     * C-style array access
     */
     
    //! read-only C-style access 
    inline  const TYPE*     array() const;

    //! read-write C-style access. BE VERY CAREFUL when modifying the array
    //! you must keep it sorted! You usually don't use this function.
            TYPE*           editArray();

            //! finds the index of an item
            ssize_t         indexOf(const TYPE& item) const;
            
            //! finds where this item should be inserted
            size_t          orderOf(const TYPE& item) const;
            
    
    /*! 
     * accessors
     */

    //! read-only access to an item at a given index
    inline  const TYPE&     operator [] (size_t index) const;
    //! alternate name for operator []
    inline  const TYPE&     itemAt(size_t index) const;
    //! stack-usage of the vector. returns the top of the stack (last element)
            const TYPE&     top() const;

    /*!
     * modifying the array
     */

            //! add an item in the right place (and replace the one that is there)
            ssize_t         add(const TYPE& item);
            
            //! editItemAt() MUST NOT change the order of this item
            TYPE&           editItemAt(size_t index) {
                return *( static_cast<TYPE *>(VectorImpl::editItemLocation(index)) );
            }

            //! merges a vector into this one
            ssize_t         merge(const Vector<TYPE>& vector);
            ssize_t         merge(const SortedVector<TYPE>& vector);
            
            //! removes an item
            ssize_t         remove(const TYPE&);

    //! remove several items
    inline  ssize_t         removeItemsAt(size_t index, size_t count = 1);
    //! remove one item
    inline  ssize_t         removeAt(size_t index)  { return removeItemsAt(index); }
            
protected:
    virtual void    do_construct(void* storage, size_t num) const;
    virtual void    do_destroy(void* storage, size_t num) const;
    virtual void    do_copy(void* dest, const void* from, size_t num) const;
    virtual void    do_splat(void* dest, const void* item, size_t num) const;
    virtual void    do_move_forward(void* dest, const void* from, size_t num) const;
    virtual void    do_move_backward(void* dest, const void* from, size_t num) const;
    virtual int     do_compare(const void* lhs, const void* rhs) const;
};

// SortedVector<T> can be trivially moved using memcpy() because moving does not
// require any change to the underlying SharedBuffer contents or reference count.
template<typename T> struct trait_trivial_move<SortedVector<T> > { enum { value = true }; };

// ---------------------------------------------------------------------------
// No user serviceable parts from here...
// ---------------------------------------------------------------------------

template<class TYPE> inline
SortedVector<TYPE>::SortedVector()
    : SortedVectorImpl(sizeof(TYPE),
                ((traits<TYPE>::has_trivial_ctor   ? HAS_TRIVIAL_CTOR   : 0)
                |(traits<TYPE>::has_trivial_dtor   ? HAS_TRIVIAL_DTOR   : 0)
                |(traits<TYPE>::has_trivial_copy   ? HAS_TRIVIAL_COPY   : 0))
                )
{
}

template<class TYPE> inline
SortedVector<TYPE>::SortedVector(const SortedVector<TYPE>& rhs)
    : SortedVectorImpl(rhs) {
}

template<class TYPE> inline
SortedVector<TYPE>::~SortedVector() {
    finish_vector();
}

template<class TYPE> inline
SortedVector<TYPE>& SortedVector<TYPE>::operator = (const SortedVector<TYPE>& rhs) {
    SortedVectorImpl::operator = (rhs);
    return *this; 
}

template<class TYPE> inline
const SortedVector<TYPE>& SortedVector<TYPE>::operator = (const SortedVector<TYPE>& rhs) const {
    SortedVectorImpl::operator = (rhs);
    return *this; 
}

template<class TYPE> inline
const TYPE* SortedVector<TYPE>::array() const {
    return static_cast<const TYPE *>(arrayImpl());
}

template<class TYPE> inline
TYPE* SortedVector<TYPE>::editArray() {
    return static_cast<TYPE *>(editArrayImpl());
}


template<class TYPE> inline
const TYPE& SortedVector<TYPE>::operator[](size_t index) const {
    LOG_FATAL_IF(index>=size(),
            "%s: index=%u out of range (%u)", __PRETTY_FUNCTION__,
            int(index), int(size()));
    return *(array() + index);
}

template<class TYPE> inline
const TYPE& SortedVector<TYPE>::itemAt(size_t index) const {
    return operator[](index);
}

template<class TYPE> inline
const TYPE& SortedVector<TYPE>::top() const {
    return *(array() + size() - 1);
}

template<class TYPE> inline
ssize_t SortedVector<TYPE>::add(const TYPE& item) {
    return SortedVectorImpl::add(&item);
}

template<class TYPE> inline
ssize_t SortedVector<TYPE>::indexOf(const TYPE& item) const {
    return SortedVectorImpl::indexOf(&item);
}

template<class TYPE> inline
size_t SortedVector<TYPE>::orderOf(const TYPE& item) const {
    return SortedVectorImpl::orderOf(&item);
}

template<class TYPE> inline
ssize_t SortedVector<TYPE>::merge(const Vector<TYPE>& vector) {
    return SortedVectorImpl::merge(reinterpret_cast<const VectorImpl&>(vector));
}

template<class TYPE> inline
ssize_t SortedVector<TYPE>::merge(const SortedVector<TYPE>& vector) {
    return SortedVectorImpl::merge(reinterpret_cast<const SortedVectorImpl&>(vector));
}

template<class TYPE> inline
ssize_t SortedVector<TYPE>::remove(const TYPE& item) {
    return SortedVectorImpl::remove(&item);
}

template<class TYPE> inline
ssize_t SortedVector<TYPE>::removeItemsAt(size_t index, size_t count) {
    return VectorImpl::removeItemsAt(index, count);
}

// ---------------------------------------------------------------------------

template<class TYPE>
void SortedVector<TYPE>::do_construct(void* storage, size_t num) const {
    construct_type( reinterpret_cast<TYPE*>(storage), num );
}

template<class TYPE>
void SortedVector<TYPE>::do_destroy(void* storage, size_t num) const {
    destroy_type( reinterpret_cast<TYPE*>(storage), num );
}

template<class TYPE>
void SortedVector<TYPE>::do_copy(void* dest, const void* from, size_t num) const {
    copy_type( reinterpret_cast<TYPE*>(dest), reinterpret_cast<const TYPE*>(from), num );
}

template<class TYPE>
void SortedVector<TYPE>::do_splat(void* dest, const void* item, size_t num) const {
    splat_type( reinterpret_cast<TYPE*>(dest), reinterpret_cast<const TYPE*>(item), num );
}

template<class TYPE>
void SortedVector<TYPE>::do_move_forward(void* dest, const void* from, size_t num) const {
    move_forward_type( reinterpret_cast<TYPE*>(dest), reinterpret_cast<const TYPE*>(from), num );
}

template<class TYPE>
void SortedVector<TYPE>::do_move_backward(void* dest, const void* from, size_t num) const {
    move_backward_type( reinterpret_cast<TYPE*>(dest), reinterpret_cast<const TYPE*>(from), num );
}

template<class TYPE>
int SortedVector<TYPE>::do_compare(const void* lhs, const void* rhs) const {
    return compare_type( *reinterpret_cast<const TYPE*>(lhs), *reinterpret_cast<const TYPE*>(rhs) );
}

}; // namespace android


// ---------------------------------------------------------------------------

#endif // ANDROID_SORTED_VECTOR_H
