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

#ifndef ANDROID_VECTOR_H
#define ANDROID_VECTOR_H

#include <stdint.h>
#include <sys/types.h>

#include <log/log.h>
#include <utils/TypeHelpers.h>
#include <utils/VectorImpl.h>

/*
 * Used to blacklist some functions from CFI.
 *
 */
#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#if __has_attribute(no_sanitize)
#define UTILS_VECTOR_NO_CFI __attribute__((no_sanitize("cfi")))
#else
#define UTILS_VECTOR_NO_CFI
#endif

// ---------------------------------------------------------------------------

namespace android {

template <typename TYPE>
class SortedVector;

/*!
 * The main templated vector class ensuring type safety
 * while making use of VectorImpl.
 * This is the class users want to use.
 *
 * DO NOT USE: please use std::vector
 */

template <class TYPE>
class Vector : private VectorImpl
{
public:
            typedef TYPE    value_type;

    /*!
     * Constructors and destructors
     */

                            Vector();
                            Vector(const Vector<TYPE>& rhs);
    explicit                Vector(const SortedVector<TYPE>& rhs);
    virtual                 ~Vector();

    /*! copy operator */
            const Vector<TYPE>&     operator = (const Vector<TYPE>& rhs) const;
            Vector<TYPE>&           operator = (const Vector<TYPE>& rhs);

            const Vector<TYPE>&     operator = (const SortedVector<TYPE>& rhs) const;
            Vector<TYPE>&           operator = (const SortedVector<TYPE>& rhs);

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
     * set the size of the vector. items are appended with the default
     * constructor, or removed from the end as needed.
     */
    inline  ssize_t         resize(size_t size)         { return VectorImpl::resize(size); }

    /*!
     * C-style array access
     */

    //! read-only C-style access
    inline  const TYPE*     array() const;
    //! read-write C-style access
            TYPE*           editArray();

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

    //! copy-on write support, grants write access to an item
            TYPE&           editItemAt(size_t index);
    //! grants right access to the top of the stack (last element)
            TYPE&           editTop();

            /*!
             * append/insert another vector
             */

    //! insert another vector at a given index
            ssize_t         insertVectorAt(const Vector<TYPE>& vector, size_t index);

    //! append another vector at the end of this one
            ssize_t         appendVector(const Vector<TYPE>& vector);


    //! insert an array at a given index
            ssize_t         insertArrayAt(const TYPE* array, size_t index, size_t length);

    //! append an array at the end of this vector
            ssize_t         appendArray(const TYPE* array, size_t length);

            /*!
             * add/insert/replace items
             */

    //! insert one or several items initialized with their default constructor
    inline  ssize_t         insertAt(size_t index, size_t numItems = 1);
    //! insert one or several items initialized from a prototype item
            ssize_t         insertAt(const TYPE& prototype_item, size_t index, size_t numItems = 1);
    //! pop the top of the stack (removes the last element). No-op if the stack's empty
    inline  void            pop();
    //! pushes an item initialized with its default constructor
    inline  void            push();
    //! pushes an item on the top of the stack
            void            push(const TYPE& item);
    //! same as push() but returns the index the item was added at (or an error)
    inline  ssize_t         add();
    //! same as push() but returns the index the item was added at (or an error)
            ssize_t         add(const TYPE& item);
    //! replace an item with a new one initialized with its default constructor
    inline  ssize_t         replaceAt(size_t index);
    //! replace an item with a new one
            ssize_t         replaceAt(const TYPE& item, size_t index);

    /*!
     * remove items
     */

    //! remove several items
    inline  ssize_t         removeItemsAt(size_t index, size_t count = 1);
    //! remove one item
    inline  ssize_t         removeAt(size_t index)  { return removeItemsAt(index); }

    /*!
     * sort (stable) the array
     */

     typedef int (*compar_t)(const TYPE* lhs, const TYPE* rhs);
     typedef int (*compar_r_t)(const TYPE* lhs, const TYPE* rhs, void* state);

     inline status_t        sort(compar_t cmp);
     inline status_t        sort(compar_r_t cmp, void* state);

     // for debugging only
     inline size_t getItemSize() const { return itemSize(); }


     /*
      * these inlines add some level of compatibility with STL. eventually
      * we should probably turn things around.
      */
     typedef TYPE* iterator;
     typedef TYPE const* const_iterator;

     inline iterator begin() { return editArray(); }
     inline iterator end()   { return editArray() + size(); }
     inline const_iterator begin() const { return array(); }
     inline const_iterator end() const   { return array() + size(); }
     inline void reserve(size_t n) { setCapacity(n); }
     inline bool empty() const{ return isEmpty(); }
     inline void push_back(const TYPE& item)  { insertAt(item, size(), 1); }
     inline void push_front(const TYPE& item) { insertAt(item, 0, 1); }
     inline iterator erase(iterator pos) {
         ssize_t index = removeItemsAt(static_cast<size_t>(pos-array()));
         return begin() + index;
     }

protected:
    virtual void    do_construct(void* storage, size_t num) const;
    virtual void    do_destroy(void* storage, size_t num) const;
    virtual void    do_copy(void* dest, const void* from, size_t num) const;
    virtual void    do_splat(void* dest, const void* item, size_t num) const;
    virtual void    do_move_forward(void* dest, const void* from, size_t num) const;
    virtual void    do_move_backward(void* dest, const void* from, size_t num) const;
};

// ---------------------------------------------------------------------------
// No user serviceable parts from here...
// ---------------------------------------------------------------------------

template<class TYPE> inline
Vector<TYPE>::Vector()
    : VectorImpl(sizeof(TYPE),
                ((traits<TYPE>::has_trivial_ctor   ? HAS_TRIVIAL_CTOR   : 0)
                |(traits<TYPE>::has_trivial_dtor   ? HAS_TRIVIAL_DTOR   : 0)
                |(traits<TYPE>::has_trivial_copy   ? HAS_TRIVIAL_COPY   : 0))
                )
{
}

template<class TYPE> inline
Vector<TYPE>::Vector(const Vector<TYPE>& rhs)
    : VectorImpl(rhs) {
}

template<class TYPE> inline
Vector<TYPE>::Vector(const SortedVector<TYPE>& rhs)
    : VectorImpl(static_cast<const VectorImpl&>(rhs)) {
}

template<class TYPE> inline
Vector<TYPE>::~Vector() {
    finish_vector();
}

template<class TYPE> inline
Vector<TYPE>& Vector<TYPE>::operator = (const Vector<TYPE>& rhs) {
    VectorImpl::operator = (rhs);
    return *this;
}

template<class TYPE> inline
const Vector<TYPE>& Vector<TYPE>::operator = (const Vector<TYPE>& rhs) const {
    VectorImpl::operator = (static_cast<const VectorImpl&>(rhs));
    return *this;
}

template<class TYPE> inline
Vector<TYPE>& Vector<TYPE>::operator = (const SortedVector<TYPE>& rhs) {
    VectorImpl::operator = (static_cast<const VectorImpl&>(rhs));
    return *this;
}

template<class TYPE> inline
const Vector<TYPE>& Vector<TYPE>::operator = (const SortedVector<TYPE>& rhs) const {
    VectorImpl::operator = (rhs);
    return *this;
}

template<class TYPE> inline
const TYPE* Vector<TYPE>::array() const {
    return static_cast<const TYPE *>(arrayImpl());
}

template<class TYPE> inline
TYPE* Vector<TYPE>::editArray() {
    return static_cast<TYPE *>(editArrayImpl());
}


template<class TYPE> inline
const TYPE& Vector<TYPE>::operator[](size_t index) const {
    LOG_FATAL_IF(index>=size(),
            "%s: index=%u out of range (%u)", __PRETTY_FUNCTION__,
            int(index), int(size()));
    return *(array() + index);
}

template<class TYPE> inline
const TYPE& Vector<TYPE>::itemAt(size_t index) const {
    return operator[](index);
}

template<class TYPE> inline
const TYPE& Vector<TYPE>::top() const {
    return *(array() + size() - 1);
}

template<class TYPE> inline
TYPE& Vector<TYPE>::editItemAt(size_t index) {
    return *( static_cast<TYPE *>(editItemLocation(index)) );
}

template<class TYPE> inline
TYPE& Vector<TYPE>::editTop() {
    return *( static_cast<TYPE *>(editItemLocation(size()-1)) );
}

template<class TYPE> inline
ssize_t Vector<TYPE>::insertVectorAt(const Vector<TYPE>& vector, size_t index) {
    return VectorImpl::insertVectorAt(reinterpret_cast<const VectorImpl&>(vector), index);
}

template<class TYPE> inline
ssize_t Vector<TYPE>::appendVector(const Vector<TYPE>& vector) {
    return VectorImpl::appendVector(reinterpret_cast<const VectorImpl&>(vector));
}

template<class TYPE> inline
ssize_t Vector<TYPE>::insertArrayAt(const TYPE* array, size_t index, size_t length) {
    return VectorImpl::insertArrayAt(array, index, length);
}

template<class TYPE> inline
ssize_t Vector<TYPE>::appendArray(const TYPE* array, size_t length) {
    return VectorImpl::appendArray(array, length);
}

template<class TYPE> inline
ssize_t Vector<TYPE>::insertAt(const TYPE& item, size_t index, size_t numItems) {
    return VectorImpl::insertAt(&item, index, numItems);
}

template<class TYPE> inline
void Vector<TYPE>::push(const TYPE& item) {
    return VectorImpl::push(&item);
}

template<class TYPE> inline
ssize_t Vector<TYPE>::add(const TYPE& item) {
    return VectorImpl::add(&item);
}

template<class TYPE> inline
ssize_t Vector<TYPE>::replaceAt(const TYPE& item, size_t index) {
    return VectorImpl::replaceAt(&item, index);
}

template<class TYPE> inline
ssize_t Vector<TYPE>::insertAt(size_t index, size_t numItems) {
    return VectorImpl::insertAt(index, numItems);
}

template<class TYPE> inline
void Vector<TYPE>::pop() {
    VectorImpl::pop();
}

template<class TYPE> inline
void Vector<TYPE>::push() {
    VectorImpl::push();
}

template<class TYPE> inline
ssize_t Vector<TYPE>::add() {
    return VectorImpl::add();
}

template<class TYPE> inline
ssize_t Vector<TYPE>::replaceAt(size_t index) {
    return VectorImpl::replaceAt(index);
}

template<class TYPE> inline
ssize_t Vector<TYPE>::removeItemsAt(size_t index, size_t count) {
    return VectorImpl::removeItemsAt(index, count);
}

template<class TYPE> inline
status_t Vector<TYPE>::sort(Vector<TYPE>::compar_t cmp) {
    return VectorImpl::sort(reinterpret_cast<VectorImpl::compar_t>(cmp));
}

template<class TYPE> inline
status_t Vector<TYPE>::sort(Vector<TYPE>::compar_r_t cmp, void* state) {
    return VectorImpl::sort(reinterpret_cast<VectorImpl::compar_r_t>(cmp), state);
}

// ---------------------------------------------------------------------------

template<class TYPE>
UTILS_VECTOR_NO_CFI void Vector<TYPE>::do_construct(void* storage, size_t num) const {
    construct_type( reinterpret_cast<TYPE*>(storage), num );
}

template<class TYPE>
void Vector<TYPE>::do_destroy(void* storage, size_t num) const {
    destroy_type( reinterpret_cast<TYPE*>(storage), num );
}

template<class TYPE>
UTILS_VECTOR_NO_CFI void Vector<TYPE>::do_copy(void* dest, const void* from, size_t num) const {
    copy_type( reinterpret_cast<TYPE*>(dest), reinterpret_cast<const TYPE*>(from), num );
}

template<class TYPE>
UTILS_VECTOR_NO_CFI void Vector<TYPE>::do_splat(void* dest, const void* item, size_t num) const {
    splat_type( reinterpret_cast<TYPE*>(dest), reinterpret_cast<const TYPE*>(item), num );
}

template<class TYPE>
UTILS_VECTOR_NO_CFI void Vector<TYPE>::do_move_forward(void* dest, const void* from, size_t num) const {
    move_forward_type( reinterpret_cast<TYPE*>(dest), reinterpret_cast<const TYPE*>(from), num );
}

template<class TYPE>
UTILS_VECTOR_NO_CFI void Vector<TYPE>::do_move_backward(void* dest, const void* from, size_t num) const {
    move_backward_type( reinterpret_cast<TYPE*>(dest), reinterpret_cast<const TYPE*>(from), num );
}

}; // namespace android


// ---------------------------------------------------------------------------

#endif // ANDROID_VECTOR_H
