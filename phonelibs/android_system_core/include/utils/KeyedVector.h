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

#ifndef ANDROID_KEYED_VECTOR_H
#define ANDROID_KEYED_VECTOR_H

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>

#include <cutils/log.h>

#include <utils/SortedVector.h>
#include <utils/TypeHelpers.h>
#include <utils/Errors.h>

// ---------------------------------------------------------------------------

namespace android {

template <typename KEY, typename VALUE>
class KeyedVector
{
public:
    typedef KEY    key_type;
    typedef VALUE  value_type;

    inline                  KeyedVector();

    /*
     * empty the vector
     */

    inline  void            clear()                     { mVector.clear(); }

    /*! 
     * vector stats
     */

    //! returns number of items in the vector
    inline  size_t          size() const                { return mVector.size(); }
    //! returns whether or not the vector is empty
    inline  bool            isEmpty() const             { return mVector.isEmpty(); }
    //! returns how many items can be stored without reallocating the backing store
    inline  size_t          capacity() const            { return mVector.capacity(); }
    //! sets the capacity. capacity can never be reduced less than size()
    inline ssize_t          setCapacity(size_t size)    { return mVector.setCapacity(size); }

    // returns true if the arguments is known to be identical to this vector
    inline bool isIdenticalTo(const KeyedVector& rhs) const;

    /*! 
     * accessors
     */
            const VALUE&    valueFor(const KEY& key) const;
            const VALUE&    valueAt(size_t index) const;
            const KEY&      keyAt(size_t index) const;
            ssize_t         indexOfKey(const KEY& key) const;
            const VALUE&    operator[] (size_t index) const;

    /*!
     * modifying the array
     */

            VALUE&          editValueFor(const KEY& key);
            VALUE&          editValueAt(size_t index);

            /*! 
             * add/insert/replace items
             */
             
            ssize_t         add(const KEY& key, const VALUE& item);
            ssize_t         replaceValueFor(const KEY& key, const VALUE& item);
            ssize_t         replaceValueAt(size_t index, const VALUE& item);

    /*!
     * remove items
     */

            ssize_t         removeItem(const KEY& key);
            ssize_t         removeItemsAt(size_t index, size_t count = 1);
            
private:
            SortedVector< key_value_pair_t<KEY, VALUE> >    mVector;
};

// KeyedVector<KEY, VALUE> can be trivially moved using memcpy() because its
// underlying SortedVector can be trivially moved.
template<typename KEY, typename VALUE> struct trait_trivial_move<KeyedVector<KEY, VALUE> > {
    enum { value = trait_trivial_move<SortedVector< key_value_pair_t<KEY, VALUE> > >::value };
};


// ---------------------------------------------------------------------------

/**
 * Variation of KeyedVector that holds a default value to return when
 * valueFor() is called with a key that doesn't exist.
 */
template <typename KEY, typename VALUE>
class DefaultKeyedVector : public KeyedVector<KEY, VALUE>
{
public:
    inline                  DefaultKeyedVector(const VALUE& defValue = VALUE());
            const VALUE&    valueFor(const KEY& key) const;

private:
            VALUE                                           mDefault;
};

// ---------------------------------------------------------------------------

template<typename KEY, typename VALUE> inline
KeyedVector<KEY,VALUE>::KeyedVector()
{
}

template<typename KEY, typename VALUE> inline
bool KeyedVector<KEY,VALUE>::isIdenticalTo(const KeyedVector<KEY,VALUE>& rhs) const {
    return mVector.array() == rhs.mVector.array();
}

template<typename KEY, typename VALUE> inline
ssize_t KeyedVector<KEY,VALUE>::indexOfKey(const KEY& key) const {
    return mVector.indexOf( key_value_pair_t<KEY,VALUE>(key) );
}

template<typename KEY, typename VALUE> inline
const VALUE& KeyedVector<KEY,VALUE>::valueFor(const KEY& key) const {
    ssize_t i = this->indexOfKey(key);
    LOG_ALWAYS_FATAL_IF(i<0, "%s: key not found", __PRETTY_FUNCTION__);
    return mVector.itemAt(i).value;
}

template<typename KEY, typename VALUE> inline
const VALUE& KeyedVector<KEY,VALUE>::valueAt(size_t index) const {
    return mVector.itemAt(index).value;
}

template<typename KEY, typename VALUE> inline
const VALUE& KeyedVector<KEY,VALUE>::operator[] (size_t index) const {
    return valueAt(index);
}

template<typename KEY, typename VALUE> inline
const KEY& KeyedVector<KEY,VALUE>::keyAt(size_t index) const {
    return mVector.itemAt(index).key;
}

template<typename KEY, typename VALUE> inline
VALUE& KeyedVector<KEY,VALUE>::editValueFor(const KEY& key) {
    ssize_t i = this->indexOfKey(key);
    LOG_ALWAYS_FATAL_IF(i<0, "%s: key not found", __PRETTY_FUNCTION__);
    return mVector.editItemAt(i).value;
}

template<typename KEY, typename VALUE> inline
VALUE& KeyedVector<KEY,VALUE>::editValueAt(size_t index) {
    return mVector.editItemAt(index).value;
}

template<typename KEY, typename VALUE> inline
ssize_t KeyedVector<KEY,VALUE>::add(const KEY& key, const VALUE& value) {
    return mVector.add( key_value_pair_t<KEY,VALUE>(key, value) );
}

template<typename KEY, typename VALUE> inline
ssize_t KeyedVector<KEY,VALUE>::replaceValueFor(const KEY& key, const VALUE& value) {
    key_value_pair_t<KEY,VALUE> pair(key, value);
    mVector.remove(pair);
    return mVector.add(pair);
}

template<typename KEY, typename VALUE> inline
ssize_t KeyedVector<KEY,VALUE>::replaceValueAt(size_t index, const VALUE& item) {
    if (index<size()) {
        mVector.editItemAt(index).value = item;
        return index;
    }
    return BAD_INDEX;
}

template<typename KEY, typename VALUE> inline
ssize_t KeyedVector<KEY,VALUE>::removeItem(const KEY& key) {
    return mVector.remove(key_value_pair_t<KEY,VALUE>(key));
}

template<typename KEY, typename VALUE> inline
ssize_t KeyedVector<KEY, VALUE>::removeItemsAt(size_t index, size_t count) {
    return mVector.removeItemsAt(index, count);
}

// ---------------------------------------------------------------------------

template<typename KEY, typename VALUE> inline
DefaultKeyedVector<KEY,VALUE>::DefaultKeyedVector(const VALUE& defValue)
    : mDefault(defValue)
{
}

template<typename KEY, typename VALUE> inline
const VALUE& DefaultKeyedVector<KEY,VALUE>::valueFor(const KEY& key) const {
    ssize_t i = this->indexOfKey(key);
    return i >= 0 ? KeyedVector<KEY,VALUE>::valueAt(i) : mDefault;
}

}; // namespace android

// ---------------------------------------------------------------------------

#endif // ANDROID_KEYED_VECTOR_H
