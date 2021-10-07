/*
 * Copyright (C) 2012 The Android Open Source Project
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

#ifndef ANDROID_UTILS_LRU_CACHE_H
#define ANDROID_UTILS_LRU_CACHE_H

#include <UniquePtr.h>
#include <utils/BasicHashtable.h>

namespace android {

/**
 * GenerationCache callback used when an item is removed
 */
template<typename EntryKey, typename EntryValue>
class OnEntryRemoved {
public:
    virtual ~OnEntryRemoved() { };
    virtual void operator()(EntryKey& key, EntryValue& value) = 0;
}; // class OnEntryRemoved

template <typename TKey, typename TValue>
class LruCache {
public:
    explicit LruCache(uint32_t maxCapacity);

    enum Capacity {
        kUnlimitedCapacity,
    };

    void setOnEntryRemovedListener(OnEntryRemoved<TKey, TValue>* listener);
    size_t size() const;
    const TValue& get(const TKey& key);
    bool put(const TKey& key, const TValue& value);
    bool remove(const TKey& key);
    bool removeOldest();
    void clear();
    const TValue& peekOldestValue();

    class Iterator {
    public:
        Iterator(const LruCache<TKey, TValue>& cache): mCache(cache), mIndex(-1) {
        }

        bool next() {
            mIndex = mCache.mTable->next(mIndex);
            return (ssize_t)mIndex != -1;
        }

        size_t index() const {
            return mIndex;
        }

        const TValue& value() const {
            return mCache.mTable->entryAt(mIndex).value;
        }

        const TKey& key() const {
            return mCache.mTable->entryAt(mIndex).key;
        }
    private:
        const LruCache<TKey, TValue>& mCache;
        size_t mIndex;
    };

private:
    LruCache(const LruCache& that);  // disallow copy constructor

    struct Entry {
        TKey key;
        TValue value;
        Entry* parent;
        Entry* child;

        Entry(TKey key_, TValue value_) : key(key_), value(value_), parent(NULL), child(NULL) {
        }
        const TKey& getKey() const { return key; }
    };

    void attachToCache(Entry& entry);
    void detachFromCache(Entry& entry);
    void rehash(size_t newCapacity);

    UniquePtr<BasicHashtable<TKey, Entry> > mTable;
    OnEntryRemoved<TKey, TValue>* mListener;
    Entry* mOldest;
    Entry* mYoungest;
    uint32_t mMaxCapacity;
    TValue mNullValue;
};

// Implementation is here, because it's fully templated
template <typename TKey, typename TValue>
LruCache<TKey, TValue>::LruCache(uint32_t maxCapacity)
    : mTable(new BasicHashtable<TKey, Entry>)
    , mListener(NULL)
    , mOldest(NULL)
    , mYoungest(NULL)
    , mMaxCapacity(maxCapacity)
    , mNullValue(NULL) {
};

template<typename K, typename V>
void LruCache<K, V>::setOnEntryRemovedListener(OnEntryRemoved<K, V>* listener) {
    mListener = listener;
}

template <typename TKey, typename TValue>
size_t LruCache<TKey, TValue>::size() const {
    return mTable->size();
}

template <typename TKey, typename TValue>
const TValue& LruCache<TKey, TValue>::get(const TKey& key) {
    hash_t hash = hash_type(key);
    ssize_t index = mTable->find(-1, hash, key);
    if (index == -1) {
        return mNullValue;
    }
    Entry& entry = mTable->editEntryAt(index);
    detachFromCache(entry);
    attachToCache(entry);
    return entry.value;
}

template <typename TKey, typename TValue>
bool LruCache<TKey, TValue>::put(const TKey& key, const TValue& value) {
    if (mMaxCapacity != kUnlimitedCapacity && size() >= mMaxCapacity) {
        removeOldest();
    }

    hash_t hash = hash_type(key);
    ssize_t index = mTable->find(-1, hash, key);
    if (index >= 0) {
        return false;
    }
    if (!mTable->hasMoreRoom()) {
        rehash(mTable->capacity() * 2);
    }

    // Would it be better to initialize a blank entry and assign key, value?
    Entry initEntry(key, value);
    index = mTable->add(hash, initEntry);
    Entry& entry = mTable->editEntryAt(index);
    attachToCache(entry);
    return true;
}

template <typename TKey, typename TValue>
bool LruCache<TKey, TValue>::remove(const TKey& key) {
    hash_t hash = hash_type(key);
    ssize_t index = mTable->find(-1, hash, key);
    if (index < 0) {
        return false;
    }
    Entry& entry = mTable->editEntryAt(index);
    if (mListener) {
        (*mListener)(entry.key, entry.value);
    }
    detachFromCache(entry);
    mTable->removeAt(index);
    return true;
}

template <typename TKey, typename TValue>
bool LruCache<TKey, TValue>::removeOldest() {
    if (mOldest != NULL) {
        return remove(mOldest->key);
        // TODO: should probably abort if false
    }
    return false;
}

template <typename TKey, typename TValue>
const TValue& LruCache<TKey, TValue>::peekOldestValue() {
    if (mOldest) {
        return mOldest->value;
    }
    return mNullValue;
}

template <typename TKey, typename TValue>
void LruCache<TKey, TValue>::clear() {
    if (mListener) {
        for (Entry* p = mOldest; p != NULL; p = p->child) {
            (*mListener)(p->key, p->value);
        }
    }
    mYoungest = NULL;
    mOldest = NULL;
    mTable->clear();
}

template <typename TKey, typename TValue>
void LruCache<TKey, TValue>::attachToCache(Entry& entry) {
    if (mYoungest == NULL) {
        mYoungest = mOldest = &entry;
    } else {
        entry.parent = mYoungest;
        mYoungest->child = &entry;
        mYoungest = &entry;
    }
}

template <typename TKey, typename TValue>
void LruCache<TKey, TValue>::detachFromCache(Entry& entry) {
    if (entry.parent != NULL) {
        entry.parent->child = entry.child;
    } else {
        mOldest = entry.child;
    }
    if (entry.child != NULL) {
        entry.child->parent = entry.parent;
    } else {
        mYoungest = entry.parent;
    }

    entry.parent = NULL;
    entry.child = NULL;
}

template <typename TKey, typename TValue>
void LruCache<TKey, TValue>::rehash(size_t newCapacity) {
    UniquePtr<BasicHashtable<TKey, Entry> > oldTable(mTable.release());
    Entry* oldest = mOldest;

    mOldest = NULL;
    mYoungest = NULL;
    mTable.reset(new BasicHashtable<TKey, Entry>(newCapacity));
    for (Entry* p = oldest; p != NULL; p = p->child) {
        put(p->key, p->value);
    }
}

}

#endif // ANDROID_UTILS_LRU_CACHE_H
