/*
 * Copyright (C) 2011 The Android Open Source Project
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

#ifndef ANDROID_BASIC_HASHTABLE_H
#define ANDROID_BASIC_HASHTABLE_H

#include <stdint.h>
#include <sys/types.h>
#include <utils/SharedBuffer.h>
#include <utils/TypeHelpers.h>

namespace android {

/* Implementation type.  Nothing to see here. */
class BasicHashtableImpl {
protected:
    struct Bucket {
        // The collision flag indicates that the bucket is part of a collision chain
        // such that at least two entries both hash to this bucket.  When true, we
        // may need to seek further along the chain to find the entry.
        static const uint32_t COLLISION = 0x80000000UL;

        // The present flag indicates that the bucket contains an initialized entry value.
        static const uint32_t PRESENT   = 0x40000000UL;

        // Mask for 30 bits worth of the hash code that are stored within the bucket to
        // speed up lookups and rehashing by eliminating the need to recalculate the
        // hash code of the entry's key.
        static const uint32_t HASH_MASK = 0x3fffffffUL;

        // Combined value that stores the collision and present flags as well as
        // a 30 bit hash code.
        uint32_t cookie;

        // Storage for the entry begins here.
        char entry[0];
    };

    BasicHashtableImpl(size_t entrySize, bool hasTrivialDestructor,
            size_t minimumInitialCapacity, float loadFactor);
    BasicHashtableImpl(const BasicHashtableImpl& other);
    virtual ~BasicHashtableImpl();

    void dispose();

    inline void edit() {
        if (mBuckets && !SharedBuffer::bufferFromData(mBuckets)->onlyOwner()) {
            clone();
        }
    }

    void setTo(const BasicHashtableImpl& other);
    void clear();

    ssize_t next(ssize_t index) const;
    ssize_t find(ssize_t index, hash_t hash, const void* __restrict__ key) const;
    size_t add(hash_t hash, const void* __restrict__ entry);
    void removeAt(size_t index);
    void rehash(size_t minimumCapacity, float loadFactor);

    const size_t mBucketSize; // number of bytes per bucket including the entry
    const bool mHasTrivialDestructor; // true if the entry type does not require destruction
    size_t mCapacity;         // number of buckets that can be filled before exceeding load factor
    float mLoadFactor;        // load factor
    size_t mSize;             // number of elements actually in the table
    size_t mFilledBuckets;    // number of buckets for which collision or present is true
    size_t mBucketCount;      // number of slots in the mBuckets array
    void* mBuckets;           // array of buckets, as a SharedBuffer

    inline const Bucket& bucketAt(const void* __restrict__ buckets, size_t index) const {
        return *reinterpret_cast<const Bucket*>(
                static_cast<const uint8_t*>(buckets) + index * mBucketSize);
    }

    inline Bucket& bucketAt(void* __restrict__ buckets, size_t index) const {
        return *reinterpret_cast<Bucket*>(static_cast<uint8_t*>(buckets) + index * mBucketSize);
    }

    virtual bool compareBucketKey(const Bucket& bucket, const void* __restrict__ key) const = 0;
    virtual void initializeBucketEntry(Bucket& bucket, const void* __restrict__ entry) const = 0;
    virtual void destroyBucketEntry(Bucket& bucket) const = 0;

private:
    void clone();

    // Allocates a bucket array as a SharedBuffer.
    void* allocateBuckets(size_t count) const;

    // Releases a bucket array's associated SharedBuffer.
    void releaseBuckets(void* __restrict__ buckets, size_t count) const;

    // Destroys the contents of buckets (invokes destroyBucketEntry for each
    // populated bucket if needed).
    void destroyBuckets(void* __restrict__ buckets, size_t count) const;

    // Copies the content of buckets (copies the cookie and invokes copyBucketEntry
    // for each populated bucket if needed).
    void copyBuckets(const void* __restrict__ fromBuckets,
            void* __restrict__ toBuckets, size_t count) const;

    // Determines the appropriate size of a bucket array to store a certain minimum
    // number of entries and returns its effective capacity.
    static void determineCapacity(size_t minimumCapacity, float loadFactor,
            size_t* __restrict__ outBucketCount, size_t* __restrict__ outCapacity);

    // Trim a hash code to 30 bits to match what we store in the bucket's cookie.
    inline static hash_t trimHash(hash_t hash) {
        return (hash & Bucket::HASH_MASK) ^ (hash >> 30);
    }

    // Returns the index of the first bucket that is in the collision chain
    // for the specified hash code, given the total number of buckets.
    // (Primary hash)
    inline static size_t chainStart(hash_t hash, size_t count) {
        return hash % count;
    }

    // Returns the increment to add to a bucket index to seek to the next bucket
    // in the collision chain for the specified hash code, given the total number of buckets.
    // (Secondary hash)
    inline static size_t chainIncrement(hash_t hash, size_t count) {
        return ((hash >> 7) | (hash << 25)) % (count - 1) + 1;
    }

    // Returns the index of the next bucket that is in the collision chain
    // that is defined by the specified increment, given the total number of buckets.
    inline static size_t chainSeek(size_t index, size_t increment, size_t count) {
        return (index + increment) % count;
    }
};

/*
 * A BasicHashtable stores entries that are indexed by hash code in place
 * within an array.  The basic operations are finding entries by key,
 * adding new entries and removing existing entries.
 *
 * This class provides a very limited set of operations with simple semantics.
 * It is intended to be used as a building block to construct more complex
 * and interesting data structures such as HashMap.  Think very hard before
 * adding anything extra to BasicHashtable, it probably belongs at a
 * higher level of abstraction.
 *
 * TKey: The key type.
 * TEntry: The entry type which is what is actually stored in the array.
 *
 * TKey must support the following contract:
 *     bool operator==(const TKey& other) const;  // return true if equal
 *     bool operator!=(const TKey& other) const;  // return true if unequal
 *
 * TEntry must support the following contract:
 *     const TKey& getKey() const;  // get the key from the entry
 *
 * This class supports storing entries with duplicate keys.  Of course, it can't
 * tell them apart during removal so only the first entry will be removed.
 * We do this because it means that operations like add() can't fail.
 */
template <typename TKey, typename TEntry>
class BasicHashtable : private BasicHashtableImpl {
public:
    /* Creates a hashtable with the specified minimum initial capacity.
     * The underlying array will be created when the first entry is added.
     *
     * minimumInitialCapacity: The minimum initial capacity for the hashtable.
     *     Default is 0.
     * loadFactor: The desired load factor for the hashtable, between 0 and 1.
     *     Default is 0.75.
     */
    BasicHashtable(size_t minimumInitialCapacity = 0, float loadFactor = 0.75f);

    /* Copies a hashtable.
     * The underlying storage is shared copy-on-write.
     */
    BasicHashtable(const BasicHashtable& other);

    /* Clears and destroys the hashtable.
     */
    virtual ~BasicHashtable();

    /* Making this hashtable a copy of the other hashtable.
     * The underlying storage is shared copy-on-write.
     *
     * other: The hashtable to copy.
     */
    inline BasicHashtable<TKey, TEntry>& operator =(const BasicHashtable<TKey, TEntry> & other) {
        setTo(other);
        return *this;
    }

    /* Returns the number of entries in the hashtable.
     */
    inline size_t size() const {
        return mSize;
    }

    /* Returns the capacity of the hashtable, which is the number of elements that can
     * added to the hashtable without requiring it to be grown.
     */
    inline size_t capacity() const {
        return mCapacity;
    }

    /* Returns the number of buckets that the hashtable has, which is the size of its
     * underlying array.
     */
    inline size_t bucketCount() const {
        return mBucketCount;
    }

    /* Returns the load factor of the hashtable. */
    inline float loadFactor() const {
        return mLoadFactor;
    };

    /* Returns a const reference to the entry at the specified index.
     *
     * index:   The index of the entry to retrieve.  Must be a valid index within
     *          the bounds of the hashtable.
     */
    inline const TEntry& entryAt(size_t index) const {
        return entryFor(bucketAt(mBuckets, index));
    }

    /* Returns a non-const reference to the entry at the specified index.
     *
     * index: The index of the entry to edit.  Must be a valid index within
     *        the bounds of the hashtable.
     */
    inline TEntry& editEntryAt(size_t index) {
        edit();
        return entryFor(bucketAt(mBuckets, index));
    }

    /* Clears the hashtable.
     * All entries in the hashtable are destroyed immediately.
     * If you need to do something special with the entries in the hashtable then iterate
     * over them and do what you need before clearing the hashtable.
     */
    inline void clear() {
        BasicHashtableImpl::clear();
    }

    /* Returns the index of the next entry in the hashtable given the index of a previous entry.
     * If the given index is -1, then returns the index of the first entry in the hashtable,
     * if there is one, or -1 otherwise.
     * If the given index is not -1, then returns the index of the next entry in the hashtable,
     * in strictly increasing order, or -1 if there are none left.
     *
     * index:   The index of the previous entry that was iterated, or -1 to begin
     *          iteration at the beginning of the hashtable.
     */
    inline ssize_t next(ssize_t index) const {
        return BasicHashtableImpl::next(index);
    }

    /* Finds the index of an entry with the specified key.
     * If the given index is -1, then returns the index of the first matching entry,
     * otherwise returns the index of the next matching entry.
     * If the hashtable contains multiple entries with keys that match the requested
     * key, then the sequence of entries returned is arbitrary.
     * Returns -1 if no entry was found.
     *
     * index:   The index of the previous entry with the specified key, or -1 to
     *          find the first matching entry.
     * hash:    The hashcode of the key.
     * key:     The key.
     */
    inline ssize_t find(ssize_t index, hash_t hash, const TKey& key) const {
        return BasicHashtableImpl::find(index, hash, &key);
    }

    /* Adds the entry to the hashtable.
     * Returns the index of the newly added entry.
     * If an entry with the same key already exists, then a duplicate entry is added.
     * If the entry will not fit, then the hashtable's capacity is increased and
     * its contents are rehashed.  See rehash().
     *
     * hash:    The hashcode of the key.
     * entry:   The entry to add.
     */
    inline size_t add(hash_t hash, const TEntry& entry) {
        return BasicHashtableImpl::add(hash, &entry);
    }

    /* Removes the entry with the specified index from the hashtable.
     * The entry is destroyed immediately.
     * The index must be valid.
     *
     * The hashtable is not compacted after an item is removed, so it is legal
     * to continue iterating over the hashtable using next() or find().
     *
     * index:   The index of the entry to remove.  Must be a valid index within the
     *          bounds of the hashtable, and it must refer to an existing entry.
     */
    inline void removeAt(size_t index) {
        BasicHashtableImpl::removeAt(index);
    }

    /* Rehashes the contents of the hashtable.
     * Grows the hashtable to at least the specified minimum capacity or the
     * current number of elements, whichever is larger.
     *
     * Rehashing causes all entries to be copied and the entry indices may change.
     * Although the hash codes are cached by the hashtable, rehashing can be an
     * expensive operation and should be avoided unless the hashtable's size
     * needs to be changed.
     *
     * Rehashing is the only way to change the capacity or load factor of the
     * hashtable once it has been created.  It can be used to compact the
     * hashtable by choosing a minimum capacity that is smaller than the current
     * capacity (such as 0).
     *
     * minimumCapacity: The desired minimum capacity after rehashing.
     * loadFactor: The desired load factor after rehashing.
     */
    inline void rehash(size_t minimumCapacity, float loadFactor) {
        BasicHashtableImpl::rehash(minimumCapacity, loadFactor);
    }

    /* Determines whether there is room to add another entry without rehashing.
     * When this returns true, a subsequent add() operation is guaranteed to
     * complete without performing a rehash.
     */
    inline bool hasMoreRoom() const {
        return mCapacity > mFilledBuckets;
    }

protected:
    static inline const TEntry& entryFor(const Bucket& bucket) {
        return reinterpret_cast<const TEntry&>(bucket.entry);
    }

    static inline TEntry& entryFor(Bucket& bucket) {
        return reinterpret_cast<TEntry&>(bucket.entry);
    }

    virtual bool compareBucketKey(const Bucket& bucket, const void* __restrict__ key) const;
    virtual void initializeBucketEntry(Bucket& bucket, const void* __restrict__ entry) const;
    virtual void destroyBucketEntry(Bucket& bucket) const;

private:
    // For dumping the raw contents of a hashtable during testing.
    friend class BasicHashtableTest;
    inline uint32_t cookieAt(size_t index) const {
        return bucketAt(mBuckets, index).cookie;
    }
};

template <typename TKey, typename TEntry>
BasicHashtable<TKey, TEntry>::BasicHashtable(size_t minimumInitialCapacity, float loadFactor) :
        BasicHashtableImpl(sizeof(TEntry), traits<TEntry>::has_trivial_dtor,
                minimumInitialCapacity, loadFactor) {
}

template <typename TKey, typename TEntry>
BasicHashtable<TKey, TEntry>::BasicHashtable(const BasicHashtable<TKey, TEntry>& other) :
        BasicHashtableImpl(other) {
}

template <typename TKey, typename TEntry>
BasicHashtable<TKey, TEntry>::~BasicHashtable() {
    dispose();
}

template <typename TKey, typename TEntry>
bool BasicHashtable<TKey, TEntry>::compareBucketKey(const Bucket& bucket,
        const void* __restrict__ key) const {
    return entryFor(bucket).getKey() == *static_cast<const TKey*>(key);
}

template <typename TKey, typename TEntry>
void BasicHashtable<TKey, TEntry>::initializeBucketEntry(Bucket& bucket,
        const void* __restrict__ entry) const {
    if (!traits<TEntry>::has_trivial_copy) {
        new (&entryFor(bucket)) TEntry(*(static_cast<const TEntry*>(entry)));
    } else {
        memcpy(&entryFor(bucket), entry, sizeof(TEntry));
    }
}

template <typename TKey, typename TEntry>
void BasicHashtable<TKey, TEntry>::destroyBucketEntry(Bucket& bucket) const {
    if (!traits<TEntry>::has_trivial_dtor) {
        entryFor(bucket).~TEntry();
    }
}

}; // namespace android

#endif // ANDROID_BASIC_HASHTABLE_H
