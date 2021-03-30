/*
 * Copyright (C) 2010 The Android Open Source Project
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

#ifndef UTILS_BITSET_H
#define UTILS_BITSET_H

#include <stdint.h>
#include <utils/TypeHelpers.h>

/*
 * Contains some bit manipulation helpers.
 */

namespace android {

// A simple set of 32 bits that can be individually marked or cleared.
struct BitSet32 {
    uint32_t value;

    inline BitSet32() : value(0UL) { }
    explicit inline BitSet32(uint32_t value) : value(value) { }

    // Gets the value associated with a particular bit index.
    static inline uint32_t valueForBit(uint32_t n) { return 0x80000000UL >> n; }

    // Clears the bit set.
    inline void clear() { clear(value); }

    static inline void clear(uint32_t& value) { value = 0UL; }

    // Returns the number of marked bits in the set.
    inline uint32_t count() const { return count(value); }

    static inline uint32_t count(uint32_t value) { return __builtin_popcountl(value); }

    // Returns true if the bit set does not contain any marked bits.
    inline bool isEmpty() const { return isEmpty(value); }

    static inline bool isEmpty(uint32_t value) { return ! value; }

    // Returns true if the bit set does not contain any unmarked bits.
    inline bool isFull() const { return isFull(value); }

    static inline bool isFull(uint32_t value) { return value == 0xffffffffUL; }

    // Returns true if the specified bit is marked.
    inline bool hasBit(uint32_t n) const { return hasBit(value, n); }

    static inline bool hasBit(uint32_t value, uint32_t n) { return value & valueForBit(n); }

    // Marks the specified bit.
    inline void markBit(uint32_t n) { markBit(value, n); }

    static inline void markBit (uint32_t& value, uint32_t n) { value |= valueForBit(n); }

    // Clears the specified bit.
    inline void clearBit(uint32_t n) { clearBit(value, n); }

    static inline void clearBit(uint32_t& value, uint32_t n) { value &= ~ valueForBit(n); }

    // Finds the first marked bit in the set.
    // Result is undefined if all bits are unmarked.
    inline uint32_t firstMarkedBit() const { return firstMarkedBit(value); }

    static uint32_t firstMarkedBit(uint32_t value) { return clz_checked(value); }

    // Finds the first unmarked bit in the set.
    // Result is undefined if all bits are marked.
    inline uint32_t firstUnmarkedBit() const { return firstUnmarkedBit(value); }

    static inline uint32_t firstUnmarkedBit(uint32_t value) { return clz_checked(~ value); }

    // Finds the last marked bit in the set.
    // Result is undefined if all bits are unmarked.
    inline uint32_t lastMarkedBit() const { return lastMarkedBit(value); }

    static inline uint32_t lastMarkedBit(uint32_t value) { return 31 - ctz_checked(value); }

    // Finds the first marked bit in the set and clears it.  Returns the bit index.
    // Result is undefined if all bits are unmarked.
    inline uint32_t clearFirstMarkedBit() { return clearFirstMarkedBit(value); }

    static inline uint32_t clearFirstMarkedBit(uint32_t& value) {
        uint32_t n = firstMarkedBit(value);
        clearBit(value, n);
        return n;
    }

    // Finds the first unmarked bit in the set and marks it.  Returns the bit index.
    // Result is undefined if all bits are marked.
    inline uint32_t markFirstUnmarkedBit() { return markFirstUnmarkedBit(value); }

    static inline uint32_t markFirstUnmarkedBit(uint32_t& value) {
        uint32_t n = firstUnmarkedBit(value);
        markBit(value, n);
        return n;
    }

    // Finds the last marked bit in the set and clears it.  Returns the bit index.
    // Result is undefined if all bits are unmarked.
    inline uint32_t clearLastMarkedBit() { return clearLastMarkedBit(value); }

    static inline uint32_t clearLastMarkedBit(uint32_t& value) {
        uint32_t n = lastMarkedBit(value);
        clearBit(value, n);
        return n;
    }

    // Gets the index of the specified bit in the set, which is the number of
    // marked bits that appear before the specified bit.
    inline uint32_t getIndexOfBit(uint32_t n) const {
        return getIndexOfBit(value, n);
    }

    static inline uint32_t getIndexOfBit(uint32_t value, uint32_t n) {
        return __builtin_popcountl(value & ~(0xffffffffUL >> n));
    }

    inline bool operator== (const BitSet32& other) const { return value == other.value; }
    inline bool operator!= (const BitSet32& other) const { return value != other.value; }
    inline BitSet32 operator& (const BitSet32& other) const {
        return BitSet32(value & other.value);
    }
    inline BitSet32& operator&= (const BitSet32& other) {
        value &= other.value;
        return *this;
    }
    inline BitSet32 operator| (const BitSet32& other) const {
        return BitSet32(value | other.value);
    }
    inline BitSet32& operator|= (const BitSet32& other) {
        value |= other.value;
        return *this;
    }

private:
    // We use these helpers as the signature of __builtin_c{l,t}z has "unsigned int" for the
    // input, which is only guaranteed to be 16b, not 32. The compiler should optimize this away.
    static inline uint32_t clz_checked(uint32_t value) {
        if (sizeof(unsigned int) == sizeof(uint32_t)) {
            return __builtin_clz(value);
        } else {
            return __builtin_clzl(value);
        }
    }

    static inline uint32_t ctz_checked(uint32_t value) {
        if (sizeof(unsigned int) == sizeof(uint32_t)) {
            return __builtin_ctz(value);
        } else {
            return __builtin_ctzl(value);
        }
    }
};

ANDROID_BASIC_TYPES_TRAITS(BitSet32)

// A simple set of 64 bits that can be individually marked or cleared.
struct BitSet64 {
    uint64_t value;

    inline BitSet64() : value(0ULL) { }
    explicit inline BitSet64(uint64_t value) : value(value) { }

    // Gets the value associated with a particular bit index.
    static inline uint64_t valueForBit(uint32_t n) { return 0x8000000000000000ULL >> n; }

    // Clears the bit set.
    inline void clear() { clear(value); }

    static inline void clear(uint64_t& value) { value = 0ULL; }

    // Returns the number of marked bits in the set.
    inline uint32_t count() const { return count(value); }

    static inline uint32_t count(uint64_t value) { return __builtin_popcountll(value); }

    // Returns true if the bit set does not contain any marked bits.
    inline bool isEmpty() const { return isEmpty(value); }

    static inline bool isEmpty(uint64_t value) { return ! value; }

    // Returns true if the bit set does not contain any unmarked bits.
    inline bool isFull() const { return isFull(value); }

    static inline bool isFull(uint64_t value) { return value == 0xffffffffffffffffULL; }

    // Returns true if the specified bit is marked.
    inline bool hasBit(uint32_t n) const { return hasBit(value, n); }

    static inline bool hasBit(uint64_t value, uint32_t n) { return value & valueForBit(n); }

    // Marks the specified bit.
    inline void markBit(uint32_t n) { markBit(value, n); }

    static inline void markBit(uint64_t& value, uint32_t n) { value |= valueForBit(n); }

    // Clears the specified bit.
    inline void clearBit(uint32_t n) { clearBit(value, n); }

    static inline void clearBit(uint64_t& value, uint32_t n) { value &= ~ valueForBit(n); }

    // Finds the first marked bit in the set.
    // Result is undefined if all bits are unmarked.
    inline uint32_t firstMarkedBit() const { return firstMarkedBit(value); }

    static inline uint32_t firstMarkedBit(uint64_t value) { return __builtin_clzll(value); }

    // Finds the first unmarked bit in the set.
    // Result is undefined if all bits are marked.
    inline uint32_t firstUnmarkedBit() const { return firstUnmarkedBit(value); }

    static inline uint32_t firstUnmarkedBit(uint64_t value) { return __builtin_clzll(~ value); }

    // Finds the last marked bit in the set.
    // Result is undefined if all bits are unmarked.
    inline uint32_t lastMarkedBit() const { return lastMarkedBit(value); }

    static inline uint32_t lastMarkedBit(uint64_t value) { return 63 - __builtin_ctzll(value); }

    // Finds the first marked bit in the set and clears it.  Returns the bit index.
    // Result is undefined if all bits are unmarked.
    inline uint32_t clearFirstMarkedBit() { return clearFirstMarkedBit(value); }

    static inline uint32_t clearFirstMarkedBit(uint64_t& value) {
        uint64_t n = firstMarkedBit(value);
        clearBit(value, n);
        return n;
    }

    // Finds the first unmarked bit in the set and marks it.  Returns the bit index.
    // Result is undefined if all bits are marked.
    inline uint32_t markFirstUnmarkedBit() { return markFirstUnmarkedBit(value); }

    static inline uint32_t markFirstUnmarkedBit(uint64_t& value) {
        uint64_t n = firstUnmarkedBit(value);
        markBit(value, n);
        return n;
    }

    // Finds the last marked bit in the set and clears it.  Returns the bit index.
    // Result is undefined if all bits are unmarked.
    inline uint32_t clearLastMarkedBit() { return clearLastMarkedBit(value); }

    static inline uint32_t clearLastMarkedBit(uint64_t& value) {
        uint64_t n = lastMarkedBit(value);
        clearBit(value, n);
        return n;
    }

    // Gets the index of the specified bit in the set, which is the number of
    // marked bits that appear before the specified bit.
    inline uint32_t getIndexOfBit(uint32_t n) const { return getIndexOfBit(value, n); }

    static inline uint32_t getIndexOfBit(uint64_t value, uint32_t n) {
        return __builtin_popcountll(value & ~(0xffffffffffffffffULL >> n));
    }

    inline bool operator== (const BitSet64& other) const { return value == other.value; }
    inline bool operator!= (const BitSet64& other) const { return value != other.value; }
    inline BitSet64 operator& (const BitSet64& other) const {
        return BitSet64(value & other.value);
    }
    inline BitSet64& operator&= (const BitSet64& other) {
        value &= other.value;
        return *this;
    }
    inline BitSet64 operator| (const BitSet64& other) const {
        return BitSet64(value | other.value);
    }
    inline BitSet64& operator|= (const BitSet64& other) {
        value |= other.value;
        return *this;
    }
};

ANDROID_BASIC_TYPES_TRAITS(BitSet64)

} // namespace android

#endif // UTILS_BITSET_H
