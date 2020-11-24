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

#ifndef __CUTILS_BITOPS_H
#define __CUTILS_BITOPS_H

#include <stdbool.h>
#include <string.h>
#include <strings.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/*
 * Bitmask Operations
 *
 * Note this doesn't provide any locking/exclusion, and isn't atomic.
 * Additionally no bounds checking is done on the bitmask array.
 *
 * Example:
 *
 * int num_resources;
 * unsigned int resource_bits[BITS_TO_WORDS(num_resources)];
 * bitmask_init(resource_bits, num_resources);
 * ...
 * int bit = bitmask_ffz(resource_bits, num_resources);
 * bitmask_set(resource_bits, bit);
 * ...
 * if (bitmask_test(resource_bits, bit)) { ... }
 * ...
 * bitmask_clear(resource_bits, bit);
 *
 */

#define BITS_PER_WORD    (sizeof(unsigned int) * 8)
#define BITS_TO_WORDS(x) (((x) + BITS_PER_WORD - 1) / BITS_PER_WORD)
#define BIT_IN_WORD(x)   ((x) % BITS_PER_WORD)
#define BIT_WORD(x)      ((x) / BITS_PER_WORD)
#define BIT_MASK(x)      (1 << BIT_IN_WORD(x))

static inline void bitmask_init(unsigned int *bitmask, int num_bits)
{
    memset(bitmask, 0, BITS_TO_WORDS(num_bits)*sizeof(unsigned int));
}

static inline int bitmask_ffz(unsigned int *bitmask, int num_bits)
{
    int bit, result;
    size_t i;

    for (i = 0; i < BITS_TO_WORDS(num_bits); i++) {
        bit = ffs(~bitmask[i]);
        if (bit) {
            // ffs is 1-indexed, return 0-indexed result
            bit--;
            result = BITS_PER_WORD * i + bit;
            if (result >= num_bits)
                return -1;
            return result;
        }
    }
    return -1;
}

static inline int bitmask_weight(unsigned int *bitmask, int num_bits)
{
    size_t i;
    int weight = 0;

    for (i = 0; i < BITS_TO_WORDS(num_bits); i++)
        weight += __builtin_popcount(bitmask[i]);
    return weight;
}

static inline void bitmask_set(unsigned int *bitmask, int bit)
{
    bitmask[BIT_WORD(bit)] |= BIT_MASK(bit);
}

static inline void bitmask_clear(unsigned int *bitmask, int bit)
{
    bitmask[BIT_WORD(bit)] &= ~BIT_MASK(bit);
}

static inline bool bitmask_test(unsigned int *bitmask, int bit)
{
    return bitmask[BIT_WORD(bit)] & BIT_MASK(bit);
}

static inline int popcount(unsigned int x)
{
    return __builtin_popcount(x);
}

static inline int popcountl(unsigned long x)
{
    return __builtin_popcountl(x);
}

static inline int popcountll(unsigned long long x)
{
    return __builtin_popcountll(x);
}

__END_DECLS

#endif /* __CUTILS_BITOPS_H */
