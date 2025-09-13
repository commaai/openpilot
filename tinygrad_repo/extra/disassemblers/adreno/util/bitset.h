/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2006  Brian Paul   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * \file bitset.h
 * \brief Bitset of arbitrary size definitions.
 * \author Michal Krol
 */

#ifndef BITSET_H
#define BITSET_H

//#include "util/bitscan.h"
//#include "util/macros.h"

/****************************************************************************
 * generic bitset implementation
 */

#define BITSET_WORD unsigned int
#define BITSET_WORDBITS (sizeof (BITSET_WORD) * 8)

/* bitset declarations
 */
#define BITSET_WORDS(bits) (((bits) + BITSET_WORDBITS - 1) / BITSET_WORDBITS)
#define BITSET_DECLARE(name, bits) BITSET_WORD name[BITSET_WORDS(bits)]

/* bitset operations
 */
#define BITSET_COPY(x, y) memcpy( (x), (y), sizeof (x) )
#define BITSET_EQUAL(x, y) (memcmp( (x), (y), sizeof (x) ) == 0)
#define BITSET_ZERO(x) memset( (x), 0, sizeof (x) )
#define BITSET_ONES(x) memset( (x), 0xff, sizeof (x) )

#define BITSET_BITWORD(b) ((b) / BITSET_WORDBITS)
#define BITSET_BIT(b) (1u << ((b) % BITSET_WORDBITS))

/* single bit operations
 */
#define BITSET_TEST(x, b) (((x)[BITSET_BITWORD(b)] & BITSET_BIT(b)) != 0)
#define BITSET_SET(x, b) ((x)[BITSET_BITWORD(b)] |= BITSET_BIT(b))
#define BITSET_CLEAR(x, b) ((x)[BITSET_BITWORD(b)] &= ~BITSET_BIT(b))

#define BITSET_MASK(b) (((b) % BITSET_WORDBITS == 0) ? ~0 : BITSET_BIT(b) - 1)
#define BITSET_RANGE(b, e) ((BITSET_MASK((e) + 1)) & ~(BITSET_BIT(b) - 1))

/* bit range operations
 */
#define BITSET_TEST_RANGE(x, b, e) \
   (BITSET_BITWORD(b) == BITSET_BITWORD(e) ? \
   (((x)[BITSET_BITWORD(b)] & BITSET_RANGE(b, e)) != 0) : \
   (assert (!"BITSET_TEST_RANGE: bit range crosses word boundary"), 0))
#define BITSET_SET_RANGE(x, b, e) \
   (BITSET_BITWORD(b) == BITSET_BITWORD(e) ? \
   ((x)[BITSET_BITWORD(b)] |= BITSET_RANGE(b, e)) : \
   (assert (!"BITSET_SET_RANGE: bit range crosses word boundary"), 0))
#define BITSET_CLEAR_RANGE(x, b, e) \
   (BITSET_BITWORD(b) == BITSET_BITWORD(e) ? \
   ((x)[BITSET_BITWORD(b)] &= ~BITSET_RANGE(b, e)) : \
   (assert (!"BITSET_CLEAR_RANGE: bit range crosses word boundary"), 0))

/* Get first bit set in a bitset.
 */
static inline int
__bitset_ffs(const BITSET_WORD *x, int n)
{
   int i;

   for (i = 0; i < n; i++) {
      if (x[i])
	 return ffs(x[i]) + BITSET_WORDBITS * i;
   }

   return 0;
}

#define BITSET_FFS(x) __bitset_ffs(x, ARRAY_SIZE(x))

static inline unsigned
__bitset_next_set(unsigned i, BITSET_WORD *tmp,
                  const BITSET_WORD *set, unsigned size)
{
   unsigned bit, word;

   /* NOTE: The initial conditions for this function are very specific.  At
    * the start of the loop, the tmp variable must be set to *set and the
    * initial i value set to 0.  This way, if there is a bit set in the first
    * word, we ignore the i-value and just grab that bit (so 0 is ok, even
    * though 0 may be returned).  If the first word is 0, then the value of
    * `word` will be 0 and we will go on to look at the second word.
    */
   word = BITSET_BITWORD(i);
   while (*tmp == 0) {
      word++;

      if (word >= BITSET_WORDS(size))
         return size;

      *tmp = set[word];
   }

   /* Find the next set bit in the non-zero word */
   bit = ffs(*tmp) - 1;

   /* Unset the bit */
   *tmp &= ~(1ull << bit);

   return word * BITSET_WORDBITS + bit;
}

/**
 * Iterates over each set bit in a set
 *
 * @param __i    iteration variable, bit number
 * @param __set  the bitset to iterate (will not be modified)
 * @param __size number of bits in the set to consider
 */
#define BITSET_FOREACH_SET(__i, __set, __size) \
   for (BITSET_WORD __tmp = *(__set), *__foo = &__tmp; __foo != NULL; __foo = NULL) \
      for (__i = 0; \
           (__i = __bitset_next_set(__i, &__tmp, __set, __size)) < __size;)

#ifdef __cplusplus

/**
 * Simple C++ wrapper of a bitset type of static size, with value semantics
 * and basic bitwise arithmetic operators.  The operators defined below are
 * expected to have the same semantics as the same operator applied to other
 * fundamental integer types.  T is the name of the struct to instantiate
 * it as, and N is the number of bits in the bitset.
 */
#define DECLARE_BITSET_T(T, N) struct T {                       \
      EXPLICIT_CONVERSION                                       \
      operator bool() const                                     \
      {                                                         \
         for (unsigned i = 0; i < BITSET_WORDS(N); i++)         \
            if (words[i])                                       \
               return true;                                     \
         return false;                                          \
      }                                                         \
                                                                \
      T &                                                       \
      operator=(int x)                                          \
      {                                                         \
         const T c = {{ (BITSET_WORD)x }};                      \
         return *this = c;                                      \
      }                                                         \
                                                                \
      friend bool                                               \
      operator==(const T &b, const T &c)                        \
      {                                                         \
         return BITSET_EQUAL(b.words, c.words);                 \
      }                                                         \
                                                                \
      friend bool                                               \
      operator!=(const T &b, const T &c)                        \
      {                                                         \
         return !(b == c);                                      \
      }                                                         \
                                                                \
      friend bool                                               \
      operator==(const T &b, int x)                             \
      {                                                         \
         const T c = {{ (BITSET_WORD)x }};                      \
         return b == c;                                         \
      }                                                         \
                                                                \
      friend bool                                               \
      operator!=(const T &b, int x)                             \
      {                                                         \
         return !(b == x);                                      \
      }                                                         \
                                                                \
      friend T                                                  \
      operator~(const T &b)                                     \
      {                                                         \
         T c;                                                   \
         for (unsigned i = 0; i < BITSET_WORDS(N); i++)         \
            c.words[i] = ~b.words[i];                           \
         return c;                                              \
      }                                                         \
                                                                \
      T &                                                       \
      operator|=(const T &b)                                    \
      {                                                         \
         for (unsigned i = 0; i < BITSET_WORDS(N); i++)         \
            words[i] |= b.words[i];                             \
         return *this;                                          \
      }                                                         \
                                                                \
      friend T                                                  \
      operator|(const T &b, const T &c)                         \
      {                                                         \
         T d = b;                                               \
         d |= c;                                                \
         return d;                                              \
      }                                                         \
                                                                \
      T &                                                       \
      operator&=(const T &b)                                    \
      {                                                         \
         for (unsigned i = 0; i < BITSET_WORDS(N); i++)         \
            words[i] &= b.words[i];                             \
         return *this;                                          \
      }                                                         \
                                                                \
      friend T                                                  \
      operator&(const T &b, const T &c)                         \
      {                                                         \
         T d = b;                                               \
         d &= c;                                                \
         return d;                                              \
      }                                                         \
                                                                \
      bool                                                      \
      test(unsigned i) const                                    \
      {                                                         \
         return BITSET_TEST(words, i);                          \
      }                                                         \
                                                                \
      T &                                                       \
      set(unsigned i)                                           \
      {                                                         \
         BITSET_SET(words, i);                                  \
         return *this;                                          \
      }                                                         \
                                                                \
      T &                                                       \
      clear(unsigned i)                                         \
      {                                                         \
         BITSET_CLEAR(words, i);                                \
         return *this;                                          \
      }                                                         \
                                                                \
      BITSET_WORD words[BITSET_WORDS(N)];                       \
   }

#endif

#endif
