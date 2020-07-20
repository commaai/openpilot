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

#ifndef ANDROID_UNICODE_H
#define ANDROID_UNICODE_H

#include <sys/types.h>
#include <stdint.h>

extern "C" {

// Standard string functions on char16_t strings.
int strcmp16(const char16_t *, const char16_t *);
int strncmp16(const char16_t *s1, const char16_t *s2, size_t n);
size_t strlen16(const char16_t *);
size_t strnlen16(const char16_t *, size_t);
char16_t *strcpy16(char16_t *, const char16_t *);
char16_t *strncpy16(char16_t *, const char16_t *, size_t);

// Version of comparison that supports embedded nulls.
// This is different than strncmp() because we don't stop
// at a nul character and consider the strings to be different
// if the lengths are different (thus we need to supply the
// lengths of both strings).  This can also be used when
// your string is not nul-terminated as it will have the
// equivalent result as strcmp16 (unlike strncmp16).
int strzcmp16(const char16_t *s1, size_t n1, const char16_t *s2, size_t n2);

// Version of strzcmp16 for comparing strings in different endianness.
int strzcmp16_h_n(const char16_t *s1H, size_t n1, const char16_t *s2N, size_t n2);

// Standard string functions on char32_t strings.
size_t strlen32(const char32_t *);
size_t strnlen32(const char32_t *, size_t);

/**
 * Measure the length of a UTF-32 string in UTF-8. If the string is invalid
 * such as containing a surrogate character, -1 will be returned.
 */
ssize_t utf32_to_utf8_length(const char32_t *src, size_t src_len);

/**
 * Stores a UTF-8 string converted from "src" in "dst", if "dst_length" is not
 * large enough to store the string, the part of the "src" string is stored
 * into "dst" as much as possible. See the examples for more detail.
 * Returns the size actually used for storing the string.
 * dst" is not null-terminated when dst_len is fully used (like strncpy).
 *
 * Example 1
 * "src" == \u3042\u3044 (\xE3\x81\x82\xE3\x81\x84)
 * "src_len" == 2
 * "dst_len" >= 7
 * ->
 * Returned value == 6
 * "dst" becomes \xE3\x81\x82\xE3\x81\x84\0
 * (note that "dst" is null-terminated)
 *
 * Example 2
 * "src" == \u3042\u3044 (\xE3\x81\x82\xE3\x81\x84)
 * "src_len" == 2
 * "dst_len" == 5
 * ->
 * Returned value == 3
 * "dst" becomes \xE3\x81\x82\0
 * (note that "dst" is null-terminated, but \u3044 is not stored in "dst"
 * since "dst" does not have enough size to store the character)
 *
 * Example 3
 * "src" == \u3042\u3044 (\xE3\x81\x82\xE3\x81\x84)
 * "src_len" == 2
 * "dst_len" == 6
 * ->
 * Returned value == 6
 * "dst" becomes \xE3\x81\x82\xE3\x81\x84
 * (note that "dst" is NOT null-terminated, like strncpy)
 */
void utf32_to_utf8(const char32_t* src, size_t src_len, char* dst, size_t dst_len);

/**
 * Returns the unicode value at "index".
 * Returns -1 when the index is invalid (equals to or more than "src_len").
 * If returned value is positive, it is able to be converted to char32_t, which
 * is unsigned. Then, if "next_index" is not NULL, the next index to be used is
 * stored in "next_index". "next_index" can be NULL.
 */
int32_t utf32_from_utf8_at(const char *src, size_t src_len, size_t index, size_t *next_index);


/**
 * Returns the UTF-8 length of UTF-16 string "src".
 */
ssize_t utf16_to_utf8_length(const char16_t *src, size_t src_len);

/**
 * Converts a UTF-16 string to UTF-8. The destination buffer must be large
 * enough to fit the UTF-16 as measured by utf16_to_utf8_length with an added
 * NULL terminator.
 */
void utf16_to_utf8(const char16_t* src, size_t src_len, char* dst, size_t dst_len);

/**
 * Returns the length of "src" when "src" is valid UTF-8 string.
 * Returns 0 if src is NULL or 0-length string. Returns -1 when the source
 * is an invalid string.
 *
 * This function should be used to determine whether "src" is valid UTF-8
 * characters with valid unicode codepoints. "src" must be null-terminated.
 *
 * If you are going to use other utf8_to_... functions defined in this header
 * with string which may not be valid UTF-8 with valid codepoint (form 0 to
 * 0x10FFFF), you should use this function before calling others, since the
 * other functions do not check whether the string is valid UTF-8 or not.
 *
 * If you do not care whether "src" is valid UTF-8 or not, you should use
 * strlen() as usual, which should be much faster.
 */
ssize_t utf8_length(const char *src);

/**
 * Measure the length of a UTF-32 string.
 */
size_t utf8_to_utf32_length(const char *src, size_t src_len);

/**
 * Stores a UTF-32 string converted from "src" in "dst". "dst" must be large
 * enough to store the entire converted string as measured by
 * utf8_to_utf32_length plus space for a NULL terminator.
 */
void utf8_to_utf32(const char* src, size_t src_len, char32_t* dst);

/**
 * Returns the UTF-16 length of UTF-8 string "src".
 */
ssize_t utf8_to_utf16_length(const uint8_t* src, size_t srcLen);

/**
 * Convert UTF-8 to UTF-16 including surrogate pairs.
 * Returns a pointer to the end of the string (where a null terminator might go
 * if you wanted to add one).
 */
char16_t* utf8_to_utf16_no_null_terminator(const uint8_t* src, size_t srcLen, char16_t* dst);

/**
 * Convert UTF-8 to UTF-16 including surrogate pairs. The destination buffer
 * must be large enough to hold the result as measured by utf8_to_utf16_length
 * plus an added NULL terminator.
 */
void utf8_to_utf16(const uint8_t* src, size_t srcLen, char16_t* dst);

/**
 * Like utf8_to_utf16_no_null_terminator, but you can supply a maximum length of the
 * decoded string.  The decoded string will fill up to that length; if it is longer
 * the returned pointer will be to the character after dstLen.
 */
char16_t* utf8_to_utf16_n(const uint8_t* src, size_t srcLen, char16_t* dst, size_t dstLen);

}

#endif
