/*
 * Copyright 2005 The Android Open Source Project
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Google Inc. nor the names of its contributors may
 *       be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Google Inc. ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL Google Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef SYSTEM_CORE_INCLUDE_MINCRYPT_SHA1_H_
#define SYSTEM_CORE_INCLUDE_MINCRYPT_SHA1_H_

#include "hash-internal.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef HASH_CTX SHA_CTX;

void SHA_init(SHA_CTX* ctx);
void SHA_update(SHA_CTX* ctx, const void* data, int len);
const uint8_t* SHA_final(SHA_CTX* ctx);

// Convenience method. Returns digest address.
// NOTE: *digest needs to hold SHA_DIGEST_SIZE bytes.
const uint8_t* SHA_hash(const void* data, int len, uint8_t* digest);

#define SHA_DIGEST_SIZE 20

#ifdef __cplusplus
}
#endif // __cplusplus

#endif  // SYSTEM_CORE_INCLUDE_MINCRYPT_SHA1_H_
