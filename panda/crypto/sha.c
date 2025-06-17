/* sha.c
**
** Copyright 2013, The Android Open Source Project
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
**     * Redistributions of source code must retain the above copyright
**       notice, this list of conditions and the following disclaimer.
**     * Redistributions in binary form must reproduce the above copyright
**       notice, this list of conditions and the following disclaimer in the
**       documentation and/or other materials provided with the distribution.
**     * Neither the name of Google Inc. nor the names of its contributors may
**       be used to endorse or promote products derived from this software
**       without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY Google Inc. ``AS IS'' AND ANY EXPRESS OR
** IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
** MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
** EVENT SHALL Google Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
** PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
** OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
** WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
** OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
** ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Optimized for minimal code size.

void *memcpy(void *str1, const void *str2, unsigned int n);

#include "sha.h"

#define rol(bits, value) (((value) << (bits)) | ((value) >> (32 - (bits))))

static void SHA1_Transform(SHA_CTX* ctx) {
    uint32_t W[80];
    uint32_t A, B, C, D, E;
    uint8_t* p = ctx->buf;
    int t;

    for(t = 0; t < 16; ++t) {
        uint32_t tmp =  *p++ << 24;
        tmp |= *p++ << 16;
        tmp |= *p++ << 8;
        tmp |= *p++;
        W[t] = tmp;
    }

    for(; t < 80; t++) {
        W[t] = rol(1,W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16]);
    }

    A = ctx->state[0];
    B = ctx->state[1];
    C = ctx->state[2];
    D = ctx->state[3];
    E = ctx->state[4];

    for(t = 0; t < 80; t++) {
        uint32_t tmp = rol(5,A) + E + W[t];

        if (t < 20)
            tmp += (D^(B&(C^D))) + 0x5A827999;
        else if ( t < 40)
            tmp += (B^C^D) + 0x6ED9EBA1;
        else if ( t < 60)
            tmp += ((B&C)|(D&(B|C))) + 0x8F1BBCDC;
        else
            tmp += (B^C^D) + 0xCA62C1D6;

        E = D;
        D = C;
        C = rol(30,B);
        B = A;
        A = tmp;
    }

    ctx->state[0] += A;
    ctx->state[1] += B;
    ctx->state[2] += C;
    ctx->state[3] += D;
    ctx->state[4] += E;
}

static const HASH_VTAB SHA_VTAB = {
    SHA_init,
    SHA_update,
    SHA_final,
    SHA_hash,
    SHA_DIGEST_SIZE
};

void SHA_init(SHA_CTX* ctx) {
    ctx->f = &SHA_VTAB;
    ctx->state[0] = 0x67452301;
    ctx->state[1] = 0xEFCDAB89;
    ctx->state[2] = 0x98BADCFE;
    ctx->state[3] = 0x10325476;
    ctx->state[4] = 0xC3D2E1F0;
    ctx->count = 0;
}


void SHA_update(SHA_CTX* ctx, const void* data, int len) {
    int i = (int) (ctx->count & 63);
    const uint8_t* p = (const uint8_t*)data;

    ctx->count += len;

    while (len--) {
        ctx->buf[i++] = *p++;
        if (i == 64) {
            SHA1_Transform(ctx);
            i = 0;
        }
    }
}


const uint8_t* SHA_final(SHA_CTX* ctx) {
    uint8_t *p = ctx->buf;
    uint64_t cnt = ctx->count * 8;
    int i;

    SHA_update(ctx, (uint8_t*)"\x80", 1);
    while ((ctx->count & 63) != 56) {
        SHA_update(ctx, (uint8_t*)"\0", 1);
    }

    /* Hack - right shift operator with non const argument requires
     *        libgcc.a which is missing in EON
     *        thus expanding for loop from

              for (i = 0; i < 8; ++i) {
                  uint8_t tmp = (uint8_t) (cnt >> ((7 - i) * 8));
                  SHA_update(ctx, &tmp, 1);
              }

              to
     */

    uint8_t tmp = 0;
    tmp = (uint8_t) (cnt >> ((7 - 0) * 8));
    SHA_update(ctx, &tmp, 1);
    tmp = (uint8_t) (cnt >> ((7 - 1) * 8));
    SHA_update(ctx, &tmp, 1);
    tmp = (uint8_t) (cnt >> ((7 - 2) * 8));
    SHA_update(ctx, &tmp, 1);
    tmp = (uint8_t) (cnt >> ((7 - 3) * 8));
    SHA_update(ctx, &tmp, 1);
    tmp = (uint8_t) (cnt >> ((7 - 4) * 8));
    SHA_update(ctx, &tmp, 1);
    tmp = (uint8_t) (cnt >> ((7 - 5) * 8));
    SHA_update(ctx, &tmp, 1);
    tmp = (uint8_t) (cnt >> ((7 - 6) * 8));
    SHA_update(ctx, &tmp, 1);
    tmp = (uint8_t) (cnt >> ((7 - 7) * 8));
    SHA_update(ctx, &tmp, 1);

    for (i = 0; i < 5; i++) {
        uint32_t tmp = ctx->state[i];
        *p++ = tmp >> 24;
        *p++ = tmp >> 16;
        *p++ = tmp >> 8;
        *p++ = tmp >> 0;
    }

    return ctx->buf;
}

/* Convenience function */
const uint8_t* SHA_hash(const void* data, int len, uint8_t* digest) {
    SHA_CTX ctx;
    SHA_init(&ctx);
    SHA_update(&ctx, data, len);
    memcpy(digest, SHA_final(&ctx), SHA_DIGEST_SIZE);
    return digest;
}
