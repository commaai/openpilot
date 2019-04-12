/*
 * DTLS implementation written by Nagendra Modadugu
 * (nagendra@cs.stanford.edu) for the OpenSSL project 2005.
 */
/* ====================================================================
 * Copyright (c) 1999-2005 The OpenSSL Project.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. All advertising materials mentioning features or use of this
 *    software must display the following acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit. (http://www.OpenSSL.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    openssl-core@OpenSSL.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.OpenSSL.org/)"
 *
 * THIS SOFTWARE IS PROVIDED BY THE OpenSSL PROJECT ``AS IS'' AND ANY
 * EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE OpenSSL PROJECT OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * ====================================================================
 *
 * This product includes cryptographic software written by Eric Young
 * (eay@cryptsoft.com).  This product includes software written by Tim
 * Hudson (tjh@cryptsoft.com). */

#ifndef OPENSSL_HEADER_PQUEUE_H
#define OPENSSL_HEADER_PQUEUE_H

#include <openssl/base.h>

#if defined(__cplusplus)
extern "C" {
#endif


/* Priority queue.
 *
 * The priority queue maintains a linked-list of nodes, each with a unique,
 * 64-bit priority, in ascending priority order. */

typedef struct _pqueue *pqueue;

typedef struct _pitem {
  uint8_t priority[8]; /* 64-bit value in big-endian encoding */
  void *data;
  struct _pitem *next;
} pitem;

typedef struct _pitem *piterator;


/* Creating and freeing queues. */

/* pqueue_new allocates a fresh, empty priority queue object and returns it, or
 * NULL on error. */
OPENSSL_EXPORT pqueue pqueue_new(void);

/* pqueue_free frees |pq| but not any of the items it points to. Thus |pq| must
 * be empty or a memory leak will occur. */
OPENSSL_EXPORT void pqueue_free(pqueue pq);


/* Creating and freeing items. */

/* pitem_new allocates a fresh priority queue item that points at |data| and
 * has a priority given by |prio64be|, which is a 64-bit, unsigned number
 * expressed in big-endian form. It returns the fresh item, or NULL on
 * error. */
OPENSSL_EXPORT pitem *pitem_new(uint8_t prio64be[8], void *data);

/* pitem_free frees |item|, but not any data that it points to. */
OPENSSL_EXPORT void pitem_free(pitem *item);


/* Queue accessor functions */

/* pqueue_peek returns the item with the smallest priority from |pq|, or NULL
 * if empty. */
OPENSSL_EXPORT pitem *pqueue_peek(pqueue pq);

/* pqueue_find returns the item whose priority matches |prio64be| or NULL if no
 * such item exists. */
OPENSSL_EXPORT pitem *pqueue_find(pqueue pq, uint8_t *prio64be);


/* Queue mutation functions */

/* pqueue_insert inserts |item| into |pq| and returns item. */
OPENSSL_EXPORT pitem *pqueue_insert(pqueue pq, pitem *item);

/* pqueue_pop takes the item with the least priority from |pq| and returns it,
 * or NULL if |pq| is empty. */
OPENSSL_EXPORT pitem *pqueue_pop(pqueue pq);

/* pqueue_size returns the number of items in |pq|. */
OPENSSL_EXPORT size_t pqueue_size(pqueue pq);


/* Iterating */

/* pqueue_iterator returns an iterator that can be used to iterate over the
 * contents of the queue. */
OPENSSL_EXPORT piterator pqueue_iterator(pqueue pq);

/* pqueue_next returns the current value of |iter| and advances it to the next
 * position. If the iterator has advanced over all the elements, it returns
 * NULL. */
OPENSSL_EXPORT pitem *pqueue_next(piterator *iter);


#if defined(__cplusplus)
}  /* extern C */
#endif

#endif  /* OPENSSL_HEADER_PQUEUE_H */
