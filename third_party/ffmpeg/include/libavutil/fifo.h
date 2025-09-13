/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * @ingroup lavu_fifo
 * A generic FIFO API
 */

#ifndef AVUTIL_FIFO_H
#define AVUTIL_FIFO_H

#include <stddef.h>
#include <stdint.h>

#include "attributes.h"
#include "version.h"

/**
 * @defgroup lavu_fifo AVFifo
 * @ingroup lavu_data
 *
 * @{
 * A generic FIFO API
 */

typedef struct AVFifo AVFifo;

/**
 * Callback for writing or reading from a FIFO, passed to (and invoked from) the
 * av_fifo_*_cb() functions. It may be invoked multiple times from a single
 * av_fifo_*_cb() call and may process less data than the maximum size indicated
 * by nb_elems.
 *
 * @param opaque the opaque pointer provided to the av_fifo_*_cb() function
 * @param buf the buffer for reading or writing the data, depending on which
 *            av_fifo_*_cb function is called
 * @param nb_elems On entry contains the maximum number of elements that can be
 *                 read from / written into buf. On success, the callback should
 *                 update it to contain the number of elements actually written.
 *
 * @return 0 on success, a negative error code on failure (will be returned from
 *         the invoking av_fifo_*_cb() function)
 */
typedef int AVFifoCB(void *opaque, void *buf, size_t *nb_elems);

/**
 * Automatically resize the FIFO on writes, so that the data fits. This
 * automatic resizing happens up to a limit that can be modified with
 * av_fifo_auto_grow_limit().
 */
#define AV_FIFO_FLAG_AUTO_GROW      (1 << 0)

/**
 * Allocate and initialize an AVFifo with a given element size.
 *
 * @param elems     initial number of elements that can be stored in the FIFO
 * @param elem_size Size in bytes of a single element. Further operations on
 *                  the returned FIFO will implicitly use this element size.
 * @param flags a combination of AV_FIFO_FLAG_*
 *
 * @return newly-allocated AVFifo on success, a negative error code on failure
 */
AVFifo *av_fifo_alloc2(size_t elems, size_t elem_size,
                       unsigned int flags);

/**
 * @return Element size for FIFO operations. This element size is set at
 *         FIFO allocation and remains constant during its lifetime
 */
size_t av_fifo_elem_size(const AVFifo *f);

/**
 * Set the maximum size (in elements) to which the FIFO can be resized
 * automatically. Has no effect unless AV_FIFO_FLAG_AUTO_GROW is used.
 */
void av_fifo_auto_grow_limit(AVFifo *f, size_t max_elems);

/**
 * @return number of elements available for reading from the given FIFO.
 */
size_t av_fifo_can_read(const AVFifo *f);

/**
 * @return Number of elements that can be written into the given FIFO without
 *         growing it.
 *
 *         In other words, this number of elements or less is guaranteed to fit
 *         into the FIFO. More data may be written when the
 *         AV_FIFO_FLAG_AUTO_GROW flag was specified at FIFO creation, but this
 *         may involve memory allocation, which can fail.
 */
size_t av_fifo_can_write(const AVFifo *f);

/**
 * Enlarge an AVFifo.
 *
 * On success, the FIFO will be large enough to hold exactly
 * inc + av_fifo_can_read() + av_fifo_can_write()
 * elements. In case of failure, the old FIFO is kept unchanged.
 *
 * @param f AVFifo to resize
 * @param inc number of elements to allocate for, in addition to the current
 *            allocated size
 * @return a non-negative number on success, a negative error code on failure
 */
int av_fifo_grow2(AVFifo *f, size_t inc);

/**
 * Write data into a FIFO.
 *
 * In case nb_elems > av_fifo_can_write(f) and the AV_FIFO_FLAG_AUTO_GROW flag
 * was not specified at FIFO creation, nothing is written and an error
 * is returned.
 *
 * Calling function is guaranteed to succeed if nb_elems <= av_fifo_can_write(f).
 *
 * @param f the FIFO buffer
 * @param buf Data to be written. nb_elems * av_fifo_elem_size(f) bytes will be
 *            read from buf on success.
 * @param nb_elems number of elements to write into FIFO
 *
 * @return a non-negative number on success, a negative error code on failure
 */
int av_fifo_write(AVFifo *f, const void *buf, size_t nb_elems);

/**
 * Write data from a user-provided callback into a FIFO.
 *
 * @param f the FIFO buffer
 * @param read_cb Callback supplying the data to the FIFO. May be called
 *                multiple times.
 * @param opaque opaque user data to be provided to read_cb
 * @param nb_elems Should point to the maximum number of elements that can be
 *                 written. Will be updated to contain the number of elements
 *                 actually written.
 *
 * @return non-negative number on success, a negative error code on failure
 */
int av_fifo_write_from_cb(AVFifo *f, AVFifoCB read_cb,
                          void *opaque, size_t *nb_elems);

/**
 * Read data from a FIFO.
 *
 * In case nb_elems > av_fifo_can_read(f), nothing is read and an error
 * is returned.
 *
 * @param f the FIFO buffer
 * @param buf Buffer to store the data. nb_elems * av_fifo_elem_size(f) bytes
 *            will be written into buf on success.
 * @param nb_elems number of elements to read from FIFO
 *
 * @return a non-negative number on success, a negative error code on failure
 */
int av_fifo_read(AVFifo *f, void *buf, size_t nb_elems);

/**
 * Feed data from a FIFO into a user-provided callback.
 *
 * @param f the FIFO buffer
 * @param write_cb Callback the data will be supplied to. May be called
 *                 multiple times.
 * @param opaque opaque user data to be provided to write_cb
 * @param nb_elems Should point to the maximum number of elements that can be
 *                 read. Will be updated to contain the total number of elements
 *                 actually sent to the callback.
 *
 * @return non-negative number on success, a negative error code on failure
 */
int av_fifo_read_to_cb(AVFifo *f, AVFifoCB write_cb,
                       void *opaque, size_t *nb_elems);

/**
 * Read data from a FIFO without modifying FIFO state.
 *
 * Returns an error if an attempt is made to peek to nonexistent elements
 * (i.e. if offset + nb_elems is larger than av_fifo_can_read(f)).
 *
 * @param f the FIFO buffer
 * @param buf Buffer to store the data. nb_elems * av_fifo_elem_size(f) bytes
 *            will be written into buf.
 * @param nb_elems number of elements to read from FIFO
 * @param offset number of initial elements to skip.
 *
 * @return a non-negative number on success, a negative error code on failure
 */
int av_fifo_peek(const AVFifo *f, void *buf, size_t nb_elems, size_t offset);

/**
 * Feed data from a FIFO into a user-provided callback.
 *
 * @param f the FIFO buffer
 * @param write_cb Callback the data will be supplied to. May be called
 *                 multiple times.
 * @param opaque opaque user data to be provided to write_cb
 * @param nb_elems Should point to the maximum number of elements that can be
 *                 read. Will be updated to contain the total number of elements
 *                 actually sent to the callback.
 * @param offset number of initial elements to skip; offset + *nb_elems must not
 *               be larger than av_fifo_can_read(f).
 *
 * @return a non-negative number on success, a negative error code on failure
 */
int av_fifo_peek_to_cb(const AVFifo *f, AVFifoCB write_cb, void *opaque,
                       size_t *nb_elems, size_t offset);

/**
 * Discard the specified amount of data from an AVFifo.
 * @param size number of elements to discard, MUST NOT be larger than
 *             av_fifo_can_read(f)
 */
void av_fifo_drain2(AVFifo *f, size_t size);

/*
 * Empty the AVFifo.
 * @param f AVFifo to reset
 */
void av_fifo_reset2(AVFifo *f);

/**
 * Free an AVFifo and reset pointer to NULL.
 * @param f Pointer to an AVFifo to free. *f == NULL is allowed.
 */
void av_fifo_freep2(AVFifo **f);


#if FF_API_FIFO_OLD_API
typedef struct AVFifoBuffer {
    uint8_t *buffer;
    uint8_t *rptr, *wptr, *end;
    uint32_t rndx, wndx;
} AVFifoBuffer;

/**
 * Initialize an AVFifoBuffer.
 * @param size of FIFO
 * @return AVFifoBuffer or NULL in case of memory allocation failure
 * @deprecated use av_fifo_alloc2()
 */
attribute_deprecated
AVFifoBuffer *av_fifo_alloc(unsigned int size);

/**
 * Initialize an AVFifoBuffer.
 * @param nmemb number of elements
 * @param size  size of the single element
 * @return AVFifoBuffer or NULL in case of memory allocation failure
 * @deprecated use av_fifo_alloc2()
 */
attribute_deprecated
AVFifoBuffer *av_fifo_alloc_array(size_t nmemb, size_t size);

/**
 * Free an AVFifoBuffer.
 * @param f AVFifoBuffer to free
 * @deprecated use the AVFifo API with av_fifo_freep2()
 */
attribute_deprecated
void av_fifo_free(AVFifoBuffer *f);

/**
 * Free an AVFifoBuffer and reset pointer to NULL.
 * @param f AVFifoBuffer to free
 * @deprecated use the AVFifo API with av_fifo_freep2()
 */
attribute_deprecated
void av_fifo_freep(AVFifoBuffer **f);

/**
 * Reset the AVFifoBuffer to the state right after av_fifo_alloc, in particular it is emptied.
 * @param f AVFifoBuffer to reset
 * @deprecated use av_fifo_reset2() with the new AVFifo-API
 */
attribute_deprecated
void av_fifo_reset(AVFifoBuffer *f);

/**
 * Return the amount of data in bytes in the AVFifoBuffer, that is the
 * amount of data you can read from it.
 * @param f AVFifoBuffer to read from
 * @return size
 * @deprecated use av_fifo_can_read() with the new AVFifo-API
 */
attribute_deprecated
int av_fifo_size(const AVFifoBuffer *f);

/**
 * Return the amount of space in bytes in the AVFifoBuffer, that is the
 * amount of data you can write into it.
 * @param f AVFifoBuffer to write into
 * @return size
 * @deprecated use av_fifo_can_write() with the new AVFifo-API
 */
attribute_deprecated
int av_fifo_space(const AVFifoBuffer *f);

/**
 * Feed data at specific position from an AVFifoBuffer to a user-supplied callback.
 * Similar as av_fifo_gereric_read but without discarding data.
 * @param f AVFifoBuffer to read from
 * @param offset offset from current read position
 * @param buf_size number of bytes to read
 * @param func generic read function
 * @param dest data destination
 *
 * @return a non-negative number on success, a negative error code on failure
 *
 * @deprecated use the new AVFifo-API with av_fifo_peek() when func == NULL,
 *             av_fifo_peek_to_cb() otherwise
 */
attribute_deprecated
int av_fifo_generic_peek_at(AVFifoBuffer *f, void *dest, int offset, int buf_size, void (*func)(void*, void*, int));

/**
 * Feed data from an AVFifoBuffer to a user-supplied callback.
 * Similar as av_fifo_gereric_read but without discarding data.
 * @param f AVFifoBuffer to read from
 * @param buf_size number of bytes to read
 * @param func generic read function
 * @param dest data destination
 *
 * @return a non-negative number on success, a negative error code on failure
 *
 * @deprecated use the new AVFifo-API with av_fifo_peek() when func == NULL,
 *             av_fifo_peek_to_cb() otherwise
 */
attribute_deprecated
int av_fifo_generic_peek(AVFifoBuffer *f, void *dest, int buf_size, void (*func)(void*, void*, int));

/**
 * Feed data from an AVFifoBuffer to a user-supplied callback.
 * @param f AVFifoBuffer to read from
 * @param buf_size number of bytes to read
 * @param func generic read function
 * @param dest data destination
 *
 * @return a non-negative number on success, a negative error code on failure
 *
 * @deprecated use the new AVFifo-API with av_fifo_read() when func == NULL,
 *             av_fifo_read_to_cb() otherwise
 */
attribute_deprecated
int av_fifo_generic_read(AVFifoBuffer *f, void *dest, int buf_size, void (*func)(void*, void*, int));

/**
 * Feed data from a user-supplied callback to an AVFifoBuffer.
 * @param f AVFifoBuffer to write to
 * @param src data source; non-const since it may be used as a
 * modifiable context by the function defined in func
 * @param size number of bytes to write
 * @param func generic write function; the first parameter is src,
 * the second is dest_buf, the third is dest_buf_size.
 * func must return the number of bytes written to dest_buf, or <= 0 to
 * indicate no more data available to write.
 * If func is NULL, src is interpreted as a simple byte array for source data.
 * @return the number of bytes written to the FIFO or a negative error code on failure
 *
 * @deprecated use the new AVFifo-API with av_fifo_write() when func == NULL,
 *             av_fifo_write_from_cb() otherwise
 */
attribute_deprecated
int av_fifo_generic_write(AVFifoBuffer *f, void *src, int size, int (*func)(void*, void*, int));

/**
 * Resize an AVFifoBuffer.
 * In case of reallocation failure, the old FIFO is kept unchanged.
 *
 * @param f AVFifoBuffer to resize
 * @param size new AVFifoBuffer size in bytes
 * @return <0 for failure, >=0 otherwise
 *
 * @deprecated use the new AVFifo-API with av_fifo_grow2() to increase FIFO size,
 *             decreasing FIFO size is not supported
 */
attribute_deprecated
int av_fifo_realloc2(AVFifoBuffer *f, unsigned int size);

/**
 * Enlarge an AVFifoBuffer.
 * In case of reallocation failure, the old FIFO is kept unchanged.
 * The new fifo size may be larger than the requested size.
 *
 * @param f AVFifoBuffer to resize
 * @param additional_space the amount of space in bytes to allocate in addition to av_fifo_size()
 * @return <0 for failure, >=0 otherwise
 *
 * @deprecated use the new AVFifo-API with av_fifo_grow2(); note that unlike
 * this function it adds to the allocated size, rather than to the used size
 */
attribute_deprecated
int av_fifo_grow(AVFifoBuffer *f, unsigned int additional_space);

/**
 * Read and discard the specified amount of data from an AVFifoBuffer.
 * @param f AVFifoBuffer to read from
 * @param size amount of data to read in bytes
 *
 * @deprecated use the new AVFifo-API with av_fifo_drain2()
 */
attribute_deprecated
void av_fifo_drain(AVFifoBuffer *f, int size);

#if FF_API_FIFO_PEEK2
/**
 * Return a pointer to the data stored in a FIFO buffer at a certain offset.
 * The FIFO buffer is not modified.
 *
 * @param f    AVFifoBuffer to peek at, f must be non-NULL
 * @param offs an offset in bytes, its absolute value must be less
 *             than the used buffer size or the returned pointer will
 *             point outside to the buffer data.
 *             The used buffer size can be checked with av_fifo_size().
 * @deprecated use the new AVFifo-API with av_fifo_peek() or av_fifo_peek_to_cb()
 */
attribute_deprecated
static inline uint8_t *av_fifo_peek2(const AVFifoBuffer *f, int offs)
{
    uint8_t *ptr = f->rptr + offs;
    if (ptr >= f->end)
        ptr = f->buffer + (ptr - f->end);
    else if (ptr < f->buffer)
        ptr = f->end - (f->buffer - ptr);
    return ptr;
}
#endif
#endif

/**
 * @}
 */

#endif /* AVUTIL_FIFO_H */
