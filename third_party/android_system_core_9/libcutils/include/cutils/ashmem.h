/* cutils/ashmem.h
 **
 ** Copyright 2008 The Android Open Source Project
 **
 ** This file is dual licensed.  It may be redistributed and/or modified
 ** under the terms of the Apache 2.0 License OR version 2 of the GNU
 ** General Public License.
 */

#ifndef _CUTILS_ASHMEM_H
#define _CUTILS_ASHMEM_H

#include <stddef.h>

#if defined(__BIONIC__)
#include <linux/ashmem.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

int ashmem_valid(int fd);
int ashmem_create_region(const char *name, size_t size);
int ashmem_set_prot_region(int fd, int prot);
int ashmem_pin_region(int fd, size_t offset, size_t len);
int ashmem_unpin_region(int fd, size_t offset, size_t len);
int ashmem_get_size_region(int fd);

#ifdef __cplusplus
}
#endif

#endif	/* _CUTILS_ASHMEM_H */
