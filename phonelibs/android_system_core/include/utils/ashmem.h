/* utils/ashmem.h
 **
 ** Copyright 2008 The Android Open Source Project
 **
 ** This file is dual licensed.  It may be redistributed and/or modified
 ** under the terms of the Apache 2.0 License OR version 2 of the GNU
 ** General Public License.
 */

#ifndef _UTILS_ASHMEM_H
#define _UTILS_ASHMEM_H

#include <linux/limits.h>
#include <linux/ioctl.h>

#define ASHMEM_NAME_LEN		256

#define ASHMEM_NAME_DEF		"dev/ashmem"

/* Return values from ASHMEM_PIN: Was the mapping purged while unpinned? */
#define ASHMEM_NOT_REAPED	0
#define ASHMEM_WAS_REAPED	1

/* Return values from ASHMEM_UNPIN: Is the mapping now pinned or unpinned? */
#define ASHMEM_NOW_UNPINNED	0
#define ASHMEM_NOW_PINNED	1

#define __ASHMEMIOC		0x77

#define ASHMEM_SET_NAME		_IOW(__ASHMEMIOC, 1, char[ASHMEM_NAME_LEN])
#define ASHMEM_GET_NAME		_IOR(__ASHMEMIOC, 2, char[ASHMEM_NAME_LEN])
#define ASHMEM_SET_SIZE		_IOW(__ASHMEMIOC, 3, size_t)
#define ASHMEM_GET_SIZE		_IO(__ASHMEMIOC, 4)
#define ASHMEM_SET_PROT_MASK	_IOW(__ASHMEMIOC, 5, unsigned long)
#define ASHMEM_GET_PROT_MASK	_IO(__ASHMEMIOC, 6)
#define ASHMEM_PIN		_IO(__ASHMEMIOC, 7)
#define ASHMEM_UNPIN		_IO(__ASHMEMIOC, 8)
#define ASHMEM_ISPINNED		_IO(__ASHMEMIOC, 9)
#define ASHMEM_PURGE_ALL_CACHES	_IO(__ASHMEMIOC, 10)

#endif	/* _UTILS_ASHMEM_H */
