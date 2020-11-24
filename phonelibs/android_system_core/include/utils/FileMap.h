/*
 * Copyright (C) 2006 The Android Open Source Project
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

//
// Encapsulate a shared file mapping.
//
#ifndef __LIBS_FILE_MAP_H
#define __LIBS_FILE_MAP_H

#include <sys/types.h>

#include <utils/Compat.h>

#if defined(__MINGW32__)
// Ensure that we always pull in winsock2.h before windows.h
#ifdef HAVE_WINSOCK
#include <winsock2.h>
#endif
#include <windows.h>
#endif

namespace android {

/*
 * This represents a memory-mapped file.  It might be the entire file or
 * only part of it.  This requires a little bookkeeping because the mapping
 * needs to be aligned on page boundaries, and in some cases we'd like to
 * have multiple references to the mapped area without creating additional
 * maps.
 *
 * This always uses MAP_SHARED.
 *
 * TODO: we should be able to create a new FileMap that is a subset of
 * an existing FileMap and shares the underlying mapped pages.  Requires
 * completing the refcounting stuff and possibly introducing the notion
 * of a FileMap hierarchy.
 */
class FileMap {
public:
    FileMap(void);

    /*
     * Create a new mapping on an open file.
     *
     * Closing the file descriptor does not unmap the pages, so we don't
     * claim ownership of the fd.
     *
     * Returns "false" on failure.
     */
    bool create(const char* origFileName, int fd,
                off64_t offset, size_t length, bool readOnly);

    ~FileMap(void);

    /*
     * Return the name of the file this map came from, if known.
     */
    const char* getFileName(void) const { return mFileName; }
    
    /*
     * Get a pointer to the piece of the file we requested.
     */
    void* getDataPtr(void) const { return mDataPtr; }

    /*
     * Get the length we requested.
     */
    size_t getDataLength(void) const { return mDataLength; }

    /*
     * Get the data offset used to create this map.
     */
    off64_t getDataOffset(void) const { return mDataOffset; }

    /*
     * This maps directly to madvise() values, but allows us to avoid
     * including <sys/mman.h> everywhere.
     */
    enum MapAdvice {
        NORMAL, RANDOM, SEQUENTIAL, WILLNEED, DONTNEED
    };

    /*
     * Apply an madvise() call to the entire file.
     *
     * Returns 0 on success, -1 on failure.
     */
    int advise(MapAdvice advice);

protected:

private:
    // these are not implemented
    FileMap(const FileMap& src);
    const FileMap& operator=(const FileMap& src);

    char*       mFileName;      // original file name, if known
    void*       mBasePtr;       // base of mmap area; page aligned
    size_t      mBaseLength;    // length, measured from "mBasePtr"
    off64_t     mDataOffset;    // offset used when map was created
    void*       mDataPtr;       // start of requested data, offset from base
    size_t      mDataLength;    // length, measured from "mDataPtr"
#if defined(__MINGW32__)
    HANDLE      mFileHandle;    // Win32 file handle
    HANDLE      mFileMapping;   // Win32 file mapping handle
#endif

    static long mPageSize;
};

}; // namespace android

#endif // __LIBS_FILE_MAP_H
