/*
 * Copyright (C) 2010 The Android Open Source Project
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

/**
 * @addtogroup Looper
 * @{
 */

/**
 * @file looper.h
 */

#ifndef ANDROID_LOOPER_H
#define ANDROID_LOOPER_H

#ifdef __cplusplus
extern "C" {
#endif

struct ALooper;
/**
 * ALooper
 *
 * A looper is the state tracking an event loop for a thread.
 * Loopers do not define event structures or other such things; rather
 * they are a lower-level facility to attach one or more discrete objects
 * listening for an event.  An "event" here is simply data available on
 * a file descriptor: each attached object has an associated file descriptor,
 * and waiting for "events" means (internally) polling on all of these file
 * descriptors until one or more of them have data available.
 *
 * A thread can have only one ALooper associated with it.
 */
typedef struct ALooper ALooper;

/**
 * Returns the looper associated with the calling thread, or NULL if
 * there is not one.
 */
ALooper* ALooper_forThread();

/** Option for for ALooper_prepare(). */
enum {
    /**
     * This looper will accept calls to ALooper_addFd() that do not
     * have a callback (that is provide NULL for the callback).  In
     * this case the caller of ALooper_pollOnce() or ALooper_pollAll()
     * MUST check the return from these functions to discover when
     * data is available on such fds and process it.
     */
    ALOOPER_PREPARE_ALLOW_NON_CALLBACKS = 1<<0
};

/**
 * Prepares a looper associated with the calling thread, and returns it.
 * If the thread already has a looper, it is returned.  Otherwise, a new
 * one is created, associated with the thread, and returned.
 *
 * The opts may be ALOOPER_PREPARE_ALLOW_NON_CALLBACKS or 0.
 */
ALooper* ALooper_prepare(int opts);

/** Result from ALooper_pollOnce() and ALooper_pollAll(). */
enum {
    /**
     * The poll was awoken using wake() before the timeout expired
     * and no callbacks were executed and no other file descriptors were ready.
     */
    ALOOPER_POLL_WAKE = -1,

    /**
     * Result from ALooper_pollOnce() and ALooper_pollAll():
     * One or more callbacks were executed.
     */
    ALOOPER_POLL_CALLBACK = -2,

    /**
     * Result from ALooper_pollOnce() and ALooper_pollAll():
     * The timeout expired.
     */
    ALOOPER_POLL_TIMEOUT = -3,

    /**
     * Result from ALooper_pollOnce() and ALooper_pollAll():
     * An error occurred.
     */
    ALOOPER_POLL_ERROR = -4,
};

/**
 * Acquire a reference on the given ALooper object.  This prevents the object
 * from being deleted until the reference is removed.  This is only needed
 * to safely hand an ALooper from one thread to another.
 */
void ALooper_acquire(ALooper* looper);

/**
 * Remove a reference that was previously acquired with ALooper_acquire().
 */
void ALooper_release(ALooper* looper);

/**
 * Flags for file descriptor events that a looper can monitor.
 *
 * These flag bits can be combined to monitor multiple events at once.
 */
enum {
    /**
     * The file descriptor is available for read operations.
     */
    ALOOPER_EVENT_INPUT = 1 << 0,

    /**
     * The file descriptor is available for write operations.
     */
    ALOOPER_EVENT_OUTPUT = 1 << 1,

    /**
     * The file descriptor has encountered an error condition.
     *
     * The looper always sends notifications about errors; it is not necessary
     * to specify this event flag in the requested event set.
     */
    ALOOPER_EVENT_ERROR = 1 << 2,

    /**
     * The file descriptor was hung up.
     * For example, indicates that the remote end of a pipe or socket was closed.
     *
     * The looper always sends notifications about hangups; it is not necessary
     * to specify this event flag in the requested event set.
     */
    ALOOPER_EVENT_HANGUP = 1 << 3,

    /**
     * The file descriptor is invalid.
     * For example, the file descriptor was closed prematurely.
     *
     * The looper always sends notifications about invalid file descriptors; it is not necessary
     * to specify this event flag in the requested event set.
     */
    ALOOPER_EVENT_INVALID = 1 << 4,
};

/**
 * For callback-based event loops, this is the prototype of the function
 * that is called when a file descriptor event occurs.
 * It is given the file descriptor it is associated with,
 * a bitmask of the poll events that were triggered (typically ALOOPER_EVENT_INPUT),
 * and the data pointer that was originally supplied.
 *
 * Implementations should return 1 to continue receiving callbacks, or 0
 * to have this file descriptor and callback unregistered from the looper.
 */
typedef int (*ALooper_callbackFunc)(int fd, int events, void* data);

/**
 * Waits for events to be available, with optional timeout in milliseconds.
 * Invokes callbacks for all file descriptors on which an event occurred.
 *
 * If the timeout is zero, returns immediately without blocking.
 * If the timeout is negative, waits indefinitely until an event appears.
 *
 * Returns ALOOPER_POLL_WAKE if the poll was awoken using wake() before
 * the timeout expired and no callbacks were invoked and no other file
 * descriptors were ready.
 *
 * Returns ALOOPER_POLL_CALLBACK if one or more callbacks were invoked.
 *
 * Returns ALOOPER_POLL_TIMEOUT if there was no data before the given
 * timeout expired.
 *
 * Returns ALOOPER_POLL_ERROR if an error occurred.
 *
 * Returns a value >= 0 containing an identifier (the same identifier
 * `ident` passed to ALooper_addFd()) if its file descriptor has data
 * and it has no callback function (requiring the caller here to
 * handle it).  In this (and only this) case outFd, outEvents and
 * outData will contain the poll events and data associated with the
 * fd, otherwise they will be set to NULL.
 *
 * This method does not return until it has finished invoking the appropriate callbacks
 * for all file descriptors that were signalled.
 */
int ALooper_pollOnce(int timeoutMillis, int* outFd, int* outEvents, void** outData);

/**
 * Like ALooper_pollOnce(), but performs all pending callbacks until all
 * data has been consumed or a file descriptor is available with no callback.
 * This function will never return ALOOPER_POLL_CALLBACK.
 */
int ALooper_pollAll(int timeoutMillis, int* outFd, int* outEvents, void** outData);

/**
 * Wakes the poll asynchronously.
 *
 * This method can be called on any thread.
 * This method returns immediately.
 */
void ALooper_wake(ALooper* looper);

/**
 * Adds a new file descriptor to be polled by the looper.
 * If the same file descriptor was previously added, it is replaced.
 *
 * "fd" is the file descriptor to be added.
 * "ident" is an identifier for this event, which is returned from ALooper_pollOnce().
 * The identifier must be >= 0, or ALOOPER_POLL_CALLBACK if providing a non-NULL callback.
 * "events" are the poll events to wake up on.  Typically this is ALOOPER_EVENT_INPUT.
 * "callback" is the function to call when there is an event on the file descriptor.
 * "data" is a private data pointer to supply to the callback.
 *
 * There are two main uses of this function:
 *
 * (1) If "callback" is non-NULL, then this function will be called when there is
 * data on the file descriptor.  It should execute any events it has pending,
 * appropriately reading from the file descriptor.  The 'ident' is ignored in this case.
 *
 * (2) If "callback" is NULL, the 'ident' will be returned by ALooper_pollOnce
 * when its file descriptor has data available, requiring the caller to take
 * care of processing it.
 *
 * Returns 1 if the file descriptor was added or -1 if an error occurred.
 *
 * This method can be called on any thread.
 * This method may block briefly if it needs to wake the poll.
 */
int ALooper_addFd(ALooper* looper, int fd, int ident, int events,
        ALooper_callbackFunc callback, void* data);

/**
 * Removes a previously added file descriptor from the looper.
 *
 * When this method returns, it is safe to close the file descriptor since the looper
 * will no longer have a reference to it.  However, it is possible for the callback to
 * already be running or for it to run one last time if the file descriptor was already
 * signalled.  Calling code is responsible for ensuring that this case is safely handled.
 * For example, if the callback takes care of removing itself during its own execution either
 * by returning 0 or by calling this method, then it can be guaranteed to not be invoked
 * again at any later time unless registered anew.
 *
 * Returns 1 if the file descriptor was removed, 0 if none was previously registered
 * or -1 if an error occurred.
 *
 * This method can be called on any thread.
 * This method may block briefly if it needs to wake the poll.
 */
int ALooper_removeFd(ALooper* looper, int fd);

#ifdef __cplusplus
};
#endif

#endif // ANDROID_LOOPER_H

/** @} */
