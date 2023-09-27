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
 * @addtogroup Storage
 * @{
 */

/**
 * @file storage_manager.h
 */

#ifndef ANDROID_STORAGE_MANAGER_H
#define ANDROID_STORAGE_MANAGER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct AStorageManager;
/**
 * {@link AStorageManager} manages application OBB storage, a pointer
 * can be obtained with AStorageManager_new().
 */
typedef struct AStorageManager AStorageManager;

/**
 * The different states of a OBB storage passed to AStorageManager_obbCallbackFunc().
 */
enum {
    /**
     * The OBB container is now mounted and ready for use. Can be returned
     * as the status for callbacks made during asynchronous OBB actions.
     */
    AOBB_STATE_MOUNTED = 1,

    /**
     * The OBB container is now unmounted and not usable. Can be returned
     * as the status for callbacks made during asynchronous OBB actions.
     */
    AOBB_STATE_UNMOUNTED = 2,

    /**
     * There was an internal system error encountered while trying to
     * mount the OBB. Can be returned as the status for callbacks made
     * during asynchronous OBB actions.
     */
    AOBB_STATE_ERROR_INTERNAL = 20,

    /**
     * The OBB could not be mounted by the system. Can be returned as the
     * status for callbacks made during asynchronous OBB actions.
     */
    AOBB_STATE_ERROR_COULD_NOT_MOUNT = 21,

    /**
     * The OBB could not be unmounted. This most likely indicates that a
     * file is in use on the OBB. Can be returned as the status for
     * callbacks made during asynchronous OBB actions.
     */
    AOBB_STATE_ERROR_COULD_NOT_UNMOUNT = 22,

    /**
     * A call was made to unmount the OBB when it was not mounted. Can be
     * returned as the status for callbacks made during asynchronous OBB
     * actions.
     */
    AOBB_STATE_ERROR_NOT_MOUNTED = 23,

    /**
     * The OBB has already been mounted. Can be returned as the status for
     * callbacks made during asynchronous OBB actions.
     */
    AOBB_STATE_ERROR_ALREADY_MOUNTED = 24,

    /**
     * The current application does not have permission to use this OBB.
     * This could be because the OBB indicates it's owned by a different
     * package. Can be returned as the status for callbacks made during
     * asynchronous OBB actions.
     */
    AOBB_STATE_ERROR_PERMISSION_DENIED = 25,
};

/**
 * Obtains a new instance of AStorageManager.
 */
AStorageManager* AStorageManager_new();

/**
 * Release AStorageManager instance.
 */
void AStorageManager_delete(AStorageManager* mgr);

/**
 * Callback function for asynchronous calls made on OBB files.
 *
 * "state" is one of the following constants:
 * - {@link AOBB_STATE_MOUNTED}
 * - {@link AOBB_STATE_UNMOUNTED}
 * - {@link AOBB_STATE_ERROR_INTERNAL}
 * - {@link AOBB_STATE_ERROR_COULD_NOT_MOUNT}
 * - {@link AOBB_STATE_ERROR_COULD_NOT_UNMOUNT}
 * - {@link AOBB_STATE_ERROR_NOT_MOUNTED}
 * - {@link AOBB_STATE_ERROR_ALREADY_MOUNTED}
 * - {@link AOBB_STATE_ERROR_PERMISSION_DENIED}
 */
typedef void (*AStorageManager_obbCallbackFunc)(const char* filename, const int32_t state, void* data);

/**
 * Attempts to mount an OBB file. This is an asynchronous operation.
 */
void AStorageManager_mountObb(AStorageManager* mgr, const char* filename, const char* key,
        AStorageManager_obbCallbackFunc cb, void* data);

/**
 * Attempts to unmount an OBB file. This is an asynchronous operation.
 */
void AStorageManager_unmountObb(AStorageManager* mgr, const char* filename, const int force,
        AStorageManager_obbCallbackFunc cb, void* data);

/**
 * Check whether an OBB is mounted.
 */
int AStorageManager_isObbMounted(AStorageManager* mgr, const char* filename);

/**
 * Get the mounted path for an OBB.
 */
const char* AStorageManager_getMountedObbPath(AStorageManager* mgr, const char* filename);


#ifdef __cplusplus
};
#endif

#endif      // ANDROID_STORAGE_MANAGER_H

/** @} */
