/*
 * Copyright (C) 2013 The Android Open Source Project
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


#ifndef ANDROID_INCLUDE_BT_GATT_H
#define ANDROID_INCLUDE_BT_GATT_H

#include <stdint.h>
#include "bt_gatt_client.h"
#include "bt_gatt_server.h"

__BEGIN_DECLS

/** BT-GATT callbacks */
typedef struct {
    /** Set to sizeof(btgatt_callbacks_t) */
    size_t size;

    /** GATT Client callbacks */
    const btgatt_client_callbacks_t* client;

    /** GATT Server callbacks */
    const btgatt_server_callbacks_t* server;
} btgatt_callbacks_t;

/** Represents the standard Bluetooth GATT interface. */
typedef struct {
    /** Set to sizeof(btgatt_interface_t) */
    size_t          size;

    /**
     * Initializes the interface and provides callback routines
     */
    bt_status_t (*init)( const btgatt_callbacks_t* callbacks );

    /** Closes the interface */
    void (*cleanup)( void );

    /** Pointer to the GATT client interface methods.*/
    const btgatt_client_interface_t* client;

    /** Pointer to the GATT server interface methods.*/
    const btgatt_server_interface_t* server;
} btgatt_interface_t;

__END_DECLS

#endif /* ANDROID_INCLUDE_BT_GATT_H */
