/*
 * Copyright (C) 2012 The Android Open Source Project
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

#ifndef ANDROID_INCLUDE_BT_PAN_H
#define ANDROID_INCLUDE_BT_PAN_H

__BEGIN_DECLS

#define BTPAN_ROLE_NONE      0
#define BTPAN_ROLE_PANNAP    1
#define BTPAN_ROLE_PANU      2

typedef enum {
    BTPAN_STATE_CONNECTED       = 0,
    BTPAN_STATE_CONNECTING      = 1,
    BTPAN_STATE_DISCONNECTED    = 2,
    BTPAN_STATE_DISCONNECTING   = 3
} btpan_connection_state_t;

typedef enum {
    BTPAN_STATE_ENABLED = 0,
    BTPAN_STATE_DISABLED = 1
} btpan_control_state_t;

/**
* Callback for pan connection state
*/
typedef void (*btpan_connection_state_callback)(btpan_connection_state_t state, bt_status_t error,
                                                const bt_bdaddr_t *bd_addr, int local_role, int remote_role);
typedef void (*btpan_control_state_callback)(btpan_control_state_t state, int local_role,
                                            bt_status_t error, const char* ifname);

typedef struct {
    size_t size;
    btpan_control_state_callback control_state_cb;
    btpan_connection_state_callback connection_state_cb;
} btpan_callbacks_t;
typedef struct {
    /** set to size of this struct*/
    size_t          size;
    /**
     * Initialize the pan interface and register the btpan callbacks
     */
    bt_status_t (*init)(const btpan_callbacks_t* callbacks);
    /*
     * enable the pan service by specified role. The result state of
     * enabl will be returned by btpan_control_state_callback. when pan-nap is enabled,
     * the state of connecting panu device will be notified by btpan_connection_state_callback
     */
    bt_status_t (*enable)(int local_role);
    /*
     * get current pan local role
     */
    int (*get_local_role)(void);
    /**
     * start bluetooth pan connection to the remote device by specified pan role. The result state will be
     * returned by btpan_connection_state_callback
     */
    bt_status_t (*connect)(const bt_bdaddr_t *bd_addr, int local_role, int remote_role);
    /**
     * stop bluetooth pan connection. The result state will be returned by btpan_connection_state_callback
     */
    bt_status_t (*disconnect)(const bt_bdaddr_t *bd_addr);

    /**
     * Cleanup the pan interface
     */
    void (*cleanup)(void);

} btpan_interface_t;

__END_DECLS

#endif /* ANDROID_INCLUDE_BT_PAN_H */
