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


#ifndef ANDROID_INCLUDE_BT_GATT_SERVER_H
#define ANDROID_INCLUDE_BT_GATT_SERVER_H

#include <stdint.h>

#include "bt_gatt_types.h"

__BEGIN_DECLS

/** GATT value type used in response to remote read requests */
typedef struct
{
    uint8_t           value[BTGATT_MAX_ATTR_LEN];
    uint16_t          handle;
    uint16_t          offset;
    uint16_t          len;
    uint8_t           auth_req;
} btgatt_value_t;

/** GATT remote read request response type */
typedef union
{
    btgatt_value_t attr_value;
    uint16_t            handle;
} btgatt_response_t;

/** BT-GATT Server callback structure. */

/** Callback invoked in response to register_server */
typedef void (*register_server_callback)(int status, int server_if,
                bt_uuid_t *app_uuid);

/** Callback indicating that a remote device has connected or been disconnected */
typedef void (*connection_callback)(int conn_id, int server_if, int connected,
                                    bt_bdaddr_t *bda);

/** Callback invoked in response to create_service */
typedef void (*service_added_callback)(int status, int server_if,
                btgatt_srvc_id_t *srvc_id, int srvc_handle);

/** Callback indicating that an included service has been added to a service */
typedef void (*included_service_added_callback)(int status, int server_if,
                int srvc_handle, int incl_srvc_handle);

/** Callback invoked when a characteristic has been added to a service */
typedef void (*characteristic_added_callback)(int status, int server_if,
                bt_uuid_t *uuid, int srvc_handle, int char_handle);

/** Callback invoked when a descriptor has been added to a characteristic */
typedef void (*descriptor_added_callback)(int status, int server_if,
                bt_uuid_t *uuid, int srvc_handle, int descr_handle);

/** Callback invoked in response to start_service */
typedef void (*service_started_callback)(int status, int server_if,
                                         int srvc_handle);

/** Callback invoked in response to stop_service */
typedef void (*service_stopped_callback)(int status, int server_if,
                                         int srvc_handle);

/** Callback triggered when a service has been deleted */
typedef void (*service_deleted_callback)(int status, int server_if,
                                         int srvc_handle);

/**
 * Callback invoked when a remote device has requested to read a characteristic
 * or descriptor. The application must respond by calling send_response
 */
typedef void (*request_read_callback)(int conn_id, int trans_id, bt_bdaddr_t *bda,
                                      int attr_handle, int offset, bool is_long);

/**
 * Callback invoked when a remote device has requested to write to a
 * characteristic or descriptor.
 */
typedef void (*request_write_callback)(int conn_id, int trans_id, bt_bdaddr_t *bda,
                                       int attr_handle, int offset, int length,
                                       bool need_rsp, bool is_prep, uint8_t* value);

/** Callback invoked when a previously prepared write is to be executed */
typedef void (*request_exec_write_callback)(int conn_id, int trans_id,
                                            bt_bdaddr_t *bda, int exec_write);

/**
 * Callback triggered in response to send_response if the remote device
 * sends a confirmation.
 */
typedef void (*response_confirmation_callback)(int status, int handle);

/**
 * Callback confirming that a notification or indication has been sent
 * to a remote device.
 */
typedef void (*indication_sent_callback)(int conn_id, int status);

/**
 * Callback notifying an application that a remote device connection is currently congested
 * and cannot receive any more data. An application should avoid sending more data until
 * a further callback is received indicating the congestion status has been cleared.
 */
typedef void (*congestion_callback)(int conn_id, bool congested);

/** Callback invoked when the MTU for a given connection changes */
typedef void (*mtu_changed_callback)(int conn_id, int mtu);

typedef struct {
    register_server_callback        register_server_cb;
    connection_callback             connection_cb;
    service_added_callback          service_added_cb;
    included_service_added_callback included_service_added_cb;
    characteristic_added_callback   characteristic_added_cb;
    descriptor_added_callback       descriptor_added_cb;
    service_started_callback        service_started_cb;
    service_stopped_callback        service_stopped_cb;
    service_deleted_callback        service_deleted_cb;
    request_read_callback           request_read_cb;
    request_write_callback          request_write_cb;
    request_exec_write_callback     request_exec_write_cb;
    response_confirmation_callback  response_confirmation_cb;
    indication_sent_callback        indication_sent_cb;
    congestion_callback             congestion_cb;
    mtu_changed_callback            mtu_changed_cb;
} btgatt_server_callbacks_t;

/** Represents the standard BT-GATT server interface. */
typedef struct {
    /** Registers a GATT server application with the stack */
    bt_status_t (*register_server)( bt_uuid_t *uuid );

    /** Unregister a server application from the stack */
    bt_status_t (*unregister_server)(int server_if );

    /** Create a connection to a remote peripheral */
    bt_status_t (*connect)(int server_if, const bt_bdaddr_t *bd_addr,
                            bool is_direct, int transport);

    /** Disconnect an established connection or cancel a pending one */
    bt_status_t (*disconnect)(int server_if, const bt_bdaddr_t *bd_addr,
                    int conn_id );

    /** Create a new service */
    bt_status_t (*add_service)( int server_if, btgatt_srvc_id_t *srvc_id, int num_handles);

    /** Assign an included service to it's parent service */
    bt_status_t (*add_included_service)( int server_if, int service_handle, int included_handle);

    /** Add a characteristic to a service */
    bt_status_t (*add_characteristic)( int server_if,
                    int service_handle, bt_uuid_t *uuid,
                    int properties, int permissions);

    /** Add a descriptor to a given service */
    bt_status_t (*add_descriptor)(int server_if, int service_handle,
                                  bt_uuid_t *uuid, int permissions);

    /** Starts a local service */
    bt_status_t (*start_service)(int server_if, int service_handle,
                                 int transport);

    /** Stops a local service */
    bt_status_t (*stop_service)(int server_if, int service_handle);

    /** Delete a local service */
    bt_status_t (*delete_service)(int server_if, int service_handle);

    /** Send value indication to a remote device */
    bt_status_t (*send_indication)(int server_if, int attribute_handle,
                                   int conn_id, int len, int confirm,
                                   char* p_value);

    /** Send a response to a read/write operation */
    bt_status_t (*send_response)(int conn_id, int trans_id,
                                 int status, btgatt_response_t *response);

} btgatt_server_interface_t;

__END_DECLS

#endif /* ANDROID_INCLUDE_BT_GATT_CLIENT_H */
