/*
 * Copyright (C) 2014 The Android Open Source Project
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

#ifndef ANDROID_INCLUDE_BT_MCE_H
#define ANDROID_INCLUDE_BT_MCE_H

__BEGIN_DECLS

/** MAS instance description */
typedef struct
{
    int  id;
    int  scn;
    int  msg_types;
    char *p_name;
} btmce_mas_instance_t;

/** callback for get_remote_mas_instances */
typedef void (*btmce_remote_mas_instances_callback)(bt_status_t status, bt_bdaddr_t *bd_addr,
                                                    int num_instances, btmce_mas_instance_t *instances);

typedef struct {
    /** set to sizeof(btmce_callbacks_t) */
    size_t      size;
    btmce_remote_mas_instances_callback  remote_mas_instances_cb;
} btmce_callbacks_t;

typedef struct {
    /** set to size of this struct */
    size_t size;

    /** register BT MCE callbacks */
    bt_status_t (*init)(btmce_callbacks_t *callbacks);

    /** search for MAS instances on remote device */
    bt_status_t (*get_remote_mas_instances)(bt_bdaddr_t *bd_addr);
} btmce_interface_t;

__END_DECLS

#endif /* ANDROID_INCLUDE_BT_MCE_H */
