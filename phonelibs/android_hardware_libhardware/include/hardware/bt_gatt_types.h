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


#ifndef ANDROID_INCLUDE_BT_GATT_TYPES_H
#define ANDROID_INCLUDE_BT_GATT_TYPES_H

#include <stdint.h>
#include <stdbool.h>

__BEGIN_DECLS

/**
 * GATT Service types
 */
#define BTGATT_SERVICE_TYPE_PRIMARY 0
#define BTGATT_SERVICE_TYPE_SECONDARY 1

/** GATT ID adding instance id tracking to the UUID */
typedef struct
{
    bt_uuid_t           uuid;
    uint8_t             inst_id;
} btgatt_gatt_id_t;

/** GATT Service ID also identifies the service type (primary/secondary) */
typedef struct
{
    btgatt_gatt_id_t    id;
    uint8_t             is_primary;
} btgatt_srvc_id_t;

/** Preferred physical Transport for GATT connection */
typedef enum
{
    GATT_TRANSPORT_AUTO,
    GATT_TRANSPORT_BREDR,
    GATT_TRANSPORT_LE
} btgatt_transport_t;

__END_DECLS

#endif /* ANDROID_INCLUDE_BT_GATT_TYPES_H */
