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

// FIXME: add well-defined names for cameras

#ifndef ANDROID_INCLUDE_CAMERA_COMMON_H
#define ANDROID_INCLUDE_CAMERA_COMMON_H

#include <stdint.h>
#include <stdbool.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <cutils/native_handle.h>
#include <system/camera.h>
#include <system/camera_vendor_tags.h>
#include <hardware/hardware.h>
#include <hardware/gralloc.h>

__BEGIN_DECLS

/**
 * The id of this module
 */
#define CAMERA_HARDWARE_MODULE_ID "camera"

/**
 * Module versioning information for the Camera hardware module, based on
 * camera_module_t.common.module_api_version. The two most significant hex
 * digits represent the major version, and the two least significant represent
 * the minor version.
 *
 *******************************************************************************
 * Versions: 0.X - 1.X [CAMERA_MODULE_API_VERSION_1_0]
 *
 *   Camera modules that report these version numbers implement the initial
 *   camera module HAL interface. All camera devices openable through this
 *   module support only version 1 of the camera device HAL. The device_version
 *   and static_camera_characteristics fields of camera_info are not valid. Only
 *   the android.hardware.Camera API can be supported by this module and its
 *   devices.
 *
 *******************************************************************************
 * Version: 2.0 [CAMERA_MODULE_API_VERSION_2_0]
 *
 *   Camera modules that report this version number implement the second version
 *   of the camera module HAL interface. Camera devices openable through this
 *   module may support either version 1.0 or version 2.0 of the camera device
 *   HAL interface. The device_version field of camera_info is always valid; the
 *   static_camera_characteristics field of camera_info is valid if the
 *   device_version field is 2.0 or higher.
 *
 *******************************************************************************
 * Version: 2.1 [CAMERA_MODULE_API_VERSION_2_1]
 *
 *   This camera module version adds support for asynchronous callbacks to the
 *   framework from the camera HAL module, which is used to notify the framework
 *   about changes to the camera module state. Modules that provide a valid
 *   set_callbacks() method must report at least this version number.
 *
 *******************************************************************************
 * Version: 2.2 [CAMERA_MODULE_API_VERSION_2_2]
 *
 *   This camera module version adds vendor tag support from the module, and
 *   deprecates the old vendor_tag_query_ops that were previously only
 *   accessible with a device open.
 *
 *******************************************************************************
 * Version: 2.3 [CAMERA_MODULE_API_VERSION_2_3]
 *
 *   This camera module version adds open legacy camera HAL device support.
 *   Framework can use it to open the camera device as lower device HAL version
 *   HAL device if the same device can support multiple device API versions.
 *   The standard hardware module open call (common.methods->open) continues
 *   to open the camera device with the latest supported version, which is
 *   also the version listed in camera_info_t.device_version.
 *
 *******************************************************************************
 * Version: 2.4 [CAMERA_MODULE_API_VERSION_2_4]
 *
 * This camera module version adds below API changes:
 *
 * 1. Torch mode support. The framework can use it to turn on torch mode for
 *    any camera device that has a flash unit, without opening a camera device. The
 *    camera device has a higher priority accessing the flash unit than the camera
 *    module; opening a camera device will turn off the torch if it had been enabled
 *    through the module interface. When there are any resource conflicts, such as
 *    open() is called to open a camera device, the camera HAL module must notify the
 *    framework through the torch mode status callback that the torch mode has been
 *    turned off.
 *
 * 2. External camera (e.g. USB hot-plug camera) support. The API updates specify that
 *    the camera static info is only available when camera is connected and ready to
 *    use for external hot-plug cameras. Calls to get static info will be invalid
 *    calls when camera status is not CAMERA_DEVICE_STATUS_PRESENT. The frameworks
 *    will only count on device status change callbacks to manage the available external
 *    camera list.
 *
 * 3. Camera arbitration hints. This module version adds support for explicitly
 *    indicating the number of camera devices that can be simultaneously opened and used.
 *    To specify valid combinations of devices, the resource_cost and conflicting_devices
 *    fields should always be set in the camera_info structure returned by the
 *    get_camera_info call.
 *
 * 4. Module initialization method. This will be called by the camera service
 *    right after the HAL module is loaded, to allow for one-time initialization
 *    of the HAL. It is called before any other module methods are invoked.
 */

/**
 * Predefined macros for currently-defined version numbers
 */

/**
 * All module versions <= HARDWARE_MODULE_API_VERSION(1, 0xFF) must be treated
 * as CAMERA_MODULE_API_VERSION_1_0
 */
#define CAMERA_MODULE_API_VERSION_1_0 HARDWARE_MODULE_API_VERSION(1, 0)
#define CAMERA_MODULE_API_VERSION_2_0 HARDWARE_MODULE_API_VERSION(2, 0)
#define CAMERA_MODULE_API_VERSION_2_1 HARDWARE_MODULE_API_VERSION(2, 1)
#define CAMERA_MODULE_API_VERSION_2_2 HARDWARE_MODULE_API_VERSION(2, 2)
#define CAMERA_MODULE_API_VERSION_2_3 HARDWARE_MODULE_API_VERSION(2, 3)
#define CAMERA_MODULE_API_VERSION_2_4 HARDWARE_MODULE_API_VERSION(2, 4)

#define CAMERA_MODULE_API_VERSION_CURRENT CAMERA_MODULE_API_VERSION_2_4

/**
 * All device versions <= HARDWARE_DEVICE_API_VERSION(1, 0xFF) must be treated
 * as CAMERA_DEVICE_API_VERSION_1_0
 */
#define CAMERA_DEVICE_API_VERSION_1_0 HARDWARE_DEVICE_API_VERSION(1, 0)
#define CAMERA_DEVICE_API_VERSION_2_0 HARDWARE_DEVICE_API_VERSION(2, 0)
#define CAMERA_DEVICE_API_VERSION_2_1 HARDWARE_DEVICE_API_VERSION(2, 1)
#define CAMERA_DEVICE_API_VERSION_3_0 HARDWARE_DEVICE_API_VERSION(3, 0)
#define CAMERA_DEVICE_API_VERSION_3_1 HARDWARE_DEVICE_API_VERSION(3, 1)
#define CAMERA_DEVICE_API_VERSION_3_2 HARDWARE_DEVICE_API_VERSION(3, 2)
#define CAMERA_DEVICE_API_VERSION_3_3 HARDWARE_DEVICE_API_VERSION(3, 3)

// Device version 3.3 is current, older HAL camera device versions are not
// recommended for new devices.
#define CAMERA_DEVICE_API_VERSION_CURRENT CAMERA_DEVICE_API_VERSION_3_3

/**
 * Defined in /system/media/camera/include/system/camera_metadata.h
 */
typedef struct camera_metadata camera_metadata_t;

typedef struct camera_info {
    /**
     * The direction that the camera faces to. See system/core/include/system/camera.h
     * for camera facing definitions.
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     * CAMERA_MODULE_API_VERSION_2_3 or lower:
     *
     *   It should be CAMERA_FACING_BACK or CAMERA_FACING_FRONT.
     *
     * CAMERA_MODULE_API_VERSION_2_4 or higher:
     *
     *   It should be CAMERA_FACING_BACK, CAMERA_FACING_FRONT or
     *   CAMERA_FACING_EXTERNAL.
     */
    int facing;

    /**
     * The orientation of the camera image. The value is the angle that the
     * camera image needs to be rotated clockwise so it shows correctly on the
     * display in its natural orientation. It should be 0, 90, 180, or 270.
     *
     * For example, suppose a device has a naturally tall screen. The
     * back-facing camera sensor is mounted in landscape. You are looking at the
     * screen. If the top side of the camera sensor is aligned with the right
     * edge of the screen in natural orientation, the value should be 90. If the
     * top side of a front-facing camera sensor is aligned with the right of the
     * screen, the value should be 270.
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     * CAMERA_MODULE_API_VERSION_2_3 or lower:
     *
     *   Valid in all camera_module versions.
     *
     * CAMERA_MODULE_API_VERSION_2_4 or higher:
     *
     *   Valid if camera facing is CAMERA_FACING_BACK or CAMERA_FACING_FRONT,
     *   not valid if camera facing is CAMERA_FACING_EXTERNAL.
     */
    int orientation;

    /**
     * The value of camera_device_t.common.version.
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     *  CAMERA_MODULE_API_VERSION_1_0:
     *
     *    Not valid. Can be assumed to be CAMERA_DEVICE_API_VERSION_1_0. Do
     *    not read this field.
     *
     *  CAMERA_MODULE_API_VERSION_2_0 or higher:
     *
     *    Always valid
     *
     */
    uint32_t device_version;

    /**
     * The camera's fixed characteristics, which include all static camera metadata
     * specified in system/media/camera/docs/docs.html. This should be a sorted metadata
     * buffer, and may not be modified or freed by the caller. The pointer should remain
     * valid for the lifetime of the camera module, and values in it may not
     * change after it is returned by get_camera_info().
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     *  CAMERA_MODULE_API_VERSION_1_0:
     *
     *    Not valid. Extra characteristics are not available. Do not read this
     *    field.
     *
     *  CAMERA_MODULE_API_VERSION_2_0 or higher:
     *
     *    Valid if device_version >= CAMERA_DEVICE_API_VERSION_2_0. Do not read
     *    otherwise.
     *
     */
    const camera_metadata_t *static_camera_characteristics;

    /**
     * The total resource "cost" of using this camera, represented as an integer
     * value in the range [0, 100] where 100 represents total usage of the shared
     * resource that is the limiting bottleneck of the camera subsystem.  This may
     * be a very rough estimate, and is used as a hint to the camera service to
     * determine when to disallow multiple applications from simultaneously
     * opening different cameras advertised by the camera service.
     *
     * The camera service must be able to simultaneously open and use any
     * combination of camera devices exposed by the HAL where the sum of
     * the resource costs of these cameras is <= 100.  For determining cost,
     * each camera device must be assumed to be configured and operating at
     * the maximally resource-consuming framerate and stream size settings
     * available in the configuration settings exposed for that device through
     * the camera metadata.
     *
     * The camera service may still attempt to simultaneously open combinations
     * of camera devices with a total resource cost > 100.  This may succeed or
     * fail.  If this succeeds, combinations of configurations that are not
     * supported due to resource constraints from having multiple open devices
     * should fail during the configure calls.  If the total resource cost is
     * <= 100, open and configure should never fail for any stream configuration
     * settings or other device capabilities that would normally succeed for a
     * device when it is the only open camera device.
     *
     * This field will be used to determine whether background applications are
     * allowed to use this camera device while other applications are using other
     * camera devices.  Note: multiple applications will never be allowed by the
     * camera service to simultaneously open the same camera device.
     *
     * Example use cases:
     *
     * Ex. 1: Camera Device 0 = Back Camera
     *        Camera Device 1 = Front Camera
     *   - Using both camera devices causes a large framerate slowdown due to
     *     limited ISP bandwidth.
     *
     *   Configuration:
     *
     *   Camera Device 0 - resource_cost = 51
     *                     conflicting_devices = null
     *   Camera Device 1 - resource_cost = 51
     *                     conflicting_devices = null
     *
     *   Result:
     *
     *   Since the sum of the resource costs is > 100, if a higher-priority
     *   application has either device open, no lower-priority applications will be
     *   allowed by the camera service to open either device.  If a lower-priority
     *   application is using a device that a higher-priority subsequently attempts
     *   to open, the lower-priority application will be forced to disconnect the
     *   the device.
     *
     *   If the highest-priority application chooses, it may still attempt to open
     *   both devices (since these devices are not listed as conflicting in the
     *   conflicting_devices fields), but usage of these devices may fail in the
     *   open or configure calls.
     *
     * Ex. 2: Camera Device 0 = Left Back Camera
     *        Camera Device 1 = Right Back Camera
     *        Camera Device 2 = Combined stereo camera using both right and left
     *                          back camera sensors used by devices 0, and 1
     *        Camera Device 3 = Front Camera
     *   - Due to do hardware constraints, up to two cameras may be open at once. The
     *     combined stereo camera may never be used at the same time as either of the
     *     two back camera devices (device 0, 1), and typically requires too much
     *     bandwidth to use at the same time as the front camera (device 3).
     *
     *   Configuration:
     *
     *   Camera Device 0 - resource_cost = 50
     *                     conflicting_devices = { 2 }
     *   Camera Device 1 - resource_cost = 50
     *                     conflicting_devices = { 2 }
     *   Camera Device 2 - resource_cost = 100
     *                     conflicting_devices = { 0, 1 }
     *   Camera Device 3 - resource_cost = 50
     *                     conflicting_devices = null
     *
     *   Result:
     *
     *   Based on the conflicting_devices fields, the camera service guarantees that
     *   the following sets of open devices will never be allowed: { 1, 2 }, { 0, 2 }.
     *
     *   Based on the resource_cost fields, if a high-priority foreground application
     *   is using camera device 0, a background application would be allowed to open
     *   camera device 1 or 3 (but would be forced to disconnect it again if the
     *   foreground application opened another device).
     *
     *   The highest priority application may still attempt to simultaneously open
     *   devices 0, 2, and 3, but the HAL may fail in open or configure calls for
     *   this combination.
     *
     * Ex. 3: Camera Device 0 = Back Camera
     *        Camera Device 1 = Front Camera
     *        Camera Device 2 = Low-power Front Camera that uses the same
     *                          sensor as device 1, but only exposes image stream
     *                          resolutions that can be used in low-power mode
     *  - Using both front cameras (device 1, 2) at the same time is impossible due
     *    a shared physical sensor.  Using the back and "high-power" front camera
     *    (device 1) may be impossible for some stream configurations due to hardware
     *    limitations, but the "low-power" front camera option may always be used as
     *    it has special dedicated hardware.
     *
     *   Configuration:
     *
     *   Camera Device 0 - resource_cost = 100
     *                     conflicting_devices = null
     *   Camera Device 1 - resource_cost = 100
     *                     conflicting_devices = { 2 }
     *   Camera Device 2 - resource_cost = 0
     *                     conflicting_devices = { 1 }
     *   Result:
     *
     *   Based on the conflicting_devices fields, the camera service guarantees that
     *   the following sets of open devices will never be allowed: { 1, 2 }.
     *
     *   Based on the resource_cost fields, only the highest priority application
     *   may attempt to open both device 0 and 1 at the same time. If a higher-priority
     *   application is not using device 1 or 2, a low-priority background application
     *   may open device 2 (but will be forced to disconnect it if a higher-priority
     *   application subsequently opens device 1 or 2).
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     *  CAMERA_MODULE_API_VERSION_2_3 or lower:
     *
     *    Not valid.  Can be assumed to be 100.  Do not read this field.
     *
     *  CAMERA_MODULE_API_VERSION_2_4 or higher:
     *
     *    Always valid.
     */
    int resource_cost;

    /**
     * An array of camera device IDs represented as NULL-terminated strings
     * indicating other devices that cannot be simultaneously opened while this
     * camera device is in use.
     *
     * This field is intended to be used to indicate that this camera device
     * is a composite of several other camera devices, or otherwise has
     * hardware dependencies that prohibit simultaneous usage. If there are no
     * dependencies, a NULL may be returned in this field to indicate this.
     *
     * The camera service will never simultaneously open any of the devices
     * in this list while this camera device is open.
     *
     * The strings pointed to in this field will not be cleaned up by the camera
     * service, and must remain while this device is plugged in.
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     *  CAMERA_MODULE_API_VERSION_2_3 or lower:
     *
     *    Not valid.  Can be assumed to be NULL.  Do not read this field.
     *
     *  CAMERA_MODULE_API_VERSION_2_4 or higher:
     *
     *    Always valid.
     */
    char** conflicting_devices;

    /**
     * The length of the array given in the conflicting_devices field.
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     *  CAMERA_MODULE_API_VERSION_2_3 or lower:
     *
     *    Not valid.  Can be assumed to be 0.  Do not read this field.
     *
     *  CAMERA_MODULE_API_VERSION_2_4 or higher:
     *
     *    Always valid.
     */
    size_t conflicting_devices_length;

} camera_info_t;

/**
 * camera_device_status_t:
 *
 * The current status of the camera device, as provided by the HAL through the
 * camera_module_callbacks.camera_device_status_change() call.
 *
 * At module load time, the framework will assume all camera devices are in the
 * CAMERA_DEVICE_STATUS_PRESENT state. The HAL should invoke
 * camera_module_callbacks::camera_device_status_change to inform the framework
 * of any initially NOT_PRESENT devices.
 *
 * Allowed transitions:
 *      PRESENT            -> NOT_PRESENT
 *      NOT_PRESENT        -> ENUMERATING
 *      NOT_PRESENT        -> PRESENT
 *      ENUMERATING        -> PRESENT
 *      ENUMERATING        -> NOT_PRESENT
 */
typedef enum camera_device_status {
    /**
     * The camera device is not currently connected, and opening it will return
     * failure.
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     * CAMERA_MODULE_API_VERSION_2_3 or lower:
     *
     *   Calls to get_camera_info must still succeed, and provide the same information
     *   it would if the camera were connected.
     *
     * CAMERA_MODULE_API_VERSION_2_4:
     *
     *   The camera device at this status must return -EINVAL for get_camera_info call,
     *   as the device is not connected.
     */
    CAMERA_DEVICE_STATUS_NOT_PRESENT = 0,

    /**
     * The camera device is connected, and opening it will succeed.
     *
     * CAMERA_MODULE_API_VERSION_2_3 or lower:
     *
     *   The information returned by get_camera_info cannot change due to this status
     *   change. By default, the framework will assume all devices are in this state.
     *
     * CAMERA_MODULE_API_VERSION_2_4:
     *
     *   The information returned by get_camera_info will become valid after a device's
     *   status changes to this. By default, the framework will assume all devices are in
     *   this state.
     */
    CAMERA_DEVICE_STATUS_PRESENT = 1,

    /**
     * The camera device is connected, but it is undergoing an enumeration and
     * so opening the device will return -EBUSY.
     *
     * CAMERA_MODULE_API_VERSION_2_3 or lower:
     *
     *   Calls to get_camera_info must still succeed, as if the camera was in the
     *   PRESENT status.
     *
     * CAMERA_MODULE_API_VERSION_2_4:
     *
     *   The camera device at this status must return -EINVAL for get_camera_info for call,
     *   as the device is not ready.
     */
    CAMERA_DEVICE_STATUS_ENUMERATING = 2,

} camera_device_status_t;

/**
 * torch_mode_status_t:
 *
 * The current status of the torch mode, as provided by the HAL through the
 * camera_module_callbacks.torch_mode_status_change() call.
 *
 * The torch mode status of a camera device is applicable only when the camera
 * device is present. The framework will not call set_torch_mode() to turn on
 * torch mode of a camera device if the camera device is not present. At module
 * load time, the framework will assume torch modes are in the
 * TORCH_MODE_STATUS_AVAILABLE_OFF state if the camera device is present and
 * android.flash.info.available is reported as true via get_camera_info() call.
 *
 * The behaviors of the camera HAL module that the framework expects in the
 * following situations when a camera device's status changes:
 *  1. A previously-disconnected camera device becomes connected.
 *      After camera_module_callbacks::camera_device_status_change() is invoked
 *      to inform the framework that the camera device is present, the framework
 *      will assume the camera device's torch mode is in
 *      TORCH_MODE_STATUS_AVAILABLE_OFF state. The camera HAL module does not need
 *      to invoke camera_module_callbacks::torch_mode_status_change() unless the
 *      flash unit is unavailable to use by set_torch_mode().
 *
 *  2. A previously-connected camera becomes disconnected.
 *      After camera_module_callbacks::camera_device_status_change() is invoked
 *      to inform the framework that the camera device is not present, the
 *      framework will not call set_torch_mode() for the disconnected camera
 *      device until its flash unit becomes available again. The camera HAL
 *      module does not need to invoke
 *      camera_module_callbacks::torch_mode_status_change() separately to inform
 *      that the flash unit has become unavailable.
 *
 *  3. open() is called to open a camera device.
 *      The camera HAL module must invoke
 *      camera_module_callbacks::torch_mode_status_change() for all flash units
 *      that have entered TORCH_MODE_STATUS_NOT_AVAILABLE state and can not be
 *      turned on by calling set_torch_mode() anymore due to this open() call.
 *      open() must not trigger TORCH_MODE_STATUS_AVAILABLE_OFF before
 *      TORCH_MODE_STATUS_NOT_AVAILABLE for all flash units that have become
 *      unavailable.
 *
 *  4. close() is called to close a camera device.
 *      The camera HAL module must invoke
 *      camera_module_callbacks::torch_mode_status_change() for all flash units
 *      that have entered TORCH_MODE_STATUS_AVAILABLE_OFF state and can be turned
 *      on by calling set_torch_mode() again because of enough resources freed
 *      up by this close() call.
 *
 *  Note that the framework calling set_torch_mode() successfully must trigger
 *  TORCH_MODE_STATUS_AVAILABLE_OFF or TORCH_MODE_STATUS_AVAILABLE_ON callback
 *  for the given camera device. Additionally it must trigger
 *  TORCH_MODE_STATUS_AVAILABLE_OFF callbacks for other previously-on torch
 *  modes if HAL cannot keep multiple torch modes on simultaneously.
 */
typedef enum torch_mode_status {

    /**
     * The flash unit is no longer available and the torch mode can not be
     * turned on by calling set_torch_mode(). If the torch mode is on, it
     * will be turned off by HAL before HAL calls torch_mode_status_change().
     */
    TORCH_MODE_STATUS_NOT_AVAILABLE = 0,

    /**
     * A torch mode has become off and available to be turned on via
     * set_torch_mode(). This may happen in the following
     * cases:
     *   1. After the resources to turn on the torch mode have become available.
     *   2. After set_torch_mode() is called to turn off the torch mode.
     *   3. After the framework turned on the torch mode of some other camera
     *      device and HAL had to turn off the torch modes of any camera devices
     *      that were previously on.
     */
    TORCH_MODE_STATUS_AVAILABLE_OFF = 1,

    /**
     * A torch mode has become on and available to be turned off via
     * set_torch_mode(). This can happen only after set_torch_mode() is called
     * to turn on the torch mode.
     */
    TORCH_MODE_STATUS_AVAILABLE_ON = 2,

} torch_mode_status_t;

/**
 * Callback functions for the camera HAL module to use to inform the framework
 * of changes to the camera subsystem.
 *
 * Version information (based on camera_module_t.common.module_api_version):
 *
 * Each callback is called only by HAL modules implementing the indicated
 * version or higher of the HAL module API interface.
 *
 *  CAMERA_MODULE_API_VERSION_2_1:
 *    camera_device_status_change()
 *
 *  CAMERA_MODULE_API_VERSION_2_4:
 *    torch_mode_status_change()

 */
typedef struct camera_module_callbacks {

    /**
     * camera_device_status_change:
     *
     * Callback to the framework to indicate that the state of a specific camera
     * device has changed. At module load time, the framework will assume all
     * camera devices are in the CAMERA_DEVICE_STATUS_PRESENT state. The HAL
     * must call this method to inform the framework of any initially
     * NOT_PRESENT devices.
     *
     * This callback is added for CAMERA_MODULE_API_VERSION_2_1.
     *
     * camera_module_callbacks: The instance of camera_module_callbacks_t passed
     *   to the module with set_callbacks.
     *
     * camera_id: The ID of the camera device that has a new status.
     *
     * new_status: The new status code, one of the camera_device_status_t enums,
     *   or a platform-specific status.
     *
     */
    void (*camera_device_status_change)(const struct camera_module_callbacks*,
            int camera_id,
            int new_status);

    /**
     * torch_mode_status_change:
     *
     * Callback to the framework to indicate that the state of the torch mode
     * of the flash unit associated with a specific camera device has changed.
     * At module load time, the framework will assume the torch modes are in
     * the TORCH_MODE_STATUS_AVAILABLE_OFF state if android.flash.info.available
     * is reported as true via get_camera_info() call.
     *
     * This callback is added for CAMERA_MODULE_API_VERSION_2_4.
     *
     * camera_module_callbacks: The instance of camera_module_callbacks_t
     *   passed to the module with set_callbacks.
     *
     * camera_id: The ID of camera device whose flash unit has a new torch mode
     *   status.
     *
     * new_status: The new status code, one of the torch_mode_status_t enums.
     */
    void (*torch_mode_status_change)(const struct camera_module_callbacks*,
            const char* camera_id,
            int new_status);


} camera_module_callbacks_t;

typedef struct camera_module {
    /**
     * Common methods of the camera module.  This *must* be the first member of
     * camera_module as users of this structure will cast a hw_module_t to
     * camera_module pointer in contexts where it's known the hw_module_t
     * references a camera_module.
     *
     * The return values for common.methods->open for camera_module are:
     *
     * 0:           On a successful open of the camera device.
     *
     * -ENODEV:     The camera device cannot be opened due to an internal
     *              error.
     *
     * -EINVAL:     The input arguments are invalid, i.e. the id is invalid,
     *              and/or the module is invalid.
     *
     * -EBUSY:      The camera device was already opened for this camera id
     *              (by using this method or open_legacy),
     *              regardless of the device HAL version it was opened as.
     *
     * -EUSERS:     The maximal number of camera devices that can be
     *              opened concurrently were opened already, either by
     *              this method or the open_legacy method.
     *
     * All other return values from common.methods->open will be treated as
     * -ENODEV.
     */
    hw_module_t common;

    /**
     * get_number_of_cameras:
     *
     * Returns the number of camera devices accessible through the camera
     * module.  The camera devices are numbered 0 through N-1, where N is the
     * value returned by this call. The name of the camera device for open() is
     * simply the number converted to a string. That is, "0" for camera ID 0,
     * "1" for camera ID 1.
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     * CAMERA_MODULE_API_VERSION_2_3 or lower:
     *
     *   The value here must be static, and cannot change after the first call
     *   to this method.
     *
     * CAMERA_MODULE_API_VERSION_2_4 or higher:
     *
     *   The value here must be static, and must count only built-in cameras,
     *   which have CAMERA_FACING_BACK or CAMERA_FACING_FRONT camera facing values
     *   (camera_info.facing). The HAL must not include the external cameras
     *   (camera_info.facing == CAMERA_FACING_EXTERNAL) into the return value
     *   of this call. Frameworks will use camera_device_status_change callback
     *   to manage number of external cameras.
     */
    int (*get_number_of_cameras)(void);

    /**
     * get_camera_info:
     *
     * Return the static camera information for a given camera device. This
     * information may not change for a camera device.
     *
     * Return values:
     *
     * 0:           On a successful operation
     *
     * -ENODEV:     The information cannot be provided due to an internal
     *              error.
     *
     * -EINVAL:     The input arguments are invalid, i.e. the id is invalid,
     *              and/or the module is invalid.
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     * CAMERA_MODULE_API_VERSION_2_4 or higher:
     *
     *   When a camera is disconnected, its camera id becomes invalid. Calling this
     *   this method with this invalid camera id will get -EINVAL and NULL camera
     *   static metadata (camera_info.static_camera_characteristics).
     */
    int (*get_camera_info)(int camera_id, struct camera_info *info);

    /**
     * set_callbacks:
     *
     * Provide callback function pointers to the HAL module to inform framework
     * of asynchronous camera module events. The framework will call this
     * function once after initial camera HAL module load, after the
     * get_number_of_cameras() method is called for the first time, and before
     * any other calls to the module.
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     *  CAMERA_MODULE_API_VERSION_1_0, CAMERA_MODULE_API_VERSION_2_0:
     *
     *    Not provided by HAL module. Framework may not call this function.
     *
     *  CAMERA_MODULE_API_VERSION_2_1:
     *
     *    Valid to be called by the framework.
     *
     * Return values:
     *
     * 0:           On a successful operation
     *
     * -ENODEV:     The operation cannot be completed due to an internal
     *              error.
     *
     * -EINVAL:     The input arguments are invalid, i.e. the callbacks are
     *              null
     */
    int (*set_callbacks)(const camera_module_callbacks_t *callbacks);

    /**
     * get_vendor_tag_ops:
     *
     * Get methods to query for vendor extension metadata tag information. The
     * HAL should fill in all the vendor tag operation methods, or leave ops
     * unchanged if no vendor tags are defined.
     *
     * The vendor_tag_ops structure used here is defined in:
     * system/media/camera/include/system/vendor_tags.h
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     *  CAMERA_MODULE_API_VERSION_1_x/2_0/2_1:
     *    Not provided by HAL module. Framework may not call this function.
     *
     *  CAMERA_MODULE_API_VERSION_2_2:
     *    Valid to be called by the framework.
     */
    void (*get_vendor_tag_ops)(vendor_tag_ops_t* ops);

    /**
     * open_legacy:
     *
     * Open a specific legacy camera HAL device if multiple device HAL API
     * versions are supported by this camera HAL module. For example, if the
     * camera module supports both CAMERA_DEVICE_API_VERSION_1_0 and
     * CAMERA_DEVICE_API_VERSION_3_2 device API for the same camera id,
     * framework can call this function to open the camera device as
     * CAMERA_DEVICE_API_VERSION_1_0 device.
     *
     * This is an optional method. A Camera HAL module does not need to support
     * more than one device HAL version per device, and such modules may return
     * -ENOSYS for all calls to this method. For all older HAL device API
     * versions that are not supported, it may return -EOPNOTSUPP. When above
     * cases occur, The normal open() method (common.methods->open) will be
     * used by the framework instead.
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     *  CAMERA_MODULE_API_VERSION_1_x/2_0/2_1/2_2:
     *    Not provided by HAL module. Framework will not call this function.
     *
     *  CAMERA_MODULE_API_VERSION_2_3:
     *    Valid to be called by the framework.
     *
     * Return values:
     *
     * 0:           On a successful open of the camera device.
     *
     * -ENOSYS      This method is not supported.
     *
     * -EOPNOTSUPP: The requested HAL version is not supported by this method.
     *
     * -EINVAL:     The input arguments are invalid, i.e. the id is invalid,
     *              and/or the module is invalid.
     *
     * -EBUSY:      The camera device was already opened for this camera id
     *              (by using this method or common.methods->open method),
     *              regardless of the device HAL version it was opened as.
     *
     * -EUSERS:     The maximal number of camera devices that can be
     *              opened concurrently were opened already, either by
     *              this method or common.methods->open method.
     */
    int (*open_legacy)(const struct hw_module_t* module, const char* id,
            uint32_t halVersion, struct hw_device_t** device);

    /**
     * set_torch_mode:
     *
     * Turn on or off the torch mode of the flash unit associated with a given
     * camera ID. If the operation is successful, HAL must notify the framework
     * torch state by invoking
     * camera_module_callbacks.torch_mode_status_change() with the new state.
     *
     * The camera device has a higher priority accessing the flash unit. When
     * there are any resource conflicts, such as open() is called to open a
     * camera device, HAL module must notify the framework through
     * camera_module_callbacks.torch_mode_status_change() that the
     * torch mode has been turned off and the torch mode state has become
     * TORCH_MODE_STATUS_NOT_AVAILABLE. When resources to turn on torch mode
     * become available again, HAL module must notify the framework through
     * camera_module_callbacks.torch_mode_status_change() that the torch mode
     * state has become TORCH_MODE_STATUS_AVAILABLE_OFF for set_torch_mode() to
     * be called.
     *
     * When the framework calls set_torch_mode() to turn on the torch mode of a
     * flash unit, if HAL cannot keep multiple torch modes on simultaneously,
     * HAL should turn off the torch mode that was turned on by
     * a previous set_torch_mode() call and notify the framework that the torch
     * mode state of that flash unit has become TORCH_MODE_STATUS_AVAILABLE_OFF.
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     * CAMERA_MODULE_API_VERSION_1_x/2_0/2_1/2_2/2_3:
     *   Not provided by HAL module. Framework will not call this function.
     *
     * CAMERA_MODULE_API_VERSION_2_4:
     *   Valid to be called by the framework.
     *
     * Return values:
     *
     * 0:           On a successful operation.
     *
     * -ENOSYS:     The camera device does not support this operation. It is
     *              returned if and only if android.flash.info.available is
     *              false.
     *
     * -EBUSY:      The camera device is already in use.
     *
     * -EUSERS:     The resources needed to turn on the torch mode are not
     *              available, typically because other camera devices are
     *              holding the resources to make using the flash unit not
     *              possible.
     *
     * -EINVAL:     camera_id is invalid.
     *
     */
    int (*set_torch_mode)(const char* camera_id, bool enabled);

    /**
     * init:
     *
     * This method is called by the camera service before any other methods
     * are invoked, right after the camera HAL library has been successfully
     * loaded. It may be left as NULL by the HAL module, if no initialization
     * in needed.
     *
     * It can be used by HAL implementations to perform initialization and
     * other one-time operations.
     *
     * Version information (based on camera_module_t.common.module_api_version):
     *
     * CAMERA_MODULE_API_VERSION_1_x/2_0/2_1/2_2/2_3:
     *   Not provided by HAL module. Framework will not call this function.
     *
     * CAMERA_MODULE_API_VERSION_2_4:
     *   If not NULL, will always be called by the framework once after the HAL
     *   module is loaded, before any other HAL module method is called.
     *
     * Return values:
     *
     * 0:           On a successful operation.
     *
     * -ENODEV:     Initialization cannot be completed due to an internal
     *              error. The HAL must be assumed to be in a nonfunctional
     *              state.
     *
     */
    int (*init)();

    /* reserved for future use */
    void* reserved[5];
} camera_module_t;

__END_DECLS

#endif /* ANDROID_INCLUDE_CAMERA_COMMON_H */
