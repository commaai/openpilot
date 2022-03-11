/*
 * Copyright (C) 2016 The Android Open Source Project
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

#ifndef ANDROID_INCLUDE_HARDWARE_VR_H
#define ANDROID_INCLUDE_HARDWARE_VR_H

#include <stdbool.h>
#include <sys/cdefs.h>
#include <hardware/hardware.h>

__BEGIN_DECLS

#define VR_HARDWARE_MODULE_ID "vr"

#define VR_MODULE_API_VERSION_1_0 HARDWARE_MODULE_API_VERSION(1, 0)

/**
 * Implement this HAL to receive callbacks when a virtual reality (VR)
 * application is being used.  VR applications characteristically have a number
 * of special display and performance requirements, including:
 * - Low sensor latency - Total end-to-end latency from the IMU, accelerometer,
 *   and gyro to an application-visible callback must be extremely low (<5ms
 *   typically).  This is required for HIFI sensor support.
 * - Low display latency - Total end-to-end latency from the GPU draw calls to
 *   the actual display update must be as low as possible.  This is achieved by
 *   using SurfaceFlinger in a single-buffered mode, and assuring that draw calls
 *   are synchronized with the display scanout correctly.  This behavior is
 *   exposed via an EGL extension to applications.  See below for the EGL
 *   extensions needed for this.
 * - Low-persistence display - Display persistence settings must be set as low as
 *   possible while still maintaining a reasonable brightness.  For a typical
 *   display running at 60Hz, pixels should be illuminated for <=3.5ms to be
 *   considered low-persistence.  This avoids ghosting during movements in a VR
 *   setting, and should be enabled from the lights.h HAL when
 *   BRIGHTNESS_MODE_LOW_PERSISTENCE is set.
 * - Consistent performance of the GPU and CPU - When given a mixed GPU/CPU
 *   workload for a VR application with bursts of work at regular intervals
 *   several times a frame, the CPU scheduling should ensure that the application
 *   render thread work is run consistently within 1ms of when scheduled, and
 *   completed before the end of the draw window.  To this end, a single CPU core
 *   must be reserved for solely for the currently running VR application's render
 *   thread while in VR mode, and made available in the "top-app" cpuset.
 *   Likewise, an appropriate CPU, GPU, and bus clockrate must be maintained to
 *   ensure that the rendering workload finishes within the time allotted to
 *   render each frame when the POWER_HINT_SUSTAINED_PERFORMANCE flag has been
 *   set in the power.h HAL while in VR mode when the device is not being
 *   thermally throttled.
 * - Required EGL extensions must be present - Any GPU settings required to allow
 *   the above capabilities are required, including the EGL extensions:
 *   EGL_ANDROID_create_native_client_buffer, EGL_ANDROID_front_buffer_auto_refresh,
 *   EGL_EXT_protected_content, EGL_KHR_mutable_render_buffer,
 *   EGL_KHR_reusable_sync, and EGL_KHR_wait_sync.
 * - Accurate thermal reporting - Accurate thermal temperatures and limits must be
 *   reported in the thermal.h HAL.  Specifically, the current skin temperature
 *   must accurately be reported for DEVICE_TEMPERATURE_SKIN and the
 *   vr_throttling_threshold reported for this device must accurately report the
 *   temperature limit above which the device's thermal governor throttles the
 *   CPU, GPU, and/or bus clockrates below the minimum necessary for consistent
 *   performance (see previous bullet point).
 *
 * In general, vendors implementing this HAL are expected to use set_vr_mode as a
 * hint to enable VR-specific performance tuning needed for any of the above
 * requirements, and to turn on any device features optimal for VR display
 * modes.  The set_vr_mode call may simply do nothing if no optimizations are
 * available or necessary to meet the above requirements.
 *
 * No methods in this HAL will be called concurrently from the Android framework.
 */
typedef struct vr_module {
    /**
     * Common methods of the  module.  This *must* be the first member of
     * vr_module as users of this structure may cast a hw_module_t to a
     * vr_module pointer in contexts where it's known that the hw_module_t
     * references a vr_module.
     */
    struct hw_module_t common;

    /**
     * Convenience method for the HAL implementation to set up any state needed
     * at runtime startup.  This is called once from the VrManagerService during
     * its boot phase.  No methods from this HAL will be called before init.
     */
    void (*init)(struct vr_module *module);

    /**
     * Set the VR mode state.  Possible states of the enabled parameter are:
     * false - VR mode is disabled, turn off all VR-specific settings.
     * true - VR mode is enabled, turn on all VR-specific settings.
     *
     * This is called whenever the the Android system enters or leaves VR mode.
     * This will typically occur when the user switches to or from a VR application
     * that is doing stereoscopic rendering.
     */
    void (*set_vr_mode)(struct vr_module *module, bool enabled);

    /* Reserved for future use. Must be NULL. */
    void* reserved[8 - 2];
} vr_module_t;

__END_DECLS

#endif /* ANDROID_INCLUDE_HARDWARE_VR_H */
