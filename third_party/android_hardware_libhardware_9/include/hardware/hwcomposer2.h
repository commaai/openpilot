/*
 * Copyright 2015 The Android Open Source Project
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

#ifndef ANDROID_HARDWARE_HWCOMPOSER2_H
#define ANDROID_HARDWARE_HWCOMPOSER2_H

#include <sys/cdefs.h>

#include <hardware/hardware.h>

#include "hwcomposer_defs.h"

__BEGIN_DECLS

/*
 * Enums
 *
 * For most of these enums, there is an invalid value defined to be 0. This is
 * an attempt to catch uninitialized fields, and these values should not be
 * used.
 */

/* Display attributes queryable through getDisplayAttribute */
typedef enum {
    HWC2_ATTRIBUTE_INVALID = 0,

    /* Dimensions in pixels */
    HWC2_ATTRIBUTE_WIDTH = 1,
    HWC2_ATTRIBUTE_HEIGHT = 2,

    /* Vsync period in nanoseconds */
    HWC2_ATTRIBUTE_VSYNC_PERIOD = 3,

    /* Dots per thousand inches (DPI * 1000). Scaling by 1000 allows these
     * numbers to be stored in an int32_t without losing too much precision. If
     * the DPI for a configuration is unavailable or is considered unreliable,
     * the device may return -1 instead */
    HWC2_ATTRIBUTE_DPI_X = 4,
    HWC2_ATTRIBUTE_DPI_Y = 5,
} hwc2_attribute_t;

/* Blend modes, settable per layer */
typedef enum {
    HWC2_BLEND_MODE_INVALID = 0,

    /* colorOut = colorSrc */
    HWC2_BLEND_MODE_NONE = 1,

    /* colorOut = colorSrc + colorDst * (1 - alphaSrc) */
    HWC2_BLEND_MODE_PREMULTIPLIED = 2,

    /* colorOut = colorSrc * alphaSrc + colorDst * (1 - alphaSrc) */
    HWC2_BLEND_MODE_COVERAGE = 3,
} hwc2_blend_mode_t;

/* See the 'Callbacks' section for more detailed descriptions of what these
 * functions do */
typedef enum {
    HWC2_CALLBACK_INVALID = 0,
    HWC2_CALLBACK_HOTPLUG = 1,
    HWC2_CALLBACK_REFRESH = 2,
    HWC2_CALLBACK_VSYNC = 3,
} hwc2_callback_descriptor_t;

/* Optional capabilities which may be supported by some devices. The particular
 * set of supported capabilities for a given device may be retrieved using
 * getCapabilities. */
typedef enum {
    HWC2_CAPABILITY_INVALID = 0,

    /* Specifies that the device supports sideband stream layers, for which
     * buffer content updates and other synchronization will not be provided
     * through the usual validate/present cycle and must be handled by an
     * external implementation-defined mechanism. Only changes to layer state
     * (such as position, size, etc.) need to be performed through the
     * validate/present cycle. */
    HWC2_CAPABILITY_SIDEBAND_STREAM = 1,

    /* Specifies that the device will apply a color transform even when either
     * the client or the device has chosen that all layers should be composed by
     * the client. This will prevent the client from applying the color
     * transform during its composition step. */
    HWC2_CAPABILITY_SKIP_CLIENT_COLOR_TRANSFORM = 2,

    /* Specifies that the present fence must not be used as an accurate
     * representation of the actual present time of a frame.
     * This capability must never be set by HWC2 devices.
     * This capability may be set for HWC1 devices that use the
     * HWC2On1Adapter where emulation of the present fence using the retire
     * fence is not feasible.
     * In the future, CTS tests will require present time to be reliable.
     */
    HWC2_CAPABILITY_PRESENT_FENCE_IS_NOT_RELIABLE = 3,

    /* Specifies that a device is able to skip the validateDisplay call before
     * receiving a call to presentDisplay. The client will always skip
     * validateDisplay and try to call presentDisplay regardless of the changes
     * in the properties of the layers. If the device returns anything else than
     * HWC2_ERROR_NONE, it will call validateDisplay then presentDisplay again.
     * For this capability to be worthwhile the device implementation of
     * presentDisplay should fail as fast as possible in the case a
     * validateDisplay step is needed.
     */
    HWC2_CAPABILITY_SKIP_VALIDATE = 4,
} hwc2_capability_t;

/* Possible composition types for a given layer */
typedef enum {
    HWC2_COMPOSITION_INVALID = 0,

    /* The client will composite this layer into the client target buffer
     * (provided to the device through setClientTarget).
     *
     * The device must not request any composition type changes for layers of
     * this type. */
    HWC2_COMPOSITION_CLIENT = 1,

    /* The device will handle the composition of this layer through a hardware
     * overlay or other similar means.
     *
     * Upon validateDisplay, the device may request a change from this type to
     * HWC2_COMPOSITION_CLIENT. */
    HWC2_COMPOSITION_DEVICE = 2,

    /* The device will render this layer using the color set through
     * setLayerColor. If this functionality is not supported on a layer that the
     * client sets to HWC2_COMPOSITION_SOLID_COLOR, the device must request that
     * the composition type of that layer is changed to HWC2_COMPOSITION_CLIENT
     * upon the next call to validateDisplay.
     *
     * Upon validateDisplay, the device may request a change from this type to
     * HWC2_COMPOSITION_CLIENT. */
    HWC2_COMPOSITION_SOLID_COLOR = 3,

    /* Similar to DEVICE, but the position of this layer may also be set
     * asynchronously through setCursorPosition. If this functionality is not
     * supported on a layer that the client sets to HWC2_COMPOSITION_CURSOR, the
     * device must request that the composition type of that layer is changed to
     * HWC2_COMPOSITION_CLIENT upon the next call to validateDisplay.
     *
     * Upon validateDisplay, the device may request a change from this type to
     * either HWC2_COMPOSITION_DEVICE or HWC2_COMPOSITION_CLIENT. Changing to
     * HWC2_COMPOSITION_DEVICE will prevent the use of setCursorPosition but
     * still permit the device to composite the layer. */
    HWC2_COMPOSITION_CURSOR = 4,

    /* The device will handle the composition of this layer, as well as its
     * buffer updates and content synchronization. Only supported on devices
     * which provide HWC2_CAPABILITY_SIDEBAND_STREAM.
     *
     * Upon validateDisplay, the device may request a change from this type to
     * either HWC2_COMPOSITION_DEVICE or HWC2_COMPOSITION_CLIENT, but it is
     * unlikely that content will display correctly in these cases. */
    HWC2_COMPOSITION_SIDEBAND = 5,
} hwc2_composition_t;

/* Possible connection options from the hotplug callback */
typedef enum {
    HWC2_CONNECTION_INVALID = 0,

    /* The display has been connected */
    HWC2_CONNECTION_CONNECTED = 1,

    /* The display has been disconnected */
    HWC2_CONNECTION_DISCONNECTED = 2,
} hwc2_connection_t;

/* Display requests returned by getDisplayRequests */
typedef enum {
    /* Instructs the client to provide a new client target buffer, even if no
     * layers are marked for client composition. */
    HWC2_DISPLAY_REQUEST_FLIP_CLIENT_TARGET = 1 << 0,

    /* Instructs the client to write the result of client composition directly
     * into the virtual display output buffer. If any of the layers are not
     * marked as HWC2_COMPOSITION_CLIENT or the given display is not a virtual
     * display, this request has no effect. */
    HWC2_DISPLAY_REQUEST_WRITE_CLIENT_TARGET_TO_OUTPUT = 1 << 1,
} hwc2_display_request_t;

/* Display types returned by getDisplayType */
typedef enum {
    HWC2_DISPLAY_TYPE_INVALID = 0,

    /* All physical displays, including both internal displays and hotpluggable
     * external displays */
    HWC2_DISPLAY_TYPE_PHYSICAL = 1,

    /* Virtual displays created by createVirtualDisplay */
    HWC2_DISPLAY_TYPE_VIRTUAL = 2,
} hwc2_display_type_t;

/* Return codes from all functions */
typedef enum {
    HWC2_ERROR_NONE = 0,
    HWC2_ERROR_BAD_CONFIG,
    HWC2_ERROR_BAD_DISPLAY,
    HWC2_ERROR_BAD_LAYER,
    HWC2_ERROR_BAD_PARAMETER,
    HWC2_ERROR_HAS_CHANGES,
    HWC2_ERROR_NO_RESOURCES,
    HWC2_ERROR_NOT_VALIDATED,
    HWC2_ERROR_UNSUPPORTED,
} hwc2_error_t;

/* Function descriptors for use with getFunction */
typedef enum {
    HWC2_FUNCTION_INVALID = 0,
    HWC2_FUNCTION_ACCEPT_DISPLAY_CHANGES,
    HWC2_FUNCTION_CREATE_LAYER,
    HWC2_FUNCTION_CREATE_VIRTUAL_DISPLAY,
    HWC2_FUNCTION_DESTROY_LAYER,
    HWC2_FUNCTION_DESTROY_VIRTUAL_DISPLAY,
    HWC2_FUNCTION_DUMP,
    HWC2_FUNCTION_GET_ACTIVE_CONFIG,
    HWC2_FUNCTION_GET_CHANGED_COMPOSITION_TYPES,
    HWC2_FUNCTION_GET_CLIENT_TARGET_SUPPORT,
    HWC2_FUNCTION_GET_COLOR_MODES,
    HWC2_FUNCTION_GET_DISPLAY_ATTRIBUTE,
    HWC2_FUNCTION_GET_DISPLAY_CONFIGS,
    HWC2_FUNCTION_GET_DISPLAY_NAME,
    HWC2_FUNCTION_GET_DISPLAY_REQUESTS,
    HWC2_FUNCTION_GET_DISPLAY_TYPE,
    HWC2_FUNCTION_GET_DOZE_SUPPORT,
    HWC2_FUNCTION_GET_HDR_CAPABILITIES,
    HWC2_FUNCTION_GET_MAX_VIRTUAL_DISPLAY_COUNT,
    HWC2_FUNCTION_GET_RELEASE_FENCES,
    HWC2_FUNCTION_PRESENT_DISPLAY,
    HWC2_FUNCTION_REGISTER_CALLBACK,
    HWC2_FUNCTION_SET_ACTIVE_CONFIG,
    HWC2_FUNCTION_SET_CLIENT_TARGET,
    HWC2_FUNCTION_SET_COLOR_MODE,
    HWC2_FUNCTION_SET_COLOR_TRANSFORM,
    HWC2_FUNCTION_SET_CURSOR_POSITION,
    HWC2_FUNCTION_SET_LAYER_BLEND_MODE,
    HWC2_FUNCTION_SET_LAYER_BUFFER,
    HWC2_FUNCTION_SET_LAYER_COLOR,
    HWC2_FUNCTION_SET_LAYER_COMPOSITION_TYPE,
    HWC2_FUNCTION_SET_LAYER_DATASPACE,
    HWC2_FUNCTION_SET_LAYER_DISPLAY_FRAME,
    HWC2_FUNCTION_SET_LAYER_PLANE_ALPHA,
    HWC2_FUNCTION_SET_LAYER_SIDEBAND_STREAM,
    HWC2_FUNCTION_SET_LAYER_SOURCE_CROP,
    HWC2_FUNCTION_SET_LAYER_SURFACE_DAMAGE,
    HWC2_FUNCTION_SET_LAYER_TRANSFORM,
    HWC2_FUNCTION_SET_LAYER_VISIBLE_REGION,
    HWC2_FUNCTION_SET_LAYER_Z_ORDER,
    HWC2_FUNCTION_SET_OUTPUT_BUFFER,
    HWC2_FUNCTION_SET_POWER_MODE,
    HWC2_FUNCTION_SET_VSYNC_ENABLED,
    HWC2_FUNCTION_VALIDATE_DISPLAY,
    HWC2_FUNCTION_SET_LAYER_FLOAT_COLOR,
    HWC2_FUNCTION_SET_LAYER_PER_FRAME_METADATA,
    HWC2_FUNCTION_GET_PER_FRAME_METADATA_KEYS,
    HWC2_FUNCTION_SET_READBACK_BUFFER,
    HWC2_FUNCTION_GET_READBACK_BUFFER_ATTRIBUTES,
    HWC2_FUNCTION_GET_READBACK_BUFFER_FENCE,
    HWC2_FUNCTION_GET_RENDER_INTENTS,
    HWC2_FUNCTION_SET_COLOR_MODE_WITH_RENDER_INTENT,
    HWC2_FUNCTION_GET_DATASPACE_SATURATION_MATRIX
} hwc2_function_descriptor_t;

/* Layer requests returned from getDisplayRequests */
typedef enum {
    /* The client should clear its target with transparent pixels where this
     * layer would be. The client may ignore this request if the layer must be
     * blended. */
    HWC2_LAYER_REQUEST_CLEAR_CLIENT_TARGET = 1 << 0,
} hwc2_layer_request_t;

/* Power modes for use with setPowerMode */
typedef enum {
    /* The display is fully off (blanked) */
    HWC2_POWER_MODE_OFF = 0,

    /* These are optional low power modes. getDozeSupport may be called to
     * determine whether a given display supports these modes. */

    /* The display is turned on and configured in a low power state that is
     * suitable for presenting ambient information to the user, possibly with
     * lower fidelity than HWC2_POWER_MODE_ON, but with greater efficiency. */
    HWC2_POWER_MODE_DOZE = 1,

    /* The display is configured as in HWC2_POWER_MODE_DOZE but may stop
     * applying display updates from the client. This is effectively a hint to
     * the device that drawing to the display has been suspended and that the
     * the device should remain on in a low power state and continue displaying
     * its current contents indefinitely until the power mode changes.
     *
     * This mode may also be used as a signal to enable hardware-based doze
     * functionality. In this case, the device is free to take over the display
     * and manage it autonomously to implement a low power always-on display. */
    HWC2_POWER_MODE_DOZE_SUSPEND = 3,

    /* The display is fully on */
    HWC2_POWER_MODE_ON = 2,
} hwc2_power_mode_t;

/* Vsync values passed to setVsyncEnabled */
typedef enum {
    HWC2_VSYNC_INVALID = 0,

    /* Enable vsync */
    HWC2_VSYNC_ENABLE = 1,

    /* Disable vsync */
    HWC2_VSYNC_DISABLE = 2,
} hwc2_vsync_t;

/* MUST match HIDL's V2_2::IComposerClient::PerFrameMetadataKey */
typedef enum {
    /* SMPTE ST 2084:2014.
     * Coordinates defined in CIE 1931 xy chromaticity space
     */
    HWC2_DISPLAY_RED_PRIMARY_X = 0,
    HWC2_DISPLAY_RED_PRIMARY_Y = 1,
    HWC2_DISPLAY_GREEN_PRIMARY_X = 2,
    HWC2_DISPLAY_GREEN_PRIMARY_Y = 3,
    HWC2_DISPLAY_BLUE_PRIMARY_X = 4,
    HWC2_DISPLAY_BLUE_PRIMARY_Y = 5,
    HWC2_WHITE_POINT_X = 6,
    HWC2_WHITE_POINT_Y = 7,
    /* SMPTE ST 2084:2014.
     * Units: nits
     * max as defined by ST 2048: 10,000 nits
     */
    HWC2_MAX_LUMINANCE = 8,
    HWC2_MIN_LUMINANCE = 9,

    /* CTA 861.3
     * Units: nits
     */
    HWC2_MAX_CONTENT_LIGHT_LEVEL = 10,
    HWC2_MAX_FRAME_AVERAGE_LIGHT_LEVEL = 11,
} hwc2_per_frame_metadata_key_t;

/*
 * Stringification Functions
 */

#ifdef HWC2_INCLUDE_STRINGIFICATION

static inline const char* getAttributeName(hwc2_attribute_t attribute) {
    switch (attribute) {
        case HWC2_ATTRIBUTE_INVALID: return "Invalid";
        case HWC2_ATTRIBUTE_WIDTH: return "Width";
        case HWC2_ATTRIBUTE_HEIGHT: return "Height";
        case HWC2_ATTRIBUTE_VSYNC_PERIOD: return "VsyncPeriod";
        case HWC2_ATTRIBUTE_DPI_X: return "DpiX";
        case HWC2_ATTRIBUTE_DPI_Y: return "DpiY";
        default: return "Unknown";
    }
}

static inline const char* getBlendModeName(hwc2_blend_mode_t mode) {
    switch (mode) {
        case HWC2_BLEND_MODE_INVALID: return "Invalid";
        case HWC2_BLEND_MODE_NONE: return "None";
        case HWC2_BLEND_MODE_PREMULTIPLIED: return "Premultiplied";
        case HWC2_BLEND_MODE_COVERAGE: return "Coverage";
        default: return "Unknown";
    }
}

static inline const char* getCallbackDescriptorName(
        hwc2_callback_descriptor_t desc) {
    switch (desc) {
        case HWC2_CALLBACK_INVALID: return "Invalid";
        case HWC2_CALLBACK_HOTPLUG: return "Hotplug";
        case HWC2_CALLBACK_REFRESH: return "Refresh";
        case HWC2_CALLBACK_VSYNC: return "Vsync";
        default: return "Unknown";
    }
}

static inline const char* getCapabilityName(hwc2_capability_t capability) {
    switch (capability) {
        case HWC2_CAPABILITY_INVALID: return "Invalid";
        case HWC2_CAPABILITY_SIDEBAND_STREAM: return "SidebandStream";
        case HWC2_CAPABILITY_SKIP_CLIENT_COLOR_TRANSFORM:
                return "SkipClientColorTransform";
        case HWC2_CAPABILITY_PRESENT_FENCE_IS_NOT_RELIABLE:
                return "PresentFenceIsNotReliable";
        default: return "Unknown";
    }
}

static inline const char* getCompositionName(hwc2_composition_t composition) {
    switch (composition) {
        case HWC2_COMPOSITION_INVALID: return "Invalid";
        case HWC2_COMPOSITION_CLIENT: return "Client";
        case HWC2_COMPOSITION_DEVICE: return "Device";
        case HWC2_COMPOSITION_SOLID_COLOR: return "SolidColor";
        case HWC2_COMPOSITION_CURSOR: return "Cursor";
        case HWC2_COMPOSITION_SIDEBAND: return "Sideband";
        default: return "Unknown";
    }
}

static inline const char* getConnectionName(hwc2_connection_t connection) {
    switch (connection) {
        case HWC2_CONNECTION_INVALID: return "Invalid";
        case HWC2_CONNECTION_CONNECTED: return "Connected";
        case HWC2_CONNECTION_DISCONNECTED: return "Disconnected";
        default: return "Unknown";
    }
}

static inline const char* getDisplayRequestName(
        hwc2_display_request_t request) {
    switch (__BIONIC_CAST(static_cast, int, request)) {
        case 0: return "None";
        case HWC2_DISPLAY_REQUEST_FLIP_CLIENT_TARGET: return "FlipClientTarget";
        case HWC2_DISPLAY_REQUEST_WRITE_CLIENT_TARGET_TO_OUTPUT:
            return "WriteClientTargetToOutput";
        case HWC2_DISPLAY_REQUEST_FLIP_CLIENT_TARGET |
                HWC2_DISPLAY_REQUEST_WRITE_CLIENT_TARGET_TO_OUTPUT:
            return "FlipClientTarget|WriteClientTargetToOutput";
        default: return "Unknown";
    }
}

static inline const char* getDisplayTypeName(hwc2_display_type_t type) {
    switch (type) {
        case HWC2_DISPLAY_TYPE_INVALID: return "Invalid";
        case HWC2_DISPLAY_TYPE_PHYSICAL: return "Physical";
        case HWC2_DISPLAY_TYPE_VIRTUAL: return "Virtual";
        default: return "Unknown";
    }
}

static inline const char* getErrorName(hwc2_error_t error) {
    switch (error) {
        case HWC2_ERROR_NONE: return "None";
        case HWC2_ERROR_BAD_CONFIG: return "BadConfig";
        case HWC2_ERROR_BAD_DISPLAY: return "BadDisplay";
        case HWC2_ERROR_BAD_LAYER: return "BadLayer";
        case HWC2_ERROR_BAD_PARAMETER: return "BadParameter";
        case HWC2_ERROR_HAS_CHANGES: return "HasChanges";
        case HWC2_ERROR_NO_RESOURCES: return "NoResources";
        case HWC2_ERROR_NOT_VALIDATED: return "NotValidated";
        case HWC2_ERROR_UNSUPPORTED: return "Unsupported";
        default: return "Unknown";
    }
}

static inline const char* getFunctionDescriptorName(
        hwc2_function_descriptor_t desc) {
    switch (desc) {
        case HWC2_FUNCTION_INVALID: return "Invalid";
        case HWC2_FUNCTION_ACCEPT_DISPLAY_CHANGES:
            return "AcceptDisplayChanges";
        case HWC2_FUNCTION_CREATE_LAYER: return "CreateLayer";
        case HWC2_FUNCTION_CREATE_VIRTUAL_DISPLAY:
            return "CreateVirtualDisplay";
        case HWC2_FUNCTION_DESTROY_LAYER: return "DestroyLayer";
        case HWC2_FUNCTION_DESTROY_VIRTUAL_DISPLAY:
            return "DestroyVirtualDisplay";
        case HWC2_FUNCTION_DUMP: return "Dump";
        case HWC2_FUNCTION_GET_ACTIVE_CONFIG: return "GetActiveConfig";
        case HWC2_FUNCTION_GET_CHANGED_COMPOSITION_TYPES:
            return "GetChangedCompositionTypes";
        case HWC2_FUNCTION_GET_CLIENT_TARGET_SUPPORT:
            return "GetClientTargetSupport";
        case HWC2_FUNCTION_GET_COLOR_MODES: return "GetColorModes";
        case HWC2_FUNCTION_GET_DISPLAY_ATTRIBUTE: return "GetDisplayAttribute";
        case HWC2_FUNCTION_GET_DISPLAY_CONFIGS: return "GetDisplayConfigs";
        case HWC2_FUNCTION_GET_DISPLAY_NAME: return "GetDisplayName";
        case HWC2_FUNCTION_GET_DISPLAY_REQUESTS: return "GetDisplayRequests";
        case HWC2_FUNCTION_GET_DISPLAY_TYPE: return "GetDisplayType";
        case HWC2_FUNCTION_GET_DOZE_SUPPORT: return "GetDozeSupport";
        case HWC2_FUNCTION_GET_HDR_CAPABILITIES: return "GetHdrCapabilities";
        case HWC2_FUNCTION_GET_MAX_VIRTUAL_DISPLAY_COUNT:
            return "GetMaxVirtualDisplayCount";
        case HWC2_FUNCTION_GET_RELEASE_FENCES: return "GetReleaseFences";
        case HWC2_FUNCTION_PRESENT_DISPLAY: return "PresentDisplay";
        case HWC2_FUNCTION_REGISTER_CALLBACK: return "RegisterCallback";
        case HWC2_FUNCTION_SET_ACTIVE_CONFIG: return "SetActiveConfig";
        case HWC2_FUNCTION_SET_CLIENT_TARGET: return "SetClientTarget";
        case HWC2_FUNCTION_SET_COLOR_MODE: return "SetColorMode";
        case HWC2_FUNCTION_SET_COLOR_TRANSFORM: return "SetColorTransform";
        case HWC2_FUNCTION_SET_CURSOR_POSITION: return "SetCursorPosition";
        case HWC2_FUNCTION_SET_LAYER_BLEND_MODE: return "SetLayerBlendMode";
        case HWC2_FUNCTION_SET_LAYER_BUFFER: return "SetLayerBuffer";
        case HWC2_FUNCTION_SET_LAYER_COLOR: return "SetLayerColor";
        case HWC2_FUNCTION_SET_LAYER_COMPOSITION_TYPE:
            return "SetLayerCompositionType";
        case HWC2_FUNCTION_SET_LAYER_DATASPACE: return "SetLayerDataspace";
        case HWC2_FUNCTION_SET_LAYER_DISPLAY_FRAME:
            return "SetLayerDisplayFrame";
        case HWC2_FUNCTION_SET_LAYER_PLANE_ALPHA: return "SetLayerPlaneAlpha";
        case HWC2_FUNCTION_SET_LAYER_SIDEBAND_STREAM:
            return "SetLayerSidebandStream";
        case HWC2_FUNCTION_SET_LAYER_SOURCE_CROP: return "SetLayerSourceCrop";
        case HWC2_FUNCTION_SET_LAYER_SURFACE_DAMAGE:
            return "SetLayerSurfaceDamage";
        case HWC2_FUNCTION_SET_LAYER_TRANSFORM: return "SetLayerTransform";
        case HWC2_FUNCTION_SET_LAYER_VISIBLE_REGION:
            return "SetLayerVisibleRegion";
        case HWC2_FUNCTION_SET_LAYER_Z_ORDER: return "SetLayerZOrder";
        case HWC2_FUNCTION_SET_OUTPUT_BUFFER: return "SetOutputBuffer";
        case HWC2_FUNCTION_SET_POWER_MODE: return "SetPowerMode";
        case HWC2_FUNCTION_SET_VSYNC_ENABLED: return "SetVsyncEnabled";
        case HWC2_FUNCTION_VALIDATE_DISPLAY: return "ValidateDisplay";
        case HWC2_FUNCTION_SET_LAYER_FLOAT_COLOR: return "SetLayerFloatColor";
        case HWC2_FUNCTION_SET_LAYER_PER_FRAME_METADATA: return "SetLayerPerFrameMetadata";
        case HWC2_FUNCTION_GET_PER_FRAME_METADATA_KEYS: return "GetPerFrameMetadataKeys";
        case HWC2_FUNCTION_SET_READBACK_BUFFER: return "SetReadbackBuffer";
        case HWC2_FUNCTION_GET_READBACK_BUFFER_ATTRIBUTES: return "GetReadbackBufferAttributes";
        case HWC2_FUNCTION_GET_READBACK_BUFFER_FENCE: return "GetReadbackBufferFence";
        case HWC2_FUNCTION_GET_RENDER_INTENTS: return "GetRenderIntents";
        case HWC2_FUNCTION_SET_COLOR_MODE_WITH_RENDER_INTENT: return "SetColorModeWithRenderIntent";
        case HWC2_FUNCTION_GET_DATASPACE_SATURATION_MATRIX: return "GetDataspaceSaturationMatrix";
        default: return "Unknown";
    }
}

static inline const char* getLayerRequestName(hwc2_layer_request_t request) {
    switch (__BIONIC_CAST(static_cast, int, request)) {
        case 0: return "None";
        case HWC2_LAYER_REQUEST_CLEAR_CLIENT_TARGET: return "ClearClientTarget";
        default: return "Unknown";
    }
}

static inline const char* getPowerModeName(hwc2_power_mode_t mode) {
    switch (mode) {
        case HWC2_POWER_MODE_OFF: return "Off";
        case HWC2_POWER_MODE_DOZE_SUSPEND: return "DozeSuspend";
        case HWC2_POWER_MODE_DOZE: return "Doze";
        case HWC2_POWER_MODE_ON: return "On";
        default: return "Unknown";
    }
}

static inline const char* getTransformName(hwc_transform_t transform) {
    switch (__BIONIC_CAST(static_cast, int, transform)) {
        case 0: return "None";
        case HWC_TRANSFORM_FLIP_H: return "FlipH";
        case HWC_TRANSFORM_FLIP_V: return "FlipV";
        case HWC_TRANSFORM_ROT_90: return "Rotate90";
        case HWC_TRANSFORM_ROT_180: return "Rotate180";
        case HWC_TRANSFORM_ROT_270: return "Rotate270";
        case HWC_TRANSFORM_FLIP_H_ROT_90: return "FlipHRotate90";
        case HWC_TRANSFORM_FLIP_V_ROT_90: return "FlipVRotate90";
        default: return "Unknown";
    }
}

static inline const char* getVsyncName(hwc2_vsync_t vsync) {
    switch (vsync) {
        case HWC2_VSYNC_INVALID: return "Invalid";
        case HWC2_VSYNC_ENABLE: return "Enable";
        case HWC2_VSYNC_DISABLE: return "Disable";
        default: return "Unknown";
    }
}

#define TO_STRING(E, T, printer) \
    inline std::string to_string(E value) { return printer(value); } \
    inline std::string to_string(T value) { return to_string(static_cast<E>(value)); }
#else // !HWC2_INCLUDE_STRINGIFICATION
#define TO_STRING(name, printer)
#endif // HWC2_INCLUDE_STRINGIFICATION

/*
 * C++11 features
 */

#ifdef HWC2_USE_CPP11
__END_DECLS

#ifdef HWC2_INCLUDE_STRINGIFICATION
#include <string>
#endif

namespace HWC2 {

enum class Attribute : int32_t {
    Invalid = HWC2_ATTRIBUTE_INVALID,
    Width = HWC2_ATTRIBUTE_WIDTH,
    Height = HWC2_ATTRIBUTE_HEIGHT,
    VsyncPeriod = HWC2_ATTRIBUTE_VSYNC_PERIOD,
    DpiX = HWC2_ATTRIBUTE_DPI_X,
    DpiY = HWC2_ATTRIBUTE_DPI_Y,
};
TO_STRING(hwc2_attribute_t, Attribute, getAttributeName)

enum class BlendMode : int32_t {
    Invalid = HWC2_BLEND_MODE_INVALID,
    None = HWC2_BLEND_MODE_NONE,
    Premultiplied = HWC2_BLEND_MODE_PREMULTIPLIED,
    Coverage = HWC2_BLEND_MODE_COVERAGE,
};
TO_STRING(hwc2_blend_mode_t, BlendMode, getBlendModeName)

enum class Callback : int32_t {
    Invalid = HWC2_CALLBACK_INVALID,
    Hotplug = HWC2_CALLBACK_HOTPLUG,
    Refresh = HWC2_CALLBACK_REFRESH,
    Vsync = HWC2_CALLBACK_VSYNC,
};
TO_STRING(hwc2_callback_descriptor_t, Callback, getCallbackDescriptorName)

enum class Capability : int32_t {
    Invalid = HWC2_CAPABILITY_INVALID,
    SidebandStream = HWC2_CAPABILITY_SIDEBAND_STREAM,
    SkipClientColorTransform = HWC2_CAPABILITY_SKIP_CLIENT_COLOR_TRANSFORM,
    PresentFenceIsNotReliable = HWC2_CAPABILITY_PRESENT_FENCE_IS_NOT_RELIABLE,
    SkipValidate = HWC2_CAPABILITY_SKIP_VALIDATE,
};
TO_STRING(hwc2_capability_t, Capability, getCapabilityName)

enum class Composition : int32_t {
    Invalid = HWC2_COMPOSITION_INVALID,
    Client = HWC2_COMPOSITION_CLIENT,
    Device = HWC2_COMPOSITION_DEVICE,
    SolidColor = HWC2_COMPOSITION_SOLID_COLOR,
    Cursor = HWC2_COMPOSITION_CURSOR,
    Sideband = HWC2_COMPOSITION_SIDEBAND,
};
TO_STRING(hwc2_composition_t, Composition, getCompositionName)

enum class Connection : int32_t {
    Invalid = HWC2_CONNECTION_INVALID,
    Connected = HWC2_CONNECTION_CONNECTED,
    Disconnected = HWC2_CONNECTION_DISCONNECTED,
};
TO_STRING(hwc2_connection_t, Connection, getConnectionName)

enum class DisplayRequest : int32_t {
    FlipClientTarget = HWC2_DISPLAY_REQUEST_FLIP_CLIENT_TARGET,
    WriteClientTargetToOutput =
        HWC2_DISPLAY_REQUEST_WRITE_CLIENT_TARGET_TO_OUTPUT,
};
TO_STRING(hwc2_display_request_t, DisplayRequest, getDisplayRequestName)

enum class DisplayType : int32_t {
    Invalid = HWC2_DISPLAY_TYPE_INVALID,
    Physical = HWC2_DISPLAY_TYPE_PHYSICAL,
    Virtual = HWC2_DISPLAY_TYPE_VIRTUAL,
};
TO_STRING(hwc2_display_type_t, DisplayType, getDisplayTypeName)

enum class Error : int32_t {
    None = HWC2_ERROR_NONE,
    BadConfig = HWC2_ERROR_BAD_CONFIG,
    BadDisplay = HWC2_ERROR_BAD_DISPLAY,
    BadLayer = HWC2_ERROR_BAD_LAYER,
    BadParameter = HWC2_ERROR_BAD_PARAMETER,
    HasChanges = HWC2_ERROR_HAS_CHANGES,
    NoResources = HWC2_ERROR_NO_RESOURCES,
    NotValidated = HWC2_ERROR_NOT_VALIDATED,
    Unsupported = HWC2_ERROR_UNSUPPORTED,
};
TO_STRING(hwc2_error_t, Error, getErrorName)

enum class FunctionDescriptor : int32_t {
    Invalid = HWC2_FUNCTION_INVALID,
    AcceptDisplayChanges = HWC2_FUNCTION_ACCEPT_DISPLAY_CHANGES,
    CreateLayer = HWC2_FUNCTION_CREATE_LAYER,
    CreateVirtualDisplay = HWC2_FUNCTION_CREATE_VIRTUAL_DISPLAY,
    DestroyLayer = HWC2_FUNCTION_DESTROY_LAYER,
    DestroyVirtualDisplay = HWC2_FUNCTION_DESTROY_VIRTUAL_DISPLAY,
    Dump = HWC2_FUNCTION_DUMP,
    GetActiveConfig = HWC2_FUNCTION_GET_ACTIVE_CONFIG,
    GetChangedCompositionTypes = HWC2_FUNCTION_GET_CHANGED_COMPOSITION_TYPES,
    GetClientTargetSupport = HWC2_FUNCTION_GET_CLIENT_TARGET_SUPPORT,
    GetColorModes = HWC2_FUNCTION_GET_COLOR_MODES,
    GetDisplayAttribute = HWC2_FUNCTION_GET_DISPLAY_ATTRIBUTE,
    GetDisplayConfigs = HWC2_FUNCTION_GET_DISPLAY_CONFIGS,
    GetDisplayName = HWC2_FUNCTION_GET_DISPLAY_NAME,
    GetDisplayRequests = HWC2_FUNCTION_GET_DISPLAY_REQUESTS,
    GetDisplayType = HWC2_FUNCTION_GET_DISPLAY_TYPE,
    GetDozeSupport = HWC2_FUNCTION_GET_DOZE_SUPPORT,
    GetHdrCapabilities = HWC2_FUNCTION_GET_HDR_CAPABILITIES,
    GetMaxVirtualDisplayCount = HWC2_FUNCTION_GET_MAX_VIRTUAL_DISPLAY_COUNT,
    GetReleaseFences = HWC2_FUNCTION_GET_RELEASE_FENCES,
    PresentDisplay = HWC2_FUNCTION_PRESENT_DISPLAY,
    RegisterCallback = HWC2_FUNCTION_REGISTER_CALLBACK,
    SetActiveConfig = HWC2_FUNCTION_SET_ACTIVE_CONFIG,
    SetClientTarget = HWC2_FUNCTION_SET_CLIENT_TARGET,
    SetColorMode = HWC2_FUNCTION_SET_COLOR_MODE,
    SetColorTransform = HWC2_FUNCTION_SET_COLOR_TRANSFORM,
    SetCursorPosition = HWC2_FUNCTION_SET_CURSOR_POSITION,
    SetLayerBlendMode = HWC2_FUNCTION_SET_LAYER_BLEND_MODE,
    SetLayerBuffer = HWC2_FUNCTION_SET_LAYER_BUFFER,
    SetLayerColor = HWC2_FUNCTION_SET_LAYER_COLOR,
    SetLayerCompositionType = HWC2_FUNCTION_SET_LAYER_COMPOSITION_TYPE,
    SetLayerDataspace = HWC2_FUNCTION_SET_LAYER_DATASPACE,
    SetLayerDisplayFrame = HWC2_FUNCTION_SET_LAYER_DISPLAY_FRAME,
    SetLayerPlaneAlpha = HWC2_FUNCTION_SET_LAYER_PLANE_ALPHA,
    SetLayerSidebandStream = HWC2_FUNCTION_SET_LAYER_SIDEBAND_STREAM,
    SetLayerSourceCrop = HWC2_FUNCTION_SET_LAYER_SOURCE_CROP,
    SetLayerSurfaceDamage = HWC2_FUNCTION_SET_LAYER_SURFACE_DAMAGE,
    SetLayerTransform = HWC2_FUNCTION_SET_LAYER_TRANSFORM,
    SetLayerVisibleRegion = HWC2_FUNCTION_SET_LAYER_VISIBLE_REGION,
    SetLayerZOrder = HWC2_FUNCTION_SET_LAYER_Z_ORDER,
    SetOutputBuffer = HWC2_FUNCTION_SET_OUTPUT_BUFFER,
    SetPowerMode = HWC2_FUNCTION_SET_POWER_MODE,
    SetVsyncEnabled = HWC2_FUNCTION_SET_VSYNC_ENABLED,
    ValidateDisplay = HWC2_FUNCTION_VALIDATE_DISPLAY,
    SetLayerFloatColor = HWC2_FUNCTION_SET_LAYER_FLOAT_COLOR,
    SetLayerPerFrameMetadata = HWC2_FUNCTION_SET_LAYER_PER_FRAME_METADATA,
    GetPerFrameMetadataKeys = HWC2_FUNCTION_GET_PER_FRAME_METADATA_KEYS,
    SetReadbackBuffer = HWC2_FUNCTION_SET_READBACK_BUFFER,
    GetReadbackBufferAttributes = HWC2_FUNCTION_GET_READBACK_BUFFER_ATTRIBUTES,
    GetReadbackBufferFence = HWC2_FUNCTION_GET_READBACK_BUFFER_FENCE,
    GetRenderIntents = HWC2_FUNCTION_GET_RENDER_INTENTS,
    SetColorModeWithRenderIntent = HWC2_FUNCTION_SET_COLOR_MODE_WITH_RENDER_INTENT,
    GetDataspaceSaturationMatrix = HWC2_FUNCTION_GET_DATASPACE_SATURATION_MATRIX,
};
TO_STRING(hwc2_function_descriptor_t, FunctionDescriptor,
        getFunctionDescriptorName)

enum class LayerRequest : int32_t {
    ClearClientTarget = HWC2_LAYER_REQUEST_CLEAR_CLIENT_TARGET,
};
TO_STRING(hwc2_layer_request_t, LayerRequest, getLayerRequestName)

enum class PowerMode : int32_t {
    Off = HWC2_POWER_MODE_OFF,
    DozeSuspend = HWC2_POWER_MODE_DOZE_SUSPEND,
    Doze = HWC2_POWER_MODE_DOZE,
    On = HWC2_POWER_MODE_ON,
};
TO_STRING(hwc2_power_mode_t, PowerMode, getPowerModeName)

enum class Transform : int32_t {
    None = 0,
    FlipH = HWC_TRANSFORM_FLIP_H,
    FlipV = HWC_TRANSFORM_FLIP_V,
    Rotate90 = HWC_TRANSFORM_ROT_90,
    Rotate180 = HWC_TRANSFORM_ROT_180,
    Rotate270 = HWC_TRANSFORM_ROT_270,
    FlipHRotate90 = HWC_TRANSFORM_FLIP_H_ROT_90,
    FlipVRotate90 = HWC_TRANSFORM_FLIP_V_ROT_90,
};
TO_STRING(hwc_transform_t, Transform, getTransformName)

enum class Vsync : int32_t {
    Invalid = HWC2_VSYNC_INVALID,
    Enable = HWC2_VSYNC_ENABLE,
    Disable = HWC2_VSYNC_DISABLE,
};
TO_STRING(hwc2_vsync_t, Vsync, getVsyncName)

} // namespace HWC2

__BEGIN_DECLS
#endif // HWC2_USE_CPP11

/*
 * Typedefs
 */

typedef void (*hwc2_function_pointer_t)();

typedef void* hwc2_callback_data_t;
typedef uint32_t hwc2_config_t;
typedef uint64_t hwc2_display_t;
typedef uint64_t hwc2_layer_t;

/*
 * Device Struct
 */

typedef struct hwc2_device {
    /* Must be the first member of this struct, since a pointer to this struct
     * will be generated by casting from a hw_device_t* */
    struct hw_device_t common;

    /* getCapabilities(..., outCount, outCapabilities)
     *
     * Provides a list of capabilities (described in the definition of
     * hwc2_capability_t above) supported by this device. This list must
     * not change after the device has been loaded.
     *
     * Parameters:
     *   outCount - if outCapabilities was NULL, the number of capabilities
     *       which would have been returned; if outCapabilities was not NULL,
     *       the number of capabilities returned, which must not exceed the
     *       value stored in outCount prior to the call
     *   outCapabilities - a list of capabilities supported by this device; may
     *       be NULL, in which case this function must write into outCount the
     *       number of capabilities which would have been written into
     *       outCapabilities
     */
    void (*getCapabilities)(struct hwc2_device* device, uint32_t* outCount,
            int32_t* /*hwc2_capability_t*/ outCapabilities);

    /* getFunction(..., descriptor)
     *
     * Returns a function pointer which implements the requested description.
     *
     * Parameters:
     *   descriptor - the function to return
     *
     * Returns either a function pointer implementing the requested descriptor
     *   or NULL if the described function is not supported by this device.
     */
    hwc2_function_pointer_t (*getFunction)(struct hwc2_device* device,
            int32_t /*hwc2_function_descriptor_t*/ descriptor);
} hwc2_device_t;

static inline int hwc2_open(const struct hw_module_t* module,
        hwc2_device_t** device) {
    return module->methods->open(module, HWC_HARDWARE_COMPOSER,
            TO_HW_DEVICE_T_OPEN(device));
}

static inline int hwc2_close(hwc2_device_t* device) {
    return device->common.close(&device->common);
}

/*
 * Callbacks
 *
 * All of these callbacks take as their first parameter the callbackData which
 * was provided at the time of callback registration, so this parameter is
 * omitted from the described parameter lists.
 */

/* hotplug(..., display, connected)
 * Descriptor: HWC2_CALLBACK_HOTPLUG
 * Will be provided to all HWC2 devices
 *
 * Notifies the client that the given display has either been connected or
 * disconnected. Every active display (even a built-in physical display) must
 * trigger at least one hotplug notification, even if it only occurs immediately
 * after callback registration.
 *
 * The client may call back into the device on the same thread to query display
 * properties (such as width, height, and vsync period), and other threads may
 * call into the device while the callback is in progress. The device must
 * serialize calls to this callback such that only one thread is calling it at a
 * time.
 *
 * Displays which have been connected are assumed to be in HWC2_POWER_MODE_OFF,
 * and the vsync callback should not be called for a display until vsync has
 * been enabled with setVsyncEnabled.
 *
 * Parameters:
 *   display - the display which has been hotplugged
 *   connected - whether the display has been connected or disconnected
 */
typedef void (*HWC2_PFN_HOTPLUG)(hwc2_callback_data_t callbackData,
        hwc2_display_t display, int32_t /*hwc2_connection_t*/ connected);

/* refresh(..., display)
 * Descriptor: HWC2_CALLBACK_REFRESH
 * Will be provided to all HWC2 devices
 *
 * Notifies the client to trigger a screen refresh. This forces all layer state
 * for this display to be resent, and the display to be validated and presented,
 * even if there have been no changes.
 *
 * This refresh will occur some time after the callback is initiated, but not
 * necessarily before it returns. This thread, however, is guaranteed not to
 * call back into the device, thus it is safe to trigger this callback from
 * other functions which call into the device.
 *
 * Parameters:
 *   display - the display to refresh
 */
typedef void (*HWC2_PFN_REFRESH)(hwc2_callback_data_t callbackData,
        hwc2_display_t display);

/* vsync(..., display, timestamp)
 * Descriptor: HWC2_CALLBACK_VSYNC
 * Will be provided to all HWC2 devices
 *
 * Notifies the client that a vsync event has occurred. This callback must
 * only be triggered when vsync is enabled for this display (through
 * setVsyncEnabled).
 *
 * This callback should be triggered from a thread of at least
 * HAL_PRIORITY_URGENT_DISPLAY with as little latency as possible, typically
 * less than 0.5 ms. This thread is guaranteed not to call back into the device.
 *
 * Parameters:
 *   display - the display which has received a vsync event
 *   timestamp - the CLOCK_MONOTONIC time at which the vsync event occurred, in
 *       nanoseconds
 */
typedef void (*HWC2_PFN_VSYNC)(hwc2_callback_data_t callbackData,
        hwc2_display_t display, int64_t timestamp);

/*
 * Device Functions
 *
 * All of these functions take as their first parameter a device pointer, so
 * this parameter is omitted from the described parameter lists.
 */

/* createVirtualDisplay(..., width, height, format, outDisplay)
 * Descriptor: HWC2_FUNCTION_CREATE_VIRTUAL_DISPLAY
 * Must be provided by all HWC2 devices
 *
 * Creates a new virtual display with the given width and height. The format
 * passed into this function is the default format requested by the consumer of
 * the virtual display output buffers. If a different format will be returned by
 * the device, it should be returned in this parameter so it can be set properly
 * when handing the buffers to the consumer.
 *
 * The display will be assumed to be on from the time the first frame is
 * presented until the display is destroyed.
 *
 * Parameters:
 *   width - width in pixels
 *   height - height in pixels
 *   format - prior to the call, the default output buffer format selected by
 *       the consumer; after the call, the format the device will produce
 *   outDisplay - the newly-created virtual display; pointer will be non-NULL
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_UNSUPPORTED - the width or height is too large for the device to
 *       be able to create a virtual display
 *   HWC2_ERROR_NO_RESOURCES - the device is unable to create a new virtual
 *       display at this time
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_CREATE_VIRTUAL_DISPLAY)(
        hwc2_device_t* device, uint32_t width, uint32_t height,
        int32_t* /*android_pixel_format_t*/ format, hwc2_display_t* outDisplay);

/* destroyVirtualDisplay(..., display)
 * Descriptor: HWC2_FUNCTION_DESTROY_VIRTUAL_DISPLAY
 * Must be provided by all HWC2 devices
 *
 * Destroys a virtual display. After this call all resources consumed by this
 * display may be freed by the device and any operations performed on this
 * display should fail.
 *
 * Parameters:
 *   display - the virtual display to destroy
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - the display handle which was passed in does not
 *       refer to a virtual display
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_DESTROY_VIRTUAL_DISPLAY)(
        hwc2_device_t* device, hwc2_display_t display);

/* dump(..., outSize, outBuffer)
 * Descriptor: HWC2_FUNCTION_DUMP
 * Must be provided by all HWC2 devices
 *
 * Retrieves implementation-defined debug information, which will be displayed
 * during, for example, `dumpsys SurfaceFlinger`.
 *
 * If called with outBuffer == NULL, the device should store a copy of the
 * desired output and return its length in bytes in outSize. If the device
 * already has a stored copy, that copy should be purged and replaced with a
 * fresh copy.
 *
 * If called with outBuffer != NULL, the device should copy its stored version
 * of the output into outBuffer and store how many bytes of data it copied into
 * outSize. Prior to this call, the client will have populated outSize with the
 * maximum number of bytes outBuffer can hold. The device must not write more
 * than this amount into outBuffer. If the device does not currently have a
 * stored copy, then it should return 0 in outSize.
 *
 * Any data written into outBuffer need not be null-terminated.
 *
 * Parameters:
 *   outSize - if outBuffer was NULL, the number of bytes needed to copy the
 *       device's stored output; if outBuffer was not NULL, the number of bytes
 *       written into it, which must not exceed the value stored in outSize
 *       prior to the call; pointer will be non-NULL
 *   outBuffer - the buffer to write the dump output into; may be NULL as
 *       described above; data written into this buffer need not be
 *       null-terminated
 */
typedef void (*HWC2_PFN_DUMP)(hwc2_device_t* device, uint32_t* outSize,
        char* outBuffer);

/* getMaxVirtualDisplayCount(...)
 * Descriptor: HWC2_FUNCTION_GET_MAX_VIRTUAL_DISPLAY_COUNT
 * Must be provided by all HWC2 devices
 *
 * Returns the maximum number of virtual displays supported by this device
 * (which may be 0). The client will not attempt to create more than this many
 * virtual displays on this device. This number must not change for the lifetime
 * of the device.
 */
typedef uint32_t (*HWC2_PFN_GET_MAX_VIRTUAL_DISPLAY_COUNT)(
        hwc2_device_t* device);

/* registerCallback(..., descriptor, callbackData, pointer)
 * Descriptor: HWC2_FUNCTION_REGISTER_CALLBACK
 * Must be provided by all HWC2 devices
 *
 * Provides a callback for the device to call. All callbacks take a callbackData
 * item as the first parameter, so this value should be stored with the callback
 * for later use. The callbackData may differ from one callback to another. If
 * this function is called multiple times with the same descriptor, later
 * callbacks replace earlier ones.
 *
 * Parameters:
 *   descriptor - which callback should be set
 *   callBackdata - opaque data which must be passed back through the callback
 *   pointer - a non-NULL function pointer corresponding to the descriptor
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_PARAMETER - descriptor was invalid
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_REGISTER_CALLBACK)(
        hwc2_device_t* device,
        int32_t /*hwc2_callback_descriptor_t*/ descriptor,
        hwc2_callback_data_t callbackData, hwc2_function_pointer_t pointer);

/* getDataspaceSaturationMatrix(..., dataspace, outMatrix)
 * Descriptor: HWC2_FUNCTION_GET_DATASPACE_SATURATION_MATRIX
 * Provided by HWC2 devices which don't return nullptr function pointer.
 *
 * Get the saturation matrix of the specified dataspace. The saturation matrix
 * can be used to approximate the dataspace saturation operation performed by
 * the HWC2 device when non-colorimetric mapping is allowed. It is to be
 * applied on linear pixel values.
 *
 * Parameters:
 *   dataspace - the dataspace to query for
 *   outMatrix - a column-major 4x4 matrix (16 floats). It must be an identity
 *       matrix unless dataspace is HAL_DATASPACE_SRGB_LINEAR.
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_PARAMETER - dataspace was invalid
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_DATASPACE_SATURATION_MATRIX)(
        hwc2_device_t* device, int32_t /*android_dataspace_t*/ dataspace,
        float* outMatrix);

/*
 * Display Functions
 *
 * All of these functions take as their first two parameters a device pointer
 * and a display handle, so these parameters are omitted from the described
 * parameter lists.
 */

/* acceptDisplayChanges(...)
 * Descriptor: HWC2_FUNCTION_ACCEPT_DISPLAY_CHANGES
 * Must be provided by all HWC2 devices
 *
 * Accepts the changes required by the device from the previous validateDisplay
 * call (which may be queried using getChangedCompositionTypes) and revalidates
 * the display. This function is equivalent to requesting the changed types from
 * getChangedCompositionTypes, setting those types on the corresponding layers,
 * and then calling validateDisplay again.
 *
 * After this call it must be valid to present this display. Calling this after
 * validateDisplay returns 0 changes must succeed with HWC2_ERROR_NONE, but
 * should have no other effect.
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_NOT_VALIDATED - validateDisplay has not been called
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_ACCEPT_DISPLAY_CHANGES)(
        hwc2_device_t* device, hwc2_display_t display);

/* createLayer(..., outLayer)
 * Descriptor: HWC2_FUNCTION_CREATE_LAYER
 * Must be provided by all HWC2 devices
 *
 * Creates a new layer on the given display.
 *
 * Parameters:
 *   outLayer - the handle of the new layer; pointer will be non-NULL
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_NO_RESOURCES - the device was unable to create this layer
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_CREATE_LAYER)(hwc2_device_t* device,
        hwc2_display_t display, hwc2_layer_t* outLayer);

/* destroyLayer(..., layer)
 * Descriptor: HWC2_FUNCTION_DESTROY_LAYER
 * Must be provided by all HWC2 devices
 *
 * Destroys the given layer.
 *
 * Parameters:
 *   layer - the handle of the layer to destroy
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_DESTROY_LAYER)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer);

/* getActiveConfig(..., outConfig)
 * Descriptor: HWC2_FUNCTION_GET_ACTIVE_CONFIG
 * Must be provided by all HWC2 devices
 *
 * Retrieves which display configuration is currently active.
 *
 * If no display configuration is currently active, this function must return
 * HWC2_ERROR_BAD_CONFIG and place no configuration handle in outConfig. It is
 * the responsibility of the client to call setActiveConfig with a valid
 * configuration before attempting to present anything on the display.
 *
 * Parameters:
 *   outConfig - the currently active display configuration; pointer will be
 *       non-NULL
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_CONFIG - no configuration is currently active
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_ACTIVE_CONFIG)(
        hwc2_device_t* device, hwc2_display_t display,
        hwc2_config_t* outConfig);

/* getChangedCompositionTypes(..., outNumElements, outLayers, outTypes)
 * Descriptor: HWC2_FUNCTION_GET_CHANGED_COMPOSITION_TYPES
 * Must be provided by all HWC2 devices
 *
 * Retrieves the layers for which the device requires a different composition
 * type than had been set prior to the last call to validateDisplay. The client
 * will either update its state with these types and call acceptDisplayChanges,
 * or will set new types and attempt to validate the display again.
 *
 * outLayers and outTypes may be NULL to retrieve the number of elements which
 * will be returned. The number of elements returned must be the same as the
 * value returned in outNumTypes from the last call to validateDisplay.
 *
 * Parameters:
 *   outNumElements - if outLayers or outTypes were NULL, the number of layers
 *       and types which would have been returned; if both were non-NULL, the
 *       number of elements returned in outLayers and outTypes, which must not
 *       exceed the value stored in outNumElements prior to the call; pointer
 *       will be non-NULL
 *   outLayers - an array of layer handles
 *   outTypes - an array of composition types, each corresponding to an element
 *       of outLayers
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_NOT_VALIDATED - validateDisplay has not been called for this
 *       display
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_CHANGED_COMPOSITION_TYPES)(
        hwc2_device_t* device, hwc2_display_t display,
        uint32_t* outNumElements, hwc2_layer_t* outLayers,
        int32_t* /*hwc2_composition_t*/ outTypes);

/* getClientTargetSupport(..., width, height, format, dataspace)
 * Descriptor: HWC2_FUNCTION_GET_CLIENT_TARGET_SUPPORT
 * Must be provided by all HWC2 devices
 *
 * Returns whether a client target with the given properties can be handled by
 * the device.
 *
 * The valid formats can be found in android_pixel_format_t in
 * <system/graphics.h>.
 *
 * For more about dataspaces, see setLayerDataspace.
 *
 * This function must return true for a client target with width and height
 * equal to the active display configuration dimensions,
 * HAL_PIXEL_FORMAT_RGBA_8888, and HAL_DATASPACE_UNKNOWN. It is not required to
 * return true for any other configuration.
 *
 * Parameters:
 *   width - client target width in pixels
 *   height - client target height in pixels
 *   format - client target format
 *   dataspace - client target dataspace, as described in setLayerDataspace
 *
 * Returns HWC2_ERROR_NONE if the given configuration is supported or one of the
 * following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_UNSUPPORTED - the given configuration is not supported
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_CLIENT_TARGET_SUPPORT)(
        hwc2_device_t* device, hwc2_display_t display, uint32_t width,
        uint32_t height, int32_t /*android_pixel_format_t*/ format,
        int32_t /*android_dataspace_t*/ dataspace);

/* getColorModes(..., outNumModes, outModes)
 * Descriptor: HWC2_FUNCTION_GET_COLOR_MODES
 * Must be provided by all HWC2 devices
 *
 * Returns the color modes supported on this display.
 *
 * The valid color modes can be found in android_color_mode_t in
 * <system/graphics.h>. All HWC2 devices must support at least
 * HAL_COLOR_MODE_NATIVE.
 *
 * outNumModes may be NULL to retrieve the number of modes which will be
 * returned.
 *
 * Parameters:
 *   outNumModes - if outModes was NULL, the number of modes which would have
 *       been returned; if outModes was not NULL, the number of modes returned,
 *       which must not exceed the value stored in outNumModes prior to the
 *       call; pointer will be non-NULL
 *   outModes - an array of color modes
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_COLOR_MODES)(
        hwc2_device_t* device, hwc2_display_t display, uint32_t* outNumModes,
        int32_t* /*android_color_mode_t*/ outModes);

/* getRenderIntents(..., mode, outNumIntents, outIntents)
 * Descriptor: HWC2_FUNCTION_GET_RENDER_INTENTS
 * Provided by HWC2 devices which don't return nullptr function pointer.
 *
 * Returns the render intents supported on this display.
 *
 * The valid render intents can be found in android_render_intent_v1_1_t in
 * <system/graphics.h>. All HWC2 devices must support at least
 * HAL_RENDER_INTENT_COLORIMETRIC.
 *
 * outNumIntents may be NULL to retrieve the number of intents which will be
 * returned.
 *
 * Parameters:
 *   mode - the color mode to query the render intents for
 *   outNumIntents - if outIntents was NULL, the number of intents which would
 *       have been returned; if outIntents was not NULL, the number of intents
 *       returned, which must not exceed the value stored in outNumIntents
 *       prior to the call; pointer will be non-NULL
 *   outIntents - an array of render intents
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_RENDER_INTENTS)(
        hwc2_device_t* device, hwc2_display_t display, int32_t mode,
        uint32_t* outNumIntents,
        int32_t* /*android_render_intent_v1_1_t*/ outIntents);

/* getDisplayAttribute(..., config, attribute, outValue)
 * Descriptor: HWC2_FUNCTION_GET_DISPLAY_ATTRIBUTE
 * Must be provided by all HWC2 devices
 *
 * Returns a display attribute value for a particular display configuration.
 *
 * Any attribute which is not supported or for which the value is unknown by the
 * device must return a value of -1.
 *
 * Parameters:
 *   config - the display configuration for which to return attribute values
 *   attribute - the attribute to query
 *   outValue - the value of the attribute; the pointer will be non-NULL
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_CONFIG - config does not name a valid configuration for this
 *       display
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_DISPLAY_ATTRIBUTE)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_config_t config,
        int32_t /*hwc2_attribute_t*/ attribute, int32_t* outValue);

/* getDisplayConfigs(..., outNumConfigs, outConfigs)
 * Descriptor: HWC2_FUNCTION_GET_DISPLAY_CONFIGS
 * Must be provided by all HWC2 devices
 *
 * Returns handles for all of the valid display configurations on this display.
 *
 * outConfigs may be NULL to retrieve the number of elements which will be
 * returned.
 *
 * Parameters:
 *   outNumConfigs - if outConfigs was NULL, the number of configurations which
 *       would have been returned; if outConfigs was not NULL, the number of
 *       configurations returned, which must not exceed the value stored in
 *       outNumConfigs prior to the call; pointer will be non-NULL
 *   outConfigs - an array of configuration handles
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_DISPLAY_CONFIGS)(
        hwc2_device_t* device, hwc2_display_t display, uint32_t* outNumConfigs,
        hwc2_config_t* outConfigs);

/* getDisplayName(..., outSize, outName)
 * Descriptor: HWC2_FUNCTION_GET_DISPLAY_NAME
 * Must be provided by all HWC2 devices
 *
 * Returns a human-readable version of the display's name.
 *
 * outName may be NULL to retrieve the length of the name.
 *
 * Parameters:
 *   outSize - if outName was NULL, the number of bytes needed to return the
 *       name if outName was not NULL, the number of bytes written into it,
 *       which must not exceed the value stored in outSize prior to the call;
 *       pointer will be non-NULL
 *   outName - the display's name
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_DISPLAY_NAME)(
        hwc2_device_t* device, hwc2_display_t display, uint32_t* outSize,
        char* outName);

/* getDisplayRequests(..., outDisplayRequests, outNumElements, outLayers,
 *     outLayerRequests)
 * Descriptor: HWC2_FUNCTION_GET_DISPLAY_REQUESTS
 * Must be provided by all HWC2 devices
 *
 * Returns the display requests and the layer requests required for the last
 * validated configuration.
 *
 * Display requests provide information about how the client should handle the
 * client target. Layer requests provide information about how the client
 * should handle an individual layer.
 *
 * If outLayers or outLayerRequests is NULL, the required number of layers and
 * requests must be returned in outNumElements, but this number may also be
 * obtained from validateDisplay as outNumRequests (outNumElements must be equal
 * to the value returned in outNumRequests from the last call to
 * validateDisplay).
 *
 * Parameters:
 *   outDisplayRequests - the display requests for the current validated state
 *   outNumElements - if outLayers or outLayerRequests were NULL, the number of
 *       elements which would have been returned, which must be equal to the
 *       value returned in outNumRequests from the last validateDisplay call on
 *       this display; if both were not NULL, the number of elements in
 *       outLayers and outLayerRequests, which must not exceed the value stored
 *       in outNumElements prior to the call; pointer will be non-NULL
 *   outLayers - an array of layers which all have at least one request
 *   outLayerRequests - the requests corresponding to each element of outLayers
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_NOT_VALIDATED - validateDisplay has not been called for this
 *       display
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_DISPLAY_REQUESTS)(
        hwc2_device_t* device, hwc2_display_t display,
        int32_t* /*hwc2_display_request_t*/ outDisplayRequests,
        uint32_t* outNumElements, hwc2_layer_t* outLayers,
        int32_t* /*hwc2_layer_request_t*/ outLayerRequests);

/* getDisplayType(..., outType)
 * Descriptor: HWC2_FUNCTION_GET_DISPLAY_TYPE
 * Must be provided by all HWC2 devices
 *
 * Returns whether the given display is a physical or virtual display.
 *
 * Parameters:
 *   outType - the type of the display; pointer will be non-NULL
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_DISPLAY_TYPE)(
        hwc2_device_t* device, hwc2_display_t display,
        int32_t* /*hwc2_display_type_t*/ outType);

/* getDozeSupport(..., outSupport)
 * Descriptor: HWC2_FUNCTION_GET_DOZE_SUPPORT
 * Must be provided by all HWC2 devices
 *
 * Returns whether the given display supports HWC2_POWER_MODE_DOZE and
 * HWC2_POWER_MODE_DOZE_SUSPEND. DOZE_SUSPEND may not provide any benefit over
 * DOZE (see the definition of hwc2_power_mode_t for more information), but if
 * both DOZE and DOZE_SUSPEND are no different from HWC2_POWER_MODE_ON, the
 * device should not claim support.
 *
 * Parameters:
 *   outSupport - whether the display supports doze modes (1 for yes, 0 for no);
 *       pointer will be non-NULL
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_DOZE_SUPPORT)(
        hwc2_device_t* device, hwc2_display_t display, int32_t* outSupport);

/* getHdrCapabilities(..., outNumTypes, outTypes, outMaxLuminance,
 *     outMaxAverageLuminance, outMinLuminance)
 * Descriptor: HWC2_FUNCTION_GET_HDR_CAPABILITIES
 * Must be provided by all HWC2 devices
 *
 * Returns the high dynamic range (HDR) capabilities of the given display, which
 * are invariant with regard to the active configuration.
 *
 * Displays which are not HDR-capable must return no types in outTypes and set
 * outNumTypes to 0.
 *
 * If outTypes is NULL, the required number of HDR types must be returned in
 * outNumTypes.
 *
 * Parameters:
 *   outNumTypes - if outTypes was NULL, the number of types which would have
 *       been returned; if it was not NULL, the number of types stored in
 *       outTypes, which must not exceed the value stored in outNumTypes prior
 *       to the call; pointer will be non-NULL
 *   outTypes - an array of HDR types, may have 0 elements if the display is not
 *       HDR-capable
 *   outMaxLuminance - the desired content maximum luminance for this display in
 *       cd/m^2; pointer will be non-NULL
 *   outMaxAverageLuminance - the desired content maximum frame-average
 *       luminance for this display in cd/m^2; pointer will be non-NULL
 *   outMinLuminance - the desired content minimum luminance for this display in
 *       cd/m^2; pointer will be non-NULL
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_HDR_CAPABILITIES)(
        hwc2_device_t* device, hwc2_display_t display, uint32_t* outNumTypes,
        int32_t* /*android_hdr_t*/ outTypes, float* outMaxLuminance,
        float* outMaxAverageLuminance, float* outMinLuminance);

/* getReleaseFences(..., outNumElements, outLayers, outFences)
 * Descriptor: HWC2_FUNCTION_GET_RELEASE_FENCES
 * Must be provided by all HWC2 devices
 *
 * Retrieves the release fences for device layers on this display which will
 * receive new buffer contents this frame.
 *
 * A release fence is a file descriptor referring to a sync fence object which
 * will be signaled after the device has finished reading from the buffer
 * presented in the prior frame. This indicates that it is safe to start writing
 * to the buffer again. If a given layer's fence is not returned from this
 * function, it will be assumed that the buffer presented on the previous frame
 * is ready to be written.
 *
 * The fences returned by this function should be unique for each layer (even if
 * they point to the same underlying sync object), and ownership of the fences
 * is transferred to the client, which is responsible for closing them.
 *
 * If outLayers or outFences is NULL, the required number of layers and fences
 * must be returned in outNumElements.
 *
 * Parameters:
 *   outNumElements - if outLayers or outFences were NULL, the number of
 *       elements which would have been returned; if both were not NULL, the
 *       number of elements in outLayers and outFences, which must not exceed
 *       the value stored in outNumElements prior to the call; pointer will be
 *       non-NULL
 *   outLayers - an array of layer handles
 *   outFences - an array of sync fence file descriptors as described above,
 *       each corresponding to an element of outLayers
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_RELEASE_FENCES)(
        hwc2_device_t* device, hwc2_display_t display, uint32_t* outNumElements,
        hwc2_layer_t* outLayers, int32_t* outFences);

/* presentDisplay(..., outPresentFence)
 * Descriptor: HWC2_FUNCTION_PRESENT_DISPLAY
 * Must be provided by all HWC2 devices
 *
 * Presents the current display contents on the screen (or in the case of
 * virtual displays, into the output buffer).
 *
 * Prior to calling this function, the display must be successfully validated
 * with validateDisplay. Note that setLayerBuffer and setLayerSurfaceDamage
 * specifically do not count as layer state, so if there are no other changes
 * to the layer state (or to the buffer's properties as described in
 * setLayerBuffer), then it is safe to call this function without first
 * validating the display.
 *
 * If this call succeeds, outPresentFence will be populated with a file
 * descriptor referring to a present sync fence object. For physical displays,
 * this fence will be signaled at the vsync when the result of composition of
 * this frame starts to appear (for video-mode panels) or starts to transfer to
 * panel memory (for command-mode panels). For virtual displays, this fence will
 * be signaled when writes to the output buffer have completed and it is safe to
 * read from it.
 *
 * Parameters:
 *   outPresentFence - a sync fence file descriptor as described above; pointer
 *       will be non-NULL
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_NO_RESOURCES - no valid output buffer has been set for a virtual
 *       display
 *   HWC2_ERROR_NOT_VALIDATED - validateDisplay has not successfully been called
 *       for this display
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_PRESENT_DISPLAY)(
        hwc2_device_t* device, hwc2_display_t display,
        int32_t* outPresentFence);

/* setActiveConfig(..., config)
 * Descriptor: HWC2_FUNCTION_SET_ACTIVE_CONFIG
 * Must be provided by all HWC2 devices
 *
 * Sets the active configuration for this display. Upon returning, the given
 * display configuration should be active and remain so until either this
 * function is called again or the display is disconnected.
 *
 * Parameters:
 *   config - the new display configuration
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_CONFIG - the configuration handle passed in is not valid for
 *       this display
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_ACTIVE_CONFIG)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_config_t config);

/* setClientTarget(..., target, acquireFence, dataspace, damage)
 * Descriptor: HWC2_FUNCTION_SET_CLIENT_TARGET
 * Must be provided by all HWC2 devices
 *
 * Sets the buffer handle which will receive the output of client composition.
 * Layers marked as HWC2_COMPOSITION_CLIENT will be composited into this buffer
 * prior to the call to presentDisplay, and layers not marked as
 * HWC2_COMPOSITION_CLIENT should be composited with this buffer by the device.
 *
 * The buffer handle provided may be null if no layers are being composited by
 * the client. This must not result in an error (unless an invalid display
 * handle is also provided).
 *
 * Also provides a file descriptor referring to an acquire sync fence object,
 * which will be signaled when it is safe to read from the client target buffer.
 * If it is already safe to read from this buffer, -1 may be passed instead.
 * The device must ensure that it is safe for the client to close this file
 * descriptor at any point after this function is called.
 *
 * For more about dataspaces, see setLayerDataspace.
 *
 * The damage parameter describes a surface damage region as defined in the
 * description of setLayerSurfaceDamage.
 *
 * Will be called before presentDisplay if any of the layers are marked as
 * HWC2_COMPOSITION_CLIENT. If no layers are so marked, then it is not
 * necessary to call this function. It is not necessary to call validateDisplay
 * after changing the target through this function.
 *
 * Parameters:
 *   target - the new target buffer
 *   acquireFence - a sync fence file descriptor as described above
 *   dataspace - the dataspace of the buffer, as described in setLayerDataspace
 *   damage - the surface damage region
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - the new target handle was invalid
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_CLIENT_TARGET)(
        hwc2_device_t* device, hwc2_display_t display, buffer_handle_t target,
        int32_t acquireFence, int32_t /*android_dataspace_t*/ dataspace,
        hwc_region_t damage);

/* setColorMode(..., mode)
 * Descriptor: HWC2_FUNCTION_SET_COLOR_MODE
 * Must be provided by all HWC2 devices
 *
 * Sets the color mode of the given display.
 *
 * This must be called outside of validateDisplay/presentDisplay, and it takes
 * effect on next presentDisplay.
 *
 * The valid color modes can be found in android_color_mode_t in
 * <system/graphics.h>. All HWC2 devices must support at least
 * HAL_COLOR_MODE_NATIVE, and displays are assumed to be in this mode upon
 * hotplug.
 *
 * Parameters:
 *   mode - the mode to set
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - mode is not a valid color mode
 *   HWC2_ERROR_UNSUPPORTED - mode is not supported on this display
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_COLOR_MODE)(
        hwc2_device_t* device, hwc2_display_t display,
        int32_t /*android_color_mode_t*/ mode);

/* setColorModeWithIntent(..., mode, intent)
 * Descriptor: HWC2_FUNCTION_SET_COLOR_MODE_WITH_RENDER_INTENT
 * Provided by HWC2 devices which don't return nullptr function pointer.
 *
 * This must be called outside of validateDisplay/presentDisplay, and it takes
 * effect on next presentDisplay.
 *
 * The valid color modes and render intents can be found in
 * android_color_mode_t and android_render_intent_v1_1_t in
 * <system/graphics.h>. All HWC2 devices must support at least
 * HAL_COLOR_MODE_NATIVE and HAL_RENDER_INTENT_COLORIMETRIC, and displays are
 * assumed to be in this mode and intent upon hotplug.
 *
 * Parameters:
 *   mode - the mode to set
 *   intent - the intent to set
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - mode/intent is not a valid color mode or
 *       render intent
 *   HWC2_ERROR_UNSUPPORTED - mode or intent is not supported on this display
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_COLOR_MODE_WITH_RENDER_INTENT)(
        hwc2_device_t* device, hwc2_display_t display,
        int32_t /*android_color_mode_t*/ mode,
        int32_t /*android_render_intent_v1_1_t */ intent);

/* setColorTransform(..., matrix, hint)
 * Descriptor: HWC2_FUNCTION_SET_COLOR_TRANSFORM
 * Must be provided by all HWC2 devices
 *
 * Sets a color transform which will be applied after composition.
 *
 * If hint is not HAL_COLOR_TRANSFORM_ARBITRARY, then the device may use the
 * hint to apply the desired color transform instead of using the color matrix
 * directly.
 *
 * If the device is not capable of either using the hint or the matrix to apply
 * the desired color transform, it should force all layers to client composition
 * during validateDisplay.
 *
 * If HWC2_CAPABILITY_SKIP_CLIENT_COLOR_TRANSFORM is present, then the client
 * will never apply the color transform during client composition, even if all
 * layers are being composed by the client.
 *
 * The matrix provided is an affine color transformation of the following form:
 *
 * |r.r r.g r.b 0|
 * |g.r g.g g.b 0|
 * |b.r b.g b.b 0|
 * |Tr  Tg  Tb  1|
 *
 * This matrix will be provided in row-major form: {r.r, r.g, r.b, 0, g.r, ...}.
 *
 * Given a matrix of this form and an input color [R_in, G_in, B_in], the output
 * color [R_out, G_out, B_out] will be:
 *
 * R_out = R_in * r.r + G_in * g.r + B_in * b.r + Tr
 * G_out = R_in * r.g + G_in * g.g + B_in * b.g + Tg
 * B_out = R_in * r.b + G_in * g.b + B_in * b.b + Tb
 *
 * Parameters:
 *   matrix - a 4x4 transform matrix (16 floats) as described above
 *   hint - a hint value which may be used instead of the given matrix unless it
 *       is HAL_COLOR_TRANSFORM_ARBITRARY
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - hint is not a valid color transform hint
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_COLOR_TRANSFORM)(
        hwc2_device_t* device, hwc2_display_t display, const float* matrix,
        int32_t /*android_color_transform_t*/ hint);

/* getPerFrameMetadataKeys(..., outKeys)
 * Descriptor: HWC2_FUNCTION_GET_PER_FRAME_METADATA_KEYS
 * Optional for HWC2 devices
 *
 * If supported (getFunction(HWC2_FUNCTION_GET_PER_FRAME_METADATA_KEYS) is non-null),
 * getPerFrameMetadataKeys returns the list of supported PerFrameMetadataKeys
 * which are invariant with regard to the active configuration.
 *
 * Devices which are not HDR-capable, must return null when getFunction is called
 * with HWC2_FUNCTION_GET_PER_FRAME_METADATA_KEYS.
 *
 * If outKeys is NULL, the required number of PerFrameMetadataKey keys
 * must be returned in outNumKeys.
 *
 * Parameters:
 *   outNumKeys - if outKeys is NULL, the number of keys which would have
 *       been returned; if outKeys is not NULL, the number of keys stored in
 *       outKeys, which must not exceed the value stored in outNumKeys prior
 *       to the call; pointer will be non-NULL
 *   outKeys - an array of hwc2_per_frame_metadata_key_t keys
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_PER_FRAME_METADATA_KEYS)(
        hwc2_device_t* device, hwc2_display_t display, uint32_t* outNumKeys,
        int32_t* /*hwc2_per_frame_metadata_key_t*/ outKeys);

/* setOutputBuffer(..., buffer, releaseFence)
 * Descriptor: HWC2_FUNCTION_SET_OUTPUT_BUFFER
 * Must be provided by all HWC2 devices
 *
 * Sets the output buffer for a virtual display. That is, the buffer to which
 * the composition result will be written.
 *
 * Also provides a file descriptor referring to a release sync fence object,
 * which will be signaled when it is safe to write to the output buffer. If it
 * is already safe to write to the output buffer, -1 may be passed instead. The
 * device must ensure that it is safe for the client to close this file
 * descriptor at any point after this function is called.
 *
 * Must be called at least once before presentDisplay, but does not have any
 * interaction with layer state or display validation.
 *
 * Parameters:
 *   buffer - the new output buffer
 *   releaseFence - a sync fence file descriptor as described above
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - the new output buffer handle was invalid
 *   HWC2_ERROR_UNSUPPORTED - display does not refer to a virtual display
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_OUTPUT_BUFFER)(
        hwc2_device_t* device, hwc2_display_t display, buffer_handle_t buffer,
        int32_t releaseFence);

/* setPowerMode(..., mode)
 * Descriptor: HWC2_FUNCTION_SET_POWER_MODE
 * Must be provided by all HWC2 devices
 *
 * Sets the power mode of the given display. The transition must be complete
 * when this function returns. It is valid to call this function multiple times
 * with the same power mode.
 *
 * All displays must support HWC2_POWER_MODE_ON and HWC2_POWER_MODE_OFF. Whether
 * a display supports HWC2_POWER_MODE_DOZE or HWC2_POWER_MODE_DOZE_SUSPEND may
 * be queried using getDozeSupport.
 *
 * Parameters:
 *   mode - the new power mode
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - mode was not a valid power mode
 *   HWC2_ERROR_UNSUPPORTED - mode was a valid power mode, but is not supported
 *       on this display
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_POWER_MODE)(
        hwc2_device_t* device, hwc2_display_t display,
        int32_t /*hwc2_power_mode_t*/ mode);

/* getReadbackBufferAttributes(..., outFormat, outDataspace)
 * Optional for HWC2 devices
 *
 * Returns the format which should be used when allocating a buffer for use by
 * device readback as well as the dataspace in which its contents should be
 * interpreted.
 *
 * If readback is not supported by this HWC implementation, this call will also
 * be able to return HWC2_ERROR_UNSUPPORTED so we can fall back to another method.
 * Returning NULL to a getFunction request for this function will also indicate
 * that readback is not supported.
 *
 * The width and height of this buffer will be those of the currently-active
 * display configuration, and the usage flags will consist of the following:
 *   BufferUsage::CPU_READ | BufferUsage::GPU_TEXTURE |
 *   BufferUsage::COMPOSER_OUTPUT
 *
 * The format and dataspace provided must be sufficient such that if a
 * correctly-configured buffer is passed into setReadbackBuffer, filled by
 * the device, and then displayed by the client as a full-screen buffer, the
 * output of the display remains the same (subject to the note about protected
 * content in the description of setReadbackBuffer).
 *
 * If the active configuration or color mode of this display has changed since
 * the previous call to this function, it will be called again prior to setting
 * a readback buffer such that the returned format and dataspace can be updated
 * accordingly.
 *
 * Parameters:
 *   outFormat - the format the client should use when allocating a device
 *       readback buffer; pointer will be non-NULL
 *   outDataspace - the dataspace the client will use when interpreting the
 *       contents of a device readback buffer; pointer will be non-NULL
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *
 * See also:
 *   setReadbackBuffer
 *   getReadbackBufferFence
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_READBACK_BUFFER_ATTRIBUTES)(
        hwc2_device_t* device, hwc2_display_t display,
        int32_t* /*android_pixel_format_t*/ outFormat,
        int32_t* /*android_dataspace_t*/ outDataspace);

/* getReadbackBufferFence(..., outFence)
 * Optional for HWC2 devices
 *
 * Returns an acquire sync fence file descriptor which will signal when the
 * buffer provided to setReadbackBuffer has been filled by the device and is
 * safe for the client to read.
 *
 * If it is already safe to read from this buffer, -1 may be returned instead.
 * The client takes ownership of this file descriptor and is responsible for
 * closing it when it is no longer needed.
 *
 * This function will be called immediately after the composition cycle being
 * captured into the readback buffer. The complete ordering of a readback buffer
 * capture is as follows:
 *
 *   getReadbackBufferAttributes
 *   // Readback buffer is allocated
 *   // Many frames may pass
 *
 *   setReadbackBuffer
 *   validateDisplay
 *   presentDisplay
 *   getReadbackBufferFence
 *   // Implicitly wait on the acquire fence before accessing the buffer
 *
 * Parameters:
 *   outFence - a sync fence file descriptor as described above; pointer
 *       will be non-NULL
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_NO_RESOURCES - the readback operation was successful, but
 *       resulted in a different validate result than would have occurred
 *       without readback
 *   HWC2_ERROR_UNSUPPORTED - the readback operation was unsuccessful because
 *       of resource constraints, the presence of protected content, or other
 *       reasons; -1 must be returned in outFence
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_GET_READBACK_BUFFER_FENCE)(
        hwc2_device_t* device, hwc2_display_t display,
        int32_t* outFence);

/* setReadbackBuffer(..., buffer, releaseFence)
 * Optional for HWC2 devices
 *
 * Sets the readback buffer to be filled with the contents of the next
 * composition performed for this display (i.e., the contents present at the
 * time of the next validateDisplay/presentDisplay cycle).
 *
 * This buffer will have been allocated as described in
 * getReadbackBufferAttributes and will be interpreted as being in the dataspace
 * provided by the same.
 *
 * If there is hardware protected content on the display at the time of the next
 * composition, the area of the readback buffer covered by such content must be
 * completely black. Any areas of the buffer not covered by such content may
 * optionally be black as well.
 *
 * The release fence file descriptor provided works identically to the one
 * described for setOutputBuffer.
 *
 * This function will not be called between any call to validateDisplay and a
 * subsequent call to presentDisplay.
 *
 * Parameters:
 *   buffer - the new readback buffer
 *   releaseFence - a sync fence file descriptor as described in setOutputBuffer
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - the new readback buffer handle was invalid
 *
 * See also:
 *   getReadbackBufferAttributes
 *   getReadbackBufferFence
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_READBACK_BUFFER)(
        hwc2_device_t* device, hwc2_display_t display,
        buffer_handle_t buffer, int32_t releaseFence);

/* setVsyncEnabled(..., enabled)
 * Descriptor: HWC2_FUNCTION_SET_VSYNC_ENABLED
 * Must be provided by all HWC2 devices
 *
 * Enables or disables the vsync signal for the given display. Virtual displays
 * never generate vsync callbacks, and any attempt to enable vsync for a virtual
 * display though this function must return HWC2_ERROR_NONE and have no other
 * effect.
 *
 * Parameters:
 *   enabled - whether to enable or disable vsync
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - enabled was an invalid value
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_VSYNC_ENABLED)(
        hwc2_device_t* device, hwc2_display_t display,
        int32_t /*hwc2_vsync_t*/ enabled);

/* validateDisplay(..., outNumTypes, outNumRequests)
 * Descriptor: HWC2_FUNCTION_VALIDATE_DISPLAY
 * Must be provided by all HWC2 devices
 *
 * Instructs the device to inspect all of the layer state and determine if
 * there are any composition type changes necessary before presenting the
 * display. Permitted changes are described in the definition of
 * hwc2_composition_t above.
 *
 * Also returns the number of layer requests required
 * by the given layer configuration.
 *
 * Parameters:
 *   outNumTypes - the number of composition type changes required by the
 *       device; if greater than 0, the client must either set and validate new
 *       types, or call acceptDisplayChanges to accept the changes returned by
 *       getChangedCompositionTypes; must be the same as the number of changes
 *       returned by getChangedCompositionTypes (see the declaration of that
 *       function for more information); pointer will be non-NULL
 *   outNumRequests - the number of layer requests required by this layer
 *       configuration; must be equal to the number of layer requests returned
 *       by getDisplayRequests (see the declaration of that function for
 *       more information); pointer will be non-NULL
 *
 * Returns HWC2_ERROR_NONE if no changes are necessary and it is safe to present
 * the display using the current layer state. Otherwise returns one of the
 * following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_HAS_CHANGES - outNumTypes was greater than 0 (see parameter list
 *       for more information)
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_VALIDATE_DISPLAY)(
        hwc2_device_t* device, hwc2_display_t display,
        uint32_t* outNumTypes, uint32_t* outNumRequests);

/*
 * Layer Functions
 *
 * These are functions which operate on layers, but which do not modify state
 * that must be validated before use. See also 'Layer State Functions' below.
 *
 * All of these functions take as their first three parameters a device pointer,
 * a display handle for the display which contains the layer, and a layer
 * handle, so these parameters are omitted from the described parameter lists.
 */

/* setCursorPosition(..., x, y)
 * Descriptor: HWC2_FUNCTION_SET_CURSOR_POSITION
 * Must be provided by all HWC2 devices
 *
 * Asynchonously sets the position of a cursor layer.
 *
 * Prior to validateDisplay, a layer may be marked as HWC2_COMPOSITION_CURSOR.
 * If validation succeeds (i.e., the device does not request a composition
 * change for that layer), then once a buffer has been set for the layer and it
 * has been presented, its position may be set by this function at any time
 * between presentDisplay and any subsequent validateDisplay calls for this
 * display.
 *
 * Once validateDisplay is called, this function will not be called again until
 * the validate/present sequence is completed.
 *
 * May be called from any thread so long as it is not interleaved with the
 * validate/present sequence as described above.
 *
 * Parameters:
 *   x - the new x coordinate (in pixels from the left of the screen)
 *   y - the new y coordinate (in pixels from the top of the screen)
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_LAYER - the layer is invalid or is not currently marked as
 *       HWC2_COMPOSITION_CURSOR
 *   HWC2_ERROR_NOT_VALIDATED - the device is currently in the middle of the
 *       validate/present sequence
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_CURSOR_POSITION)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        int32_t x, int32_t y);

/* setLayerBuffer(..., buffer, acquireFence)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_BUFFER
 * Must be provided by all HWC2 devices
 *
 * Sets the buffer handle to be displayed for this layer. If the buffer
 * properties set at allocation time (width, height, format, and usage) have not
 * changed since the previous frame, it is not necessary to call validateDisplay
 * before calling presentDisplay unless new state needs to be validated in the
 * interim.
 *
 * Also provides a file descriptor referring to an acquire sync fence object,
 * which will be signaled when it is safe to read from the given buffer. If it
 * is already safe to read from the buffer, -1 may be passed instead. The
 * device must ensure that it is safe for the client to close this file
 * descriptor at any point after this function is called.
 *
 * This function must return HWC2_ERROR_NONE and have no other effect if called
 * for a layer with a composition type of HWC2_COMPOSITION_SOLID_COLOR (because
 * it has no buffer) or HWC2_COMPOSITION_SIDEBAND or HWC2_COMPOSITION_CLIENT
 * (because synchronization and buffer updates for these layers are handled
 * elsewhere).
 *
 * Parameters:
 *   buffer - the buffer handle to set
 *   acquireFence - a sync fence file descriptor as described above
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - the buffer handle passed in was invalid
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_BUFFER)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        buffer_handle_t buffer, int32_t acquireFence);

/* setLayerSurfaceDamage(..., damage)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_SURFACE_DAMAGE
 * Must be provided by all HWC2 devices
 *
 * Provides the region of the source buffer which has been modified since the
 * last frame. This region does not need to be validated before calling
 * presentDisplay.
 *
 * Once set through this function, the damage region remains the same until a
 * subsequent call to this function.
 *
 * If damage.numRects > 0, then it may be assumed that any portion of the source
 * buffer not covered by one of the rects has not been modified this frame. If
 * damage.numRects == 0, then the whole source buffer must be treated as if it
 * has been modified.
 *
 * If the layer's contents are not modified relative to the prior frame, damage
 * will contain exactly one empty rect([0, 0, 0, 0]).
 *
 * The damage rects are relative to the pre-transformed buffer, and their origin
 * is the top-left corner. They will not exceed the dimensions of the latched
 * buffer.
 *
 * Parameters:
 *   damage - the new surface damage region
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_SURFACE_DAMAGE)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        hwc_region_t damage);

/* setLayerPerFrameMetadata(..., numMetadata, metadata)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_PER_FRAME_METADATA
 * Optional for HWC2 devices
 *
 * If supported (getFunction(HWC2_FUNCTION_SET_LAYER_PER_FRAME_METADATA) is
 * non-null), sets the metadata for the given display for all following
 * frames.
 *
 * Upon returning from this function, the metadata change must have
 * fully taken effect.
 *
 * This function will only be called if getPerFrameMetadataKeys is non-NULL
 * and returns at least one key.
 *
 * Parameters:
 *   numElements is the number of elements in each of the keys and metadata arrays
 *   keys is a pointer to the array of keys.
 *   outMetadata is a pointer to the corresponding array of metadata.
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_DISPLAY - an invalid display handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - metadata is not valid
 *   HWC2_ERROR_UNSUPPORTED - metadata is not supported on this display
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_PER_FRAME_METADATA)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        uint32_t numElements, const int32_t* /*hw2_per_frame_metadata_key_t*/ keys,
        const float* metadata);

/*
 * Layer State Functions
 *
 * These functions modify the state of a given layer. They do not take effect
 * until the display configuration is successfully validated with
 * validateDisplay and the display contents are presented with presentDisplay.
 *
 * All of these functions take as their first three parameters a device pointer,
 * a display handle for the display which contains the layer, and a layer
 * handle, so these parameters are omitted from the described parameter lists.
 */

/* setLayerBlendMode(..., mode)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_BLEND_MODE
 * Must be provided by all HWC2 devices
 *
 * Sets the blend mode of the given layer.
 *
 * Parameters:
 *   mode - the new blend mode
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - an invalid blend mode was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_BLEND_MODE)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        int32_t /*hwc2_blend_mode_t*/ mode);

/* setLayerColor(..., color)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_COLOR
 * Must be provided by all HWC2 devices
 *
 * Sets the color of the given layer. If the composition type of the layer is
 * not HWC2_COMPOSITION_SOLID_COLOR, this call must return HWC2_ERROR_NONE and
 * have no other effect.
 *
 * Parameters:
 *   color - the new color
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_COLOR)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        hwc_color_t color);

/* setLayerFloatColor(..., color)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_FLOAT_COLOR
 * Provided by HWC2 devices which don't return nullptr function pointer.
 *
 * Sets the color of the given layer. If the composition type of the layer is
 * not HWC2_COMPOSITION_SOLID_COLOR, this call must return HWC2_ERROR_NONE and
 * have no other effect.
 *
 * Parameters:
 *   color - the new color in float type, rage is [0.0, 1.0], the colorspace is
 *   defined by the dataspace that gets set by calling setLayerDataspace.
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_FLOAT_COLOR)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        hwc_float_color_t color);

/* setLayerCompositionType(..., type)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_COMPOSITION_TYPE
 * Must be provided by all HWC2 devices
 *
 * Sets the desired composition type of the given layer. During validateDisplay,
 * the device may request changes to the composition types of any of the layers
 * as described in the definition of hwc2_composition_t above.
 *
 * Parameters:
 *   type - the new composition type
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - an invalid composition type was passed in
 *   HWC2_ERROR_UNSUPPORTED - a valid composition type was passed in, but it is
 *       not supported by this device
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_COMPOSITION_TYPE)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        int32_t /*hwc2_composition_t*/ type);

/* setLayerDataspace(..., dataspace)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_DATASPACE
 * Must be provided by all HWC2 devices
 *
 * Sets the dataspace that the current buffer on this layer is in.
 *
 * The dataspace provides more information about how to interpret the buffer
 * contents, such as the encoding standard and color transform.
 *
 * See the values of android_dataspace_t in <system/graphics.h> for more
 * information.
 *
 * Parameters:
 *   dataspace - the new dataspace
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_DATASPACE)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        int32_t /*android_dataspace_t*/ dataspace);

/* setLayerDisplayFrame(..., frame)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_DISPLAY_FRAME
 * Must be provided by all HWC2 devices
 *
 * Sets the display frame (the portion of the display covered by a layer) of the
 * given layer. This frame will not exceed the display dimensions.
 *
 * Parameters:
 *   frame - the new display frame
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_DISPLAY_FRAME)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        hwc_rect_t frame);

/* setLayerPlaneAlpha(..., alpha)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_PLANE_ALPHA
 * Must be provided by all HWC2 devices
 *
 * Sets an alpha value (a floating point value in the range [0.0, 1.0]) which
 * will be applied to the whole layer. It can be conceptualized as a
 * preprocessing step which applies the following function:
 *   if (blendMode == HWC2_BLEND_MODE_PREMULTIPLIED)
 *       out.rgb = in.rgb * planeAlpha
 *   out.a = in.a * planeAlpha
 *
 * If the device does not support this operation on a layer which is marked
 * HWC2_COMPOSITION_DEVICE, it must request a composition type change to
 * HWC2_COMPOSITION_CLIENT upon the next validateDisplay call.
 *
 * Parameters:
 *   alpha - the plane alpha value to apply
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_PLANE_ALPHA)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        float alpha);

/* setLayerSidebandStream(..., stream)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_SIDEBAND_STREAM
 * Provided by HWC2 devices which support HWC2_CAPABILITY_SIDEBAND_STREAM
 *
 * Sets the sideband stream for this layer. If the composition type of the given
 * layer is not HWC2_COMPOSITION_SIDEBAND, this call must return HWC2_ERROR_NONE
 * and have no other effect.
 *
 * Parameters:
 *   stream - the new sideband stream
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - an invalid sideband stream was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_SIDEBAND_STREAM)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        const native_handle_t* stream);

/* setLayerSourceCrop(..., crop)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_SOURCE_CROP
 * Must be provided by all HWC2 devices
 *
 * Sets the source crop (the portion of the source buffer which will fill the
 * display frame) of the given layer. This crop rectangle will not exceed the
 * dimensions of the latched buffer.
 *
 * If the device is not capable of supporting a true float source crop (i.e., it
 * will truncate or round the floats to integers), it should set this layer to
 * HWC2_COMPOSITION_CLIENT when crop is non-integral for the most accurate
 * rendering.
 *
 * If the device cannot support float source crops, but still wants to handle
 * the layer, it should use the following code (or similar) to convert to
 * an integer crop:
 *   intCrop.left = (int) ceilf(crop.left);
 *   intCrop.top = (int) ceilf(crop.top);
 *   intCrop.right = (int) floorf(crop.right);
 *   intCrop.bottom = (int) floorf(crop.bottom);
 *
 * Parameters:
 *   crop - the new source crop
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_SOURCE_CROP)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        hwc_frect_t crop);

/* setLayerTransform(..., transform)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_TRANSFORM
 * Must be provided by all HWC2 devices
 *
 * Sets the transform (rotation/flip) of the given layer.
 *
 * Parameters:
 *   transform - the new transform
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 *   HWC2_ERROR_BAD_PARAMETER - an invalid transform was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_TRANSFORM)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        int32_t /*hwc_transform_t*/ transform);

/* setLayerVisibleRegion(..., visible)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_VISIBLE_REGION
 * Must be provided by all HWC2 devices
 *
 * Specifies the portion of the layer that is visible, including portions under
 * translucent areas of other layers. The region is in screen space, and will
 * not exceed the dimensions of the screen.
 *
 * Parameters:
 *   visible - the new visible region, in screen space
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_VISIBLE_REGION)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        hwc_region_t visible);

/* setLayerZOrder(..., z)
 * Descriptor: HWC2_FUNCTION_SET_LAYER_Z_ORDER
 * Must be provided by all HWC2 devices
 *
 * Sets the desired Z order (height) of the given layer. A layer with a greater
 * Z value occludes a layer with a lesser Z value.
 *
 * Parameters:
 *   z - the new Z order
 *
 * Returns HWC2_ERROR_NONE or one of the following errors:
 *   HWC2_ERROR_BAD_LAYER - an invalid layer handle was passed in
 */
typedef int32_t /*hwc2_error_t*/ (*HWC2_PFN_SET_LAYER_Z_ORDER)(
        hwc2_device_t* device, hwc2_display_t display, hwc2_layer_t layer,
        uint32_t z);

__END_DECLS

#endif
