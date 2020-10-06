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

#ifndef ANDROID_INCLUDE_CAMERA3_H
#define ANDROID_INCLUDE_CAMERA3_H

#include <system/camera_metadata.h>
#include "camera_common.h"

/**
 * Camera device HAL 3.3 [ CAMERA_DEVICE_API_VERSION_3_3 ]
 *
 * This is the current recommended version of the camera device HAL.
 *
 * Supports the android.hardware.Camera API, and as of v3.2, the
 * android.hardware.camera2 API in LIMITED or FULL modes.
 *
 * Camera devices that support this version of the HAL must return
 * CAMERA_DEVICE_API_VERSION_3_3 in camera_device_t.common.version and in
 * camera_info_t.device_version (from camera_module_t.get_camera_info).
 *
 * CAMERA_DEVICE_API_VERSION_3_3:
 *    Camera modules that may contain version 3.3 devices must implement at
 *    least version 2.2 of the camera module interface (as defined by
 *    camera_module_t.common.module_api_version).
 *
 * CAMERA_DEVICE_API_VERSION_3_2:
 *    Camera modules that may contain version 3.2 devices must implement at
 *    least version 2.2 of the camera module interface (as defined by
 *    camera_module_t.common.module_api_version).
 *
 * <= CAMERA_DEVICE_API_VERSION_3_1:
 *    Camera modules that may contain version 3.1 (or 3.0) devices must
 *    implement at least version 2.0 of the camera module interface
 *    (as defined by camera_module_t.common.module_api_version).
 *
 * See camera_common.h for more versioning details.
 *
 * Documentation index:
 *   S1. Version history
 *   S2. Startup and operation sequencing
 *   S3. Operational modes
 *   S4. 3A modes and state machines
 *   S5. Cropping
 *   S6. Error management
 *   S7. Key Performance Indicator (KPI) glossary
 *   S8. Sample Use Cases
 *   S9. Notes on Controls and Metadata
 *   S10. Reprocessing flow and controls
 */

/**
 * S1. Version history:
 *
 * 1.0: Initial Android camera HAL (Android 4.0) [camera.h]:
 *
 *   - Converted from C++ CameraHardwareInterface abstraction layer.
 *
 *   - Supports android.hardware.Camera API.
 *
 * 2.0: Initial release of expanded-capability HAL (Android 4.2) [camera2.h]:
 *
 *   - Sufficient for implementing existing android.hardware.Camera API.
 *
 *   - Allows for ZSL queue in camera service layer
 *
 *   - Not tested for any new features such manual capture control, Bayer RAW
 *     capture, reprocessing of RAW data.
 *
 * 3.0: First revision of expanded-capability HAL:
 *
 *   - Major version change since the ABI is completely different. No change to
 *     the required hardware capabilities or operational model from 2.0.
 *
 *   - Reworked input request and stream queue interfaces: Framework calls into
 *     HAL with next request and stream buffers already dequeued. Sync framework
 *     support is included, necessary for efficient implementations.
 *
 *   - Moved triggers into requests, most notifications into results.
 *
 *   - Consolidated all callbacks into framework into one structure, and all
 *     setup methods into a single initialize() call.
 *
 *   - Made stream configuration into a single call to simplify stream
 *     management. Bidirectional streams replace STREAM_FROM_STREAM construct.
 *
 *   - Limited mode semantics for older/limited hardware devices.
 *
 * 3.1: Minor revision of expanded-capability HAL:
 *
 *   - configure_streams passes consumer usage flags to the HAL.
 *
 *   - flush call to drop all in-flight requests/buffers as fast as possible.
 *
 * 3.2: Minor revision of expanded-capability HAL:
 *
 *   - Deprecates get_metadata_vendor_tag_ops.  Please use get_vendor_tag_ops
 *     in camera_common.h instead.
 *
 *   - register_stream_buffers deprecated. All gralloc buffers provided
 *     by framework to HAL in process_capture_request may be new at any time.
 *
 *   - add partial result support. process_capture_result may be called
 *     multiple times with a subset of the available result before the full
 *     result is available.
 *
 *   - add manual template to camera3_request_template. The applications may
 *     use this template to control the capture settings directly.
 *
 *   - Rework the bidirectional and input stream specifications.
 *
 *   - change the input buffer return path. The buffer is returned in
 *     process_capture_result instead of process_capture_request.
 *
 * 3.3: Minor revision of expanded-capability HAL:
 *
 *   - OPAQUE and YUV reprocessing API updates.
 *
 *   - Basic support for depth output buffers.
 *
 *   - Addition of data_space field to camera3_stream_t.
 *
 *   - Addition of rotation field to camera3_stream_t.
 *
 *   - Addition of camera3 stream configuration operation mode to camera3_stream_configuration_t
 *
 */

/**
 * S2. Startup and general expected operation sequence:
 *
 * 1. Framework calls camera_module_t->common.open(), which returns a
 *    hardware_device_t structure.
 *
 * 2. Framework inspects the hardware_device_t->version field, and instantiates
 *    the appropriate handler for that version of the camera hardware device. In
 *    case the version is CAMERA_DEVICE_API_VERSION_3_0, the device is cast to
 *    a camera3_device_t.
 *
 * 3. Framework calls camera3_device_t->ops->initialize() with the framework
 *    callback function pointers. This will only be called this one time after
 *    open(), before any other functions in the ops structure are called.
 *
 * 4. The framework calls camera3_device_t->ops->configure_streams() with a list
 *    of input/output streams to the HAL device.
 *
 * 5. <= CAMERA_DEVICE_API_VERSION_3_1:
 *
 *    The framework allocates gralloc buffers and calls
 *    camera3_device_t->ops->register_stream_buffers() for at least one of the
 *    output streams listed in configure_streams. The same stream is registered
 *    only once.
 *
 *    >= CAMERA_DEVICE_API_VERSION_3_2:
 *
 *    camera3_device_t->ops->register_stream_buffers() is not called and must
 *    be NULL.
 *
 * 6. The framework requests default settings for some number of use cases with
 *    calls to camera3_device_t->ops->construct_default_request_settings(). This
 *    may occur any time after step 3.
 *
 * 7. The framework constructs and sends the first capture request to the HAL,
 *    with settings based on one of the sets of default settings, and with at
 *    least one output stream, which has been registered earlier by the
 *    framework. This is sent to the HAL with
 *    camera3_device_t->ops->process_capture_request(). The HAL must block the
 *    return of this call until it is ready for the next request to be sent.
 *
 *    >= CAMERA_DEVICE_API_VERSION_3_2:
 *
 *    The buffer_handle_t provided in the camera3_stream_buffer_t array
 *    in the camera3_capture_request_t may be new and never-before-seen
 *    by the HAL on any given new request.
 *
 * 8. The framework continues to submit requests, and call
 *    construct_default_request_settings to get default settings buffers for
 *    other use cases.
 *
 *    <= CAMERA_DEVICE_API_VERSION_3_1:
 *
 *    The framework may call register_stream_buffers() at this time for
 *    not-yet-registered streams.
 *
 * 9. When the capture of a request begins (sensor starts exposing for the
 *    capture) or processing a reprocess request begins, the HAL
 *    calls camera3_callback_ops_t->notify() with the SHUTTER event, including
 *    the frame number and the timestamp for start of exposure. For a reprocess
 *    request, the timestamp must be the start of exposure of the input image
 *    which can be looked up with android.sensor.timestamp from
 *    camera3_capture_request_t.settings when process_capture_request() is
 *    called.
 *
 *    <= CAMERA_DEVICE_API_VERSION_3_1:
 *
 *    This notify call must be made before the first call to
 *    process_capture_result() for that frame number.
 *
 *    >= CAMERA_DEVICE_API_VERSION_3_2:
 *
 *    The camera3_callback_ops_t->notify() call with the SHUTTER event should
 *    be made as early as possible since the framework will be unable to
 *    deliver gralloc buffers to the application layer (for that frame) until
 *    it has a valid timestamp for the start of exposure (or the input image's
 *    start of exposure for a reprocess request).
 *
 *    Both partial metadata results and the gralloc buffers may be sent to the
 *    framework at any time before or after the SHUTTER event.
 *
 * 10. After some pipeline delay, the HAL begins to return completed captures to
 *    the framework with camera3_callback_ops_t->process_capture_result(). These
 *    are returned in the same order as the requests were submitted. Multiple
 *    requests can be in flight at once, depending on the pipeline depth of the
 *    camera HAL device.
 *
 *    >= CAMERA_DEVICE_API_VERSION_3_2:
 *
 *    Once a buffer is returned by process_capture_result as part of the
 *    camera3_stream_buffer_t array, and the fence specified by release_fence
 *    has been signaled (this is a no-op for -1 fences), the ownership of that
 *    buffer is considered to be transferred back to the framework. After that,
 *    the HAL must no longer retain that particular buffer, and the
 *    framework may clean up the memory for it immediately.
 *
 *    process_capture_result may be called multiple times for a single frame,
 *    each time with a new disjoint piece of metadata and/or set of gralloc
 *    buffers. The framework will accumulate these partial metadata results
 *    into one result.
 *
 *    In particular, it is legal for a process_capture_result to be called
 *    simultaneously for both a frame N and a frame N+1 as long as the
 *    above rule holds for gralloc buffers (both input and output).
 *
 * 11. After some time, the framework may stop submitting new requests, wait for
 *    the existing captures to complete (all buffers filled, all results
 *    returned), and then call configure_streams() again. This resets the camera
 *    hardware and pipeline for a new set of input/output streams. Some streams
 *    may be reused from the previous configuration; if these streams' buffers
 *    had already been registered with the HAL, they will not be registered
 *    again. The framework then continues from step 7, if at least one
 *    registered output stream remains (otherwise, step 5 is required first).
 *
 * 12. Alternatively, the framework may call camera3_device_t->common->close()
 *    to end the camera session. This may be called at any time when no other
 *    calls from the framework are active, although the call may block until all
 *    in-flight captures have completed (all results returned, all buffers
 *    filled). After the close call returns, no more calls to the
 *    camera3_callback_ops_t functions are allowed from the HAL. Once the
 *    close() call is underway, the framework may not call any other HAL device
 *    functions.
 *
 * 13. In case of an error or other asynchronous event, the HAL must call
 *    camera3_callback_ops_t->notify() with the appropriate error/event
 *    message. After returning from a fatal device-wide error notification, the
 *    HAL should act as if close() had been called on it. However, the HAL must
 *    either cancel or complete all outstanding captures before calling
 *    notify(), so that once notify() is called with a fatal error, the
 *    framework will not receive further callbacks from the device. Methods
 *    besides close() should return -ENODEV or NULL after the notify() method
 *    returns from a fatal error message.
 */

/**
 * S3. Operational modes:
 *
 * The camera 3 HAL device can implement one of two possible operational modes;
 * limited and full. Full support is expected from new higher-end
 * devices. Limited mode has hardware requirements roughly in line with those
 * for a camera HAL device v1 implementation, and is expected from older or
 * inexpensive devices. Full is a strict superset of limited, and they share the
 * same essential operational flow, as documented above.
 *
 * The HAL must indicate its level of support with the
 * android.info.supportedHardwareLevel static metadata entry, with 0 indicating
 * limited mode, and 1 indicating full mode support.
 *
 * Roughly speaking, limited-mode devices do not allow for application control
 * of capture settings (3A control only), high-rate capture of high-resolution
 * images, raw sensor readout, or support for YUV output streams above maximum
 * recording resolution (JPEG only for large images).
 *
 * ** Details of limited mode behavior:
 *
 * - Limited-mode devices do not need to implement accurate synchronization
 *   between capture request settings and the actual image data
 *   captured. Instead, changes to settings may take effect some time in the
 *   future, and possibly not for the same output frame for each settings
 *   entry. Rapid changes in settings may result in some settings never being
 *   used for a capture. However, captures that include high-resolution output
 *   buffers ( > 1080p ) have to use the settings as specified (but see below
 *   for processing rate).
 *
 * - Limited-mode devices do not need to support most of the
 *   settings/result/static info metadata. Specifically, only the following settings
 *   are expected to be consumed or produced by a limited-mode HAL device:
 *
 *   android.control.aeAntibandingMode (controls and dynamic)
 *   android.control.aeExposureCompensation (controls and dynamic)
 *   android.control.aeLock (controls and dynamic)
 *   android.control.aeMode (controls and dynamic)
 *   android.control.aeRegions (controls and dynamic)
 *   android.control.aeTargetFpsRange (controls and dynamic)
 *   android.control.aePrecaptureTrigger (controls and dynamic)
 *   android.control.afMode (controls and dynamic)
 *   android.control.afRegions (controls and dynamic)
 *   android.control.awbLock (controls and dynamic)
 *   android.control.awbMode (controls and dynamic)
 *   android.control.awbRegions (controls and dynamic)
 *   android.control.captureIntent (controls and dynamic)
 *   android.control.effectMode (controls and dynamic)
 *   android.control.mode (controls and dynamic)
 *   android.control.sceneMode (controls and dynamic)
 *   android.control.videoStabilizationMode (controls and dynamic)
 *   android.control.aeAvailableAntibandingModes (static)
 *   android.control.aeAvailableModes (static)
 *   android.control.aeAvailableTargetFpsRanges (static)
 *   android.control.aeCompensationRange (static)
 *   android.control.aeCompensationStep (static)
 *   android.control.afAvailableModes (static)
 *   android.control.availableEffects (static)
 *   android.control.availableSceneModes (static)
 *   android.control.availableVideoStabilizationModes (static)
 *   android.control.awbAvailableModes (static)
 *   android.control.maxRegions (static)
 *   android.control.sceneModeOverrides (static)
 *   android.control.aeState (dynamic)
 *   android.control.afState (dynamic)
 *   android.control.awbState (dynamic)
 *
 *   android.flash.mode (controls and dynamic)
 *   android.flash.info.available (static)
 *
 *   android.info.supportedHardwareLevel (static)
 *
 *   android.jpeg.gpsCoordinates (controls and dynamic)
 *   android.jpeg.gpsProcessingMethod (controls and dynamic)
 *   android.jpeg.gpsTimestamp (controls and dynamic)
 *   android.jpeg.orientation (controls and dynamic)
 *   android.jpeg.quality (controls and dynamic)
 *   android.jpeg.thumbnailQuality (controls and dynamic)
 *   android.jpeg.thumbnailSize (controls and dynamic)
 *   android.jpeg.availableThumbnailSizes (static)
 *   android.jpeg.maxSize (static)
 *
 *   android.lens.info.minimumFocusDistance (static)
 *
 *   android.request.id (controls and dynamic)
 *
 *   android.scaler.cropRegion (controls and dynamic)
 *   android.scaler.availableStreamConfigurations (static)
 *   android.scaler.availableMinFrameDurations (static)
 *   android.scaler.availableStallDurations (static)
 *   android.scaler.availableMaxDigitalZoom (static)
 *   android.scaler.maxDigitalZoom (static)
 *   android.scaler.croppingType (static)
 *
 *   android.sensor.orientation (static)
 *   android.sensor.timestamp (dynamic)
 *
 *   android.statistics.faceDetectMode (controls and dynamic)
 *   android.statistics.info.availableFaceDetectModes (static)
 *   android.statistics.faceIds (dynamic)
 *   android.statistics.faceLandmarks (dynamic)
 *   android.statistics.faceRectangles (dynamic)
 *   android.statistics.faceScores (dynamic)
 *
 *   android.sync.frameNumber (dynamic)
 *   android.sync.maxLatency (static)
 *
 * - Captures in limited mode that include high-resolution (> 1080p) output
 *   buffers may block in process_capture_request() until all the output buffers
 *   have been filled. A full-mode HAL device must process sequences of
 *   high-resolution requests at the rate indicated in the static metadata for
 *   that pixel format. The HAL must still call process_capture_result() to
 *   provide the output; the framework must simply be prepared for
 *   process_capture_request() to block until after process_capture_result() for
 *   that request completes for high-resolution captures for limited-mode
 *   devices.
 *
 * - Full-mode devices must support below additional capabilities:
 *   - 30fps at maximum resolution is preferred, more than 20fps is required.
 *   - Per frame control (android.sync.maxLatency == PER_FRAME_CONTROL).
 *   - Sensor manual control metadata. See MANUAL_SENSOR defined in
 *     android.request.availableCapabilities.
 *   - Post-processing manual control metadata. See MANUAL_POST_PROCESSING defined
 *     in android.request.availableCapabilities.
 *
 */

/**
 * S4. 3A modes and state machines:
 *
 * While the actual 3A algorithms are up to the HAL implementation, a high-level
 * state machine description is defined by the HAL interface, to allow the HAL
 * device and the framework to communicate about the current state of 3A, and to
 * trigger 3A events.
 *
 * When the device is opened, all the individual 3A states must be
 * STATE_INACTIVE. Stream configuration does not reset 3A. For example, locked
 * focus must be maintained across the configure() call.
 *
 * Triggering a 3A action involves simply setting the relevant trigger entry in
 * the settings for the next request to indicate start of trigger. For example,
 * the trigger for starting an autofocus scan is setting the entry
 * ANDROID_CONTROL_AF_TRIGGER to ANDROID_CONTROL_AF_TRIGGER_START for one
 * request, and cancelling an autofocus scan is triggered by setting
 * ANDROID_CONTROL_AF_TRIGGER to ANDROID_CONTRL_AF_TRIGGER_CANCEL. Otherwise,
 * the entry will not exist, or be set to ANDROID_CONTROL_AF_TRIGGER_IDLE. Each
 * request with a trigger entry set to a non-IDLE value will be treated as an
 * independent triggering event.
 *
 * At the top level, 3A is controlled by the ANDROID_CONTROL_MODE setting, which
 * selects between no 3A (ANDROID_CONTROL_MODE_OFF), normal AUTO mode
 * (ANDROID_CONTROL_MODE_AUTO), and using the scene mode setting
 * (ANDROID_CONTROL_USE_SCENE_MODE).
 *
 * - In OFF mode, each of the individual AE/AF/AWB modes are effectively OFF,
 *   and none of the capture controls may be overridden by the 3A routines.
 *
 * - In AUTO mode, Auto-focus, auto-exposure, and auto-whitebalance all run
 *   their own independent algorithms, and have their own mode, state, and
 *   trigger metadata entries, as listed in the next section.
 *
 * - In USE_SCENE_MODE, the value of the ANDROID_CONTROL_SCENE_MODE entry must
 *   be used to determine the behavior of 3A routines. In SCENE_MODEs other than
 *   FACE_PRIORITY, the HAL must override the values of
 *   ANDROId_CONTROL_AE/AWB/AF_MODE to be the mode it prefers for the selected
 *   SCENE_MODE. For example, the HAL may prefer SCENE_MODE_NIGHT to use
 *   CONTINUOUS_FOCUS AF mode. Any user selection of AE/AWB/AF_MODE when scene
 *   must be ignored for these scene modes.
 *
 * - For SCENE_MODE_FACE_PRIORITY, the AE/AWB/AF_MODE controls work as in
 *   ANDROID_CONTROL_MODE_AUTO, but the 3A routines must bias toward metering
 *   and focusing on any detected faces in the scene.
 *
 * S4.1. Auto-focus settings and result entries:
 *
 *  Main metadata entries:
 *
 *   ANDROID_CONTROL_AF_MODE: Control for selecting the current autofocus
 *      mode. Set by the framework in the request settings.
 *
 *     AF_MODE_OFF: AF is disabled; the framework/app directly controls lens
 *         position.
 *
 *     AF_MODE_AUTO: Single-sweep autofocus. No lens movement unless AF is
 *         triggered.
 *
 *     AF_MODE_MACRO: Single-sweep up-close autofocus. No lens movement unless
 *         AF is triggered.
 *
 *     AF_MODE_CONTINUOUS_VIDEO: Smooth continuous focusing, for recording
 *         video. Triggering immediately locks focus in current
 *         position. Canceling resumes cotinuous focusing.
 *
 *     AF_MODE_CONTINUOUS_PICTURE: Fast continuous focusing, for
 *        zero-shutter-lag still capture. Triggering locks focus once currently
 *        active sweep concludes. Canceling resumes continuous focusing.
 *
 *     AF_MODE_EDOF: Advanced extended depth of field focusing. There is no
 *        autofocus scan, so triggering one or canceling one has no effect.
 *        Images are focused automatically by the HAL.
 *
 *   ANDROID_CONTROL_AF_STATE: Dynamic metadata describing the current AF
 *       algorithm state, reported by the HAL in the result metadata.
 *
 *     AF_STATE_INACTIVE: No focusing has been done, or algorithm was
 *        reset. Lens is not moving. Always the state for MODE_OFF or MODE_EDOF.
 *        When the device is opened, it must start in this state.
 *
 *     AF_STATE_PASSIVE_SCAN: A continuous focus algorithm is currently scanning
 *        for good focus. The lens is moving.
 *
 *     AF_STATE_PASSIVE_FOCUSED: A continuous focus algorithm believes it is
 *        well focused. The lens is not moving. The HAL may spontaneously leave
 *        this state.
 *
 *     AF_STATE_PASSIVE_UNFOCUSED: A continuous focus algorithm believes it is
 *        not well focused. The lens is not moving. The HAL may spontaneously
 *        leave this state.
 *
 *     AF_STATE_ACTIVE_SCAN: A scan triggered by the user is underway.
 *
 *     AF_STATE_FOCUSED_LOCKED: The AF algorithm believes it is focused. The
 *        lens is not moving.
 *
 *     AF_STATE_NOT_FOCUSED_LOCKED: The AF algorithm has been unable to
 *        focus. The lens is not moving.
 *
 *   ANDROID_CONTROL_AF_TRIGGER: Control for starting an autofocus scan, the
 *       meaning of which is mode- and state- dependent. Set by the framework in
 *       the request settings.
 *
 *     AF_TRIGGER_IDLE: No current trigger.
 *
 *     AF_TRIGGER_START: Trigger start of AF scan. Effect is mode and state
 *         dependent.
 *
 *     AF_TRIGGER_CANCEL: Cancel current AF scan if any, and reset algorithm to
 *         default.
 *
 *  Additional metadata entries:
 *
 *   ANDROID_CONTROL_AF_REGIONS: Control for selecting the regions of the FOV
 *       that should be used to determine good focus. This applies to all AF
 *       modes that scan for focus. Set by the framework in the request
 *       settings.
 *
 * S4.2. Auto-exposure settings and result entries:
 *
 *  Main metadata entries:
 *
 *   ANDROID_CONTROL_AE_MODE: Control for selecting the current auto-exposure
 *       mode. Set by the framework in the request settings.
 *
 *     AE_MODE_OFF: Autoexposure is disabled; the user controls exposure, gain,
 *         frame duration, and flash.
 *
 *     AE_MODE_ON: Standard autoexposure, with flash control disabled. User may
 *         set flash to fire or to torch mode.
 *
 *     AE_MODE_ON_AUTO_FLASH: Standard autoexposure, with flash on at HAL's
 *         discretion for precapture and still capture. User control of flash
 *         disabled.
 *
 *     AE_MODE_ON_ALWAYS_FLASH: Standard autoexposure, with flash always fired
 *         for capture, and at HAL's discretion for precapture.. User control of
 *         flash disabled.
 *
 *     AE_MODE_ON_AUTO_FLASH_REDEYE: Standard autoexposure, with flash on at
 *         HAL's discretion for precapture and still capture. Use a flash burst
 *         at end of precapture sequence to reduce redeye in the final
 *         picture. User control of flash disabled.
 *
 *   ANDROID_CONTROL_AE_STATE: Dynamic metadata describing the current AE
 *       algorithm state, reported by the HAL in the result metadata.
 *
 *     AE_STATE_INACTIVE: Initial AE state after mode switch. When the device is
 *         opened, it must start in this state.
 *
 *     AE_STATE_SEARCHING: AE is not converged to a good value, and is adjusting
 *         exposure parameters.
 *
 *     AE_STATE_CONVERGED: AE has found good exposure values for the current
 *         scene, and the exposure parameters are not changing. HAL may
 *         spontaneously leave this state to search for better solution.
 *
 *     AE_STATE_LOCKED: AE has been locked with the AE_LOCK control. Exposure
 *         values are not changing.
 *
 *     AE_STATE_FLASH_REQUIRED: The HAL has converged exposure, but believes
 *         flash is required for a sufficiently bright picture. Used for
 *         determining if a zero-shutter-lag frame can be used.
 *
 *     AE_STATE_PRECAPTURE: The HAL is in the middle of a precapture
 *         sequence. Depending on AE mode, this mode may involve firing the
 *         flash for metering, or a burst of flash pulses for redeye reduction.
 *
 *   ANDROID_CONTROL_AE_PRECAPTURE_TRIGGER: Control for starting a metering
 *       sequence before capturing a high-quality image. Set by the framework in
 *       the request settings.
 *
 *      PRECAPTURE_TRIGGER_IDLE: No current trigger.
 *
 *      PRECAPTURE_TRIGGER_START: Start a precapture sequence. The HAL should
 *         use the subsequent requests to measure good exposure/white balance
 *         for an upcoming high-resolution capture.
 *
 *  Additional metadata entries:
 *
 *   ANDROID_CONTROL_AE_LOCK: Control for locking AE controls to their current
 *       values
 *
 *   ANDROID_CONTROL_AE_EXPOSURE_COMPENSATION: Control for adjusting AE
 *       algorithm target brightness point.
 *
 *   ANDROID_CONTROL_AE_TARGET_FPS_RANGE: Control for selecting the target frame
 *       rate range for the AE algorithm. The AE routine cannot change the frame
 *       rate to be outside these bounds.
 *
 *   ANDROID_CONTROL_AE_REGIONS: Control for selecting the regions of the FOV
 *       that should be used to determine good exposure levels. This applies to
 *       all AE modes besides OFF.
 *
 * S4.3. Auto-whitebalance settings and result entries:
 *
 *  Main metadata entries:
 *
 *   ANDROID_CONTROL_AWB_MODE: Control for selecting the current white-balance
 *       mode.
 *
 *     AWB_MODE_OFF: Auto-whitebalance is disabled. User controls color matrix.
 *
 *     AWB_MODE_AUTO: Automatic white balance is enabled; 3A controls color
 *        transform, possibly using more complex transforms than a simple
 *        matrix.
 *
 *     AWB_MODE_INCANDESCENT: Fixed white balance settings good for indoor
 *        incandescent (tungsten) lighting, roughly 2700K.
 *
 *     AWB_MODE_FLUORESCENT: Fixed white balance settings good for fluorescent
 *        lighting, roughly 5000K.
 *
 *     AWB_MODE_WARM_FLUORESCENT: Fixed white balance settings good for
 *        fluorescent lighting, roughly 3000K.
 *
 *     AWB_MODE_DAYLIGHT: Fixed white balance settings good for daylight,
 *        roughly 5500K.
 *
 *     AWB_MODE_CLOUDY_DAYLIGHT: Fixed white balance settings good for clouded
 *        daylight, roughly 6500K.
 *
 *     AWB_MODE_TWILIGHT: Fixed white balance settings good for
 *        near-sunset/sunrise, roughly 15000K.
 *
 *     AWB_MODE_SHADE: Fixed white balance settings good for areas indirectly
 *        lit by the sun, roughly 7500K.
 *
 *   ANDROID_CONTROL_AWB_STATE: Dynamic metadata describing the current AWB
 *       algorithm state, reported by the HAL in the result metadata.
 *
 *     AWB_STATE_INACTIVE: Initial AWB state after mode switch. When the device
 *         is opened, it must start in this state.
 *
 *     AWB_STATE_SEARCHING: AWB is not converged to a good value, and is
 *         changing color adjustment parameters.
 *
 *     AWB_STATE_CONVERGED: AWB has found good color adjustment values for the
 *         current scene, and the parameters are not changing. HAL may
 *         spontaneously leave this state to search for better solution.
 *
 *     AWB_STATE_LOCKED: AWB has been locked with the AWB_LOCK control. Color
 *         adjustment values are not changing.
 *
 *  Additional metadata entries:
 *
 *   ANDROID_CONTROL_AWB_LOCK: Control for locking AWB color adjustments to
 *       their current values.
 *
 *   ANDROID_CONTROL_AWB_REGIONS: Control for selecting the regions of the FOV
 *       that should be used to determine good color balance. This applies only
 *       to auto-WB mode.
 *
 * S4.4. General state machine transition notes
 *
 *   Switching between AF, AE, or AWB modes always resets the algorithm's state
 *   to INACTIVE.  Similarly, switching between CONTROL_MODE or
 *   CONTROL_SCENE_MODE if CONTROL_MODE == USE_SCENE_MODE resets all the
 *   algorithm states to INACTIVE.
 *
 *   The tables below are per-mode.
 *
 * S4.5. AF state machines
 *
 *                       when enabling AF or changing AF mode
 *| state              | trans. cause  | new state          | notes            |
 *+--------------------+---------------+--------------------+------------------+
 *| Any                | AF mode change| INACTIVE           |                  |
 *+--------------------+---------------+--------------------+------------------+
 *
 *                            mode = AF_MODE_OFF or AF_MODE_EDOF
 *| state              | trans. cause  | new state          | notes            |
 *+--------------------+---------------+--------------------+------------------+
 *| INACTIVE           |               | INACTIVE           | Never changes    |
 *+--------------------+---------------+--------------------+------------------+
 *
 *                            mode = AF_MODE_AUTO or AF_MODE_MACRO
 *| state              | trans. cause  | new state          | notes            |
 *+--------------------+---------------+--------------------+------------------+
 *| INACTIVE           | AF_TRIGGER    | ACTIVE_SCAN        | Start AF sweep   |
 *|                    |               |                    | Lens now moving  |
 *+--------------------+---------------+--------------------+------------------+
 *| ACTIVE_SCAN        | AF sweep done | FOCUSED_LOCKED     | If AF successful |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| ACTIVE_SCAN        | AF sweep done | NOT_FOCUSED_LOCKED | If AF successful |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| ACTIVE_SCAN        | AF_CANCEL     | INACTIVE           | Cancel/reset AF  |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| FOCUSED_LOCKED     | AF_CANCEL     | INACTIVE           | Cancel/reset AF  |
 *+--------------------+---------------+--------------------+------------------+
 *| FOCUSED_LOCKED     | AF_TRIGGER    | ACTIVE_SCAN        | Start new sweep  |
 *|                    |               |                    | Lens now moving  |
 *+--------------------+---------------+--------------------+------------------+
 *| NOT_FOCUSED_LOCKED | AF_CANCEL     | INACTIVE           | Cancel/reset AF  |
 *+--------------------+---------------+--------------------+------------------+
 *| NOT_FOCUSED_LOCKED | AF_TRIGGER    | ACTIVE_SCAN        | Start new sweep  |
 *|                    |               |                    | Lens now moving  |
 *+--------------------+---------------+--------------------+------------------+
 *| All states         | mode change   | INACTIVE           |                  |
 *+--------------------+---------------+--------------------+------------------+
 *
 *                            mode = AF_MODE_CONTINUOUS_VIDEO
 *| state              | trans. cause  | new state          | notes            |
 *+--------------------+---------------+--------------------+------------------+
 *| INACTIVE           | HAL initiates | PASSIVE_SCAN       | Start AF scan    |
 *|                    | new scan      |                    | Lens now moving  |
 *+--------------------+---------------+--------------------+------------------+
 *| INACTIVE           | AF_TRIGGER    | NOT_FOCUSED_LOCKED | AF state query   |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_SCAN       | HAL completes | PASSIVE_FOCUSED    | End AF scan      |
 *|                    | current scan  |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_SCAN       | HAL fails     | PASSIVE_UNFOCUSED  | End AF scan      |
 *|                    | current scan  |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_SCAN       | AF_TRIGGER    | FOCUSED_LOCKED     | Immediate trans. |
 *|                    |               |                    | if focus is good |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_SCAN       | AF_TRIGGER    | NOT_FOCUSED_LOCKED | Immediate trans. |
 *|                    |               |                    | if focus is bad  |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_SCAN       | AF_CANCEL     | INACTIVE           | Reset lens       |
 *|                    |               |                    | position         |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_FOCUSED    | HAL initiates | PASSIVE_SCAN       | Start AF scan    |
 *|                    | new scan      |                    | Lens now moving  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_UNFOCUSED  | HAL initiates | PASSIVE_SCAN       | Start AF scan    |
 *|                    | new scan      |                    | Lens now moving  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_FOCUSED    | AF_TRIGGER    | FOCUSED_LOCKED     | Immediate trans. |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_UNFOCUSED  | AF_TRIGGER    | NOT_FOCUSED_LOCKED | Immediate trans. |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| FOCUSED_LOCKED     | AF_TRIGGER    | FOCUSED_LOCKED     | No effect        |
 *+--------------------+---------------+--------------------+------------------+
 *| FOCUSED_LOCKED     | AF_CANCEL     | INACTIVE           | Restart AF scan  |
 *+--------------------+---------------+--------------------+------------------+
 *| NOT_FOCUSED_LOCKED | AF_TRIGGER    | NOT_FOCUSED_LOCKED | No effect        |
 *+--------------------+---------------+--------------------+------------------+
 *| NOT_FOCUSED_LOCKED | AF_CANCEL     | INACTIVE           | Restart AF scan  |
 *+--------------------+---------------+--------------------+------------------+
 *
 *                            mode = AF_MODE_CONTINUOUS_PICTURE
 *| state              | trans. cause  | new state          | notes            |
 *+--------------------+---------------+--------------------+------------------+
 *| INACTIVE           | HAL initiates | PASSIVE_SCAN       | Start AF scan    |
 *|                    | new scan      |                    | Lens now moving  |
 *+--------------------+---------------+--------------------+------------------+
 *| INACTIVE           | AF_TRIGGER    | NOT_FOCUSED_LOCKED | AF state query   |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_SCAN       | HAL completes | PASSIVE_FOCUSED    | End AF scan      |
 *|                    | current scan  |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_SCAN       | HAL fails     | PASSIVE_UNFOCUSED  | End AF scan      |
 *|                    | current scan  |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_SCAN       | AF_TRIGGER    | FOCUSED_LOCKED     | Eventual trans.  |
 *|                    |               |                    | once focus good  |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_SCAN       | AF_TRIGGER    | NOT_FOCUSED_LOCKED | Eventual trans.  |
 *|                    |               |                    | if cannot focus  |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_SCAN       | AF_CANCEL     | INACTIVE           | Reset lens       |
 *|                    |               |                    | position         |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_FOCUSED    | HAL initiates | PASSIVE_SCAN       | Start AF scan    |
 *|                    | new scan      |                    | Lens now moving  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_UNFOCUSED  | HAL initiates | PASSIVE_SCAN       | Start AF scan    |
 *|                    | new scan      |                    | Lens now moving  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_FOCUSED    | AF_TRIGGER    | FOCUSED_LOCKED     | Immediate trans. |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| PASSIVE_UNFOCUSED  | AF_TRIGGER    | NOT_FOCUSED_LOCKED | Immediate trans. |
 *|                    |               |                    | Lens now locked  |
 *+--------------------+---------------+--------------------+------------------+
 *| FOCUSED_LOCKED     | AF_TRIGGER    | FOCUSED_LOCKED     | No effect        |
 *+--------------------+---------------+--------------------+------------------+
 *| FOCUSED_LOCKED     | AF_CANCEL     | INACTIVE           | Restart AF scan  |
 *+--------------------+---------------+--------------------+------------------+
 *| NOT_FOCUSED_LOCKED | AF_TRIGGER    | NOT_FOCUSED_LOCKED | No effect        |
 *+--------------------+---------------+--------------------+------------------+
 *| NOT_FOCUSED_LOCKED | AF_CANCEL     | INACTIVE           | Restart AF scan  |
 *+--------------------+---------------+--------------------+------------------+
 *
 * S4.6. AE and AWB state machines
 *
 *   The AE and AWB state machines are mostly identical. AE has additional
 *   FLASH_REQUIRED and PRECAPTURE states. So rows below that refer to those two
 *   states should be ignored for the AWB state machine.
 *
 *                  when enabling AE/AWB or changing AE/AWB mode
 *| state              | trans. cause  | new state          | notes            |
 *+--------------------+---------------+--------------------+------------------+
 *| Any                |  mode change  | INACTIVE           |                  |
 *+--------------------+---------------+--------------------+------------------+
 *
 *                            mode = AE_MODE_OFF / AWB mode not AUTO
 *| state              | trans. cause  | new state          | notes            |
 *+--------------------+---------------+--------------------+------------------+
 *| INACTIVE           |               | INACTIVE           | AE/AWB disabled  |
 *+--------------------+---------------+--------------------+------------------+
 *
 *                            mode = AE_MODE_ON_* / AWB_MODE_AUTO
 *| state              | trans. cause  | new state          | notes            |
 *+--------------------+---------------+--------------------+------------------+
 *| INACTIVE           | HAL initiates | SEARCHING          |                  |
 *|                    | AE/AWB scan   |                    |                  |
 *+--------------------+---------------+--------------------+------------------+
 *| INACTIVE           | AE/AWB_LOCK   | LOCKED             | values locked    |
 *|                    | on            |                    |                  |
 *+--------------------+---------------+--------------------+------------------+
 *| SEARCHING          | HAL finishes  | CONVERGED          | good values, not |
 *|                    | AE/AWB scan   |                    | changing         |
 *+--------------------+---------------+--------------------+------------------+
 *| SEARCHING          | HAL finishes  | FLASH_REQUIRED     | converged but too|
 *|                    | AE scan       |                    | dark w/o flash   |
 *+--------------------+---------------+--------------------+------------------+
 *| SEARCHING          | AE/AWB_LOCK   | LOCKED             | values locked    |
 *|                    | on            |                    |                  |
 *+--------------------+---------------+--------------------+------------------+
 *| CONVERGED          | HAL initiates | SEARCHING          | values locked    |
 *|                    | AE/AWB scan   |                    |                  |
 *+--------------------+---------------+--------------------+------------------+
 *| CONVERGED          | AE/AWB_LOCK   | LOCKED             | values locked    |
 *|                    | on            |                    |                  |
 *+--------------------+---------------+--------------------+------------------+
 *| FLASH_REQUIRED     | HAL initiates | SEARCHING          | values locked    |
 *|                    | AE/AWB scan   |                    |                  |
 *+--------------------+---------------+--------------------+------------------+
 *| FLASH_REQUIRED     | AE/AWB_LOCK   | LOCKED             | values locked    |
 *|                    | on            |                    |                  |
 *+--------------------+---------------+--------------------+------------------+
 *| LOCKED             | AE/AWB_LOCK   | SEARCHING          | values not good  |
 *|                    | off           |                    | after unlock     |
 *+--------------------+---------------+--------------------+------------------+
 *| LOCKED             | AE/AWB_LOCK   | CONVERGED          | values good      |
 *|                    | off           |                    | after unlock     |
 *+--------------------+---------------+--------------------+------------------+
 *| LOCKED             | AE_LOCK       | FLASH_REQUIRED     | exposure good,   |
 *|                    | off           |                    | but too dark     |
 *+--------------------+---------------+--------------------+------------------+
 *| All AE states      | PRECAPTURE_   | PRECAPTURE         | Start precapture |
 *|                    | START         |                    | sequence         |
 *+--------------------+---------------+--------------------+------------------+
 *| PRECAPTURE         | Sequence done.| CONVERGED          | Ready for high-  |
 *|                    | AE_LOCK off   |                    | quality capture  |
 *+--------------------+---------------+--------------------+------------------+
 *| PRECAPTURE         | Sequence done.| LOCKED             | Ready for high-  |
 *|                    | AE_LOCK on    |                    | quality capture  |
 *+--------------------+---------------+--------------------+------------------+
 *
 */

/**
 * S5. Cropping:
 *
 * Cropping of the full pixel array (for digital zoom and other use cases where
 * a smaller FOV is desirable) is communicated through the
 * ANDROID_SCALER_CROP_REGION setting. This is a per-request setting, and can
 * change on a per-request basis, which is critical for implementing smooth
 * digital zoom.
 *
 * The region is defined as a rectangle (x, y, width, height), with (x, y)
 * describing the top-left corner of the rectangle. The rectangle is defined on
 * the coordinate system of the sensor active pixel array, with (0,0) being the
 * top-left pixel of the active pixel array. Therefore, the width and height
 * cannot be larger than the dimensions reported in the
 * ANDROID_SENSOR_ACTIVE_PIXEL_ARRAY static info field. The minimum allowed
 * width and height are reported by the HAL through the
 * ANDROID_SCALER_MAX_DIGITAL_ZOOM static info field, which describes the
 * maximum supported zoom factor. Therefore, the minimum crop region width and
 * height are:
 *
 * {width, height} =
 *    { floor(ANDROID_SENSOR_ACTIVE_PIXEL_ARRAY[0] /
 *        ANDROID_SCALER_MAX_DIGITAL_ZOOM),
 *      floor(ANDROID_SENSOR_ACTIVE_PIXEL_ARRAY[1] /
 *        ANDROID_SCALER_MAX_DIGITAL_ZOOM) }
 *
 * If the crop region needs to fulfill specific requirements (for example, it
 * needs to start on even coordinates, and its width/height needs to be even),
 * the HAL must do the necessary rounding and write out the final crop region
 * used in the output result metadata. Similarly, if the HAL implements video
 * stabilization, it must adjust the result crop region to describe the region
 * actually included in the output after video stabilization is applied. In
 * general, a camera-using application must be able to determine the field of
 * view it is receiving based on the crop region, the dimensions of the image
 * sensor, and the lens focal length.
 *
 * It is assumed that the cropping is applied after raw to other color space
 * conversion. Raw streams (RAW16 and RAW_OPAQUE) don't have this conversion stage,
 * and are not croppable. Therefore, the crop region must be ignored by the HAL
 * for raw streams.
 *
 * Since the crop region applies to all non-raw streams, which may have different aspect
 * ratios than the crop region, the exact sensor region used for each stream may
 * be smaller than the crop region. Specifically, each stream should maintain
 * square pixels and its aspect ratio by minimally further cropping the defined
 * crop region. If the stream's aspect ratio is wider than the crop region, the
 * stream should be further cropped vertically, and if the stream's aspect ratio
 * is narrower than the crop region, the stream should be further cropped
 * horizontally.
 *
 * In all cases, the stream crop must be centered within the full crop region,
 * and each stream is only either cropped horizontally or vertical relative to
 * the full crop region, never both.
 *
 * For example, if two streams are defined, a 640x480 stream (4:3 aspect), and a
 * 1280x720 stream (16:9 aspect), below demonstrates the expected output regions
 * for each stream for a few sample crop regions, on a hypothetical 3 MP (2000 x
 * 1500 pixel array) sensor.
 *
 * Crop region: (500, 375, 1000, 750) (4:3 aspect ratio)
 *
 *   640x480 stream crop: (500, 375, 1000, 750) (equal to crop region)
 *   1280x720 stream crop: (500, 469, 1000, 562) (marked with =)
 *
 * 0                   1000               2000
 * +---------+---------+---------+----------+
 * | Active pixel array                     |
 * |                                        |
 * |                                        |
 * +         +-------------------+          + 375
 * |         |                   |          |
 * |         O===================O          |
 * |         I 1280x720 stream   I          |
 * +         I                   I          + 750
 * |         I                   I          |
 * |         O===================O          |
 * |         |                   |          |
 * +         +-------------------+          + 1125
 * |          Crop region, 640x480 stream   |
 * |                                        |
 * |                                        |
 * +---------+---------+---------+----------+ 1500
 *
 * Crop region: (500, 375, 1333, 750) (16:9 aspect ratio)
 *
 *   640x480 stream crop: (666, 375, 1000, 750) (marked with =)
 *   1280x720 stream crop: (500, 375, 1333, 750) (equal to crop region)
 *
 * 0                   1000               2000
 * +---------+---------+---------+----------+
 * | Active pixel array                     |
 * |                                        |
 * |                                        |
 * +         +---O==================O---+   + 375
 * |         |   I 640x480 stream   I   |   |
 * |         |   I                  I   |   |
 * |         |   I                  I   |   |
 * +         |   I                  I   |   + 750
 * |         |   I                  I   |   |
 * |         |   I                  I   |   |
 * |         |   I                  I   |   |
 * +         +---O==================O---+   + 1125
 * |          Crop region, 1280x720 stream  |
 * |                                        |
 * |                                        |
 * +---------+---------+---------+----------+ 1500
 *
 * Crop region: (500, 375, 750, 750) (1:1 aspect ratio)
 *
 *   640x480 stream crop: (500, 469, 750, 562) (marked with =)
 *   1280x720 stream crop: (500, 543, 750, 414) (marged with #)
 *
 * 0                   1000               2000
 * +---------+---------+---------+----------+
 * | Active pixel array                     |
 * |                                        |
 * |                                        |
 * +         +--------------+               + 375
 * |         O==============O               |
 * |         ################               |
 * |         #              #               |
 * +         #              #               + 750
 * |         #              #               |
 * |         ################ 1280x720      |
 * |         O==============O 640x480       |
 * +         +--------------+               + 1125
 * |          Crop region                   |
 * |                                        |
 * |                                        |
 * +---------+---------+---------+----------+ 1500
 *
 * And a final example, a 1024x1024 square aspect ratio stream instead of the
 * 480p stream:
 *
 * Crop region: (500, 375, 1000, 750) (4:3 aspect ratio)
 *
 *   1024x1024 stream crop: (625, 375, 750, 750) (marked with #)
 *   1280x720 stream crop: (500, 469, 1000, 562) (marked with =)
 *
 * 0                   1000               2000
 * +---------+---------+---------+----------+
 * | Active pixel array                     |
 * |                                        |
 * |              1024x1024 stream          |
 * +         +--###############--+          + 375
 * |         |  #             #  |          |
 * |         O===================O          |
 * |         I 1280x720 stream   I          |
 * +         I                   I          + 750
 * |         I                   I          |
 * |         O===================O          |
 * |         |  #             #  |          |
 * +         +--###############--+          + 1125
 * |          Crop region                   |
 * |                                        |
 * |                                        |
 * +---------+---------+---------+----------+ 1500
 *
 */

/**
 * S6. Error management:
 *
 * Camera HAL device ops functions that have a return value will all return
 * -ENODEV / NULL in case of a serious error. This means the device cannot
 * continue operation, and must be closed by the framework. Once this error is
 * returned by some method, or if notify() is called with ERROR_DEVICE, only
 * the close() method can be called successfully. All other methods will return
 * -ENODEV / NULL.
 *
 * If a device op is called in the wrong sequence, for example if the framework
 * calls configure_streams() is called before initialize(), the device must
 * return -ENOSYS from the call, and do nothing.
 *
 * Transient errors in image capture must be reported through notify() as follows:
 *
 * - The failure of an entire capture to occur must be reported by the HAL by
 *   calling notify() with ERROR_REQUEST. Individual errors for the result
 *   metadata or the output buffers must not be reported in this case.
 *
 * - If the metadata for a capture cannot be produced, but some image buffers
 *   were filled, the HAL must call notify() with ERROR_RESULT.
 *
 * - If an output image buffer could not be filled, but either the metadata was
 *   produced or some other buffers were filled, the HAL must call notify() with
 *   ERROR_BUFFER for each failed buffer.
 *
 * In each of these transient failure cases, the HAL must still call
 * process_capture_result, with valid output and input (if an input buffer was
 * submitted) buffer_handle_t. If the result metadata could not be produced, it
 * should be NULL. If some buffers could not be filled, they must be returned with
 * process_capture_result in the error state, their release fences must be set to
 * the acquire fences passed by the framework, or -1 if they have been waited on by
 * the HAL already.
 *
 * Invalid input arguments result in -EINVAL from the appropriate methods. In
 * that case, the framework must act as if that call had never been made.
 *
 */

/**
 * S7. Key Performance Indicator (KPI) glossary:
 *
 * This includes some critical definitions that are used by KPI metrics.
 *
 * Pipeline Latency:
 *  For a given capture request, the duration from the framework calling
 *  process_capture_request to the HAL sending capture result and all buffers
 *  back by process_capture_result call. To make the Pipeline Latency measure
 *  independent of frame rate, it is measured by frame count.
 *
 *  For example, when frame rate is 30 (fps), the frame duration (time interval
 *  between adjacent frame capture time) is 33 (ms).
 *  If it takes 5 frames for framework to get the result and buffers back for
 *  a given request, then the Pipeline Latency is 5 (frames), instead of
 *  5 x 33 = 165 (ms).
 *
 *  The Pipeline Latency is determined by android.request.pipelineDepth and
 *  android.request.pipelineMaxDepth, see their definitions for more details.
 *
 */

/**
 * S8. Sample Use Cases:
 *
 * This includes some typical use case examples the camera HAL may support.
 *
 * S8.1 Zero Shutter Lag (ZSL) with CAMERA3_STREAM_BIDIRECTIONAL stream.
 *
 *   For this use case, the bidirectional stream will be used by the framework as follows:
 *
 *   1. The framework includes a buffer from this stream as output buffer in a
 *      request as normal.
 *
 *   2. Once the HAL device returns a filled output buffer to the framework,
 *      the framework may do one of two things with the filled buffer:
 *
 *   2. a. The framework uses the filled data, and returns the now-used buffer
 *         to the stream queue for reuse. This behavior exactly matches the
 *         OUTPUT type of stream.
 *
 *   2. b. The framework wants to reprocess the filled data, and uses the
 *         buffer as an input buffer for a request. Once the HAL device has
 *         used the reprocessing buffer, it then returns it to the
 *         framework. The framework then returns the now-used buffer to the
 *         stream queue for reuse.
 *
 *   3. The HAL device will be given the buffer again as an output buffer for
 *        a request at some future point.
 *
 *   For ZSL use case, the pixel format for bidirectional stream will be
 *   HAL_PIXEL_FORMAT_RAW_OPAQUE or HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED if it
 *   is listed in android.scaler.availableInputOutputFormatsMap. When
 *   HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED is used, the gralloc
 *   usage flags for the consumer endpoint will be set to GRALLOC_USAGE_HW_CAMERA_ZSL.
 *   A configuration stream list that has BIDIRECTIONAL stream used as input, will
 *   usually also have a distinct OUTPUT stream to get the reprocessing data. For example,
 *   for the ZSL use case, the stream list might be configured with the following:
 *
 *     - A HAL_PIXEL_FORMAT_RAW_OPAQUE bidirectional stream is used
 *       as input.
 *     - And a HAL_PIXEL_FORMAT_BLOB (JPEG) output stream.
 *
 * S8.2 ZSL (OPAQUE) reprocessing with CAMERA3_STREAM_INPUT stream.
 *
 * CAMERA_DEVICE_API_VERSION_3_3:
 *   When OPAQUE_REPROCESSING capability is supported by the camera device, the INPUT stream
 *   can be used for application/framework implemented use case like Zero Shutter Lag (ZSL).
 *   This kind of stream will be used by the framework as follows:
 *
 *   1. Application/framework configures an opaque (RAW or YUV based) format output stream that is
 *      used to produce the ZSL output buffers. The stream pixel format will be
 *      HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED.
 *
 *   2. Application/framework configures an opaque format input stream that is used to
 *      send the reprocessing ZSL buffers to the HAL. The stream pixel format will
 *      also be HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED.
 *
 *   3. Application/framework configures a YUV/JPEG output stream that is used to receive the
 *      reprocessed data. The stream pixel format will be YCbCr_420/HAL_PIXEL_FORMAT_BLOB.
 *
 *   4. Application/framework picks a ZSL buffer from the ZSL output stream when a ZSL capture is
 *      issued by the application, and sends the data back as an input buffer in a
 *      reprocessing request, then sends to the HAL for reprocessing.
 *
 *   5. The HAL sends back the output YUV/JPEG result to framework.
 *
 *   The HAL can select the actual opaque buffer format and configure the ISP pipeline
 *   appropriately based on the HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED format and
 *   the gralloc usage flag GRALLOC_USAGE_HW_CAMERA_ZSL.

 * S8.3 YUV reprocessing with CAMERA3_STREAM_INPUT stream.
 *
 *   When YUV reprocessing is supported by the HAL, the INPUT stream
 *   can be used for the YUV reprocessing use cases like lucky-shot and image fusion.
 *   This kind of stream will be used by the framework as follows:
 *
 *   1. Application/framework configures an YCbCr_420 format output stream that is
 *      used to produce the output buffers.
 *
 *   2. Application/framework configures an YCbCr_420 format input stream that is used to
 *      send the reprocessing YUV buffers to the HAL.
 *
 *   3. Application/framework configures a YUV/JPEG output stream that is used to receive the
 *      reprocessed data. The stream pixel format will be YCbCr_420/HAL_PIXEL_FORMAT_BLOB.
 *
 *   4. Application/framework processes the output buffers (could be as simple as picking
 *      an output buffer directly) from the output stream when a capture is issued, and sends
 *      the data back as an input buffer in a reprocessing request, then sends to the HAL
 *      for reprocessing.
 *
 *   5. The HAL sends back the output YUV/JPEG result to framework.
 *
 */

/**
 *   S9. Notes on Controls and Metadata
 *
 *   This section contains notes about the interpretation and usage of various metadata tags.
 *
 *   S9.1 HIGH_QUALITY and FAST modes.
 *
 *   Many camera post-processing blocks may be listed as having HIGH_QUALITY,
 *   FAST, and OFF operating modes. These blocks will typically also have an
 *   'available modes' tag representing which of these operating modes are
 *   available on a given device. The general policy regarding implementing
 *   these modes is as follows:
 *
 *   1. Operating mode controls of hardware blocks that cannot be disabled
 *      must not list OFF in their corresponding 'available modes' tags.
 *
 *   2. OFF will always be included in their corresponding 'available modes'
 *      tag if it is possible to disable that hardware block.
 *
 *   3. FAST must always be included in the 'available modes' tags for all
 *      post-processing blocks supported on the device.  If a post-processing
 *      block also has a slower and higher quality operating mode that does
 *      not meet the framerate requirements for FAST mode, HIGH_QUALITY should
 *      be included in the 'available modes' tag to represent this operating
 *      mode.
 */

/**
 *   S10. Reprocessing flow and controls
 *
 *   This section describes the OPAQUE and YUV reprocessing flow and controls. OPAQUE reprocessing
 *   uses an opaque format that is not directly application-visible, and the application can
 *   only select some of the output buffers and send back to HAL for reprocessing, while YUV
 *   reprocessing gives the application opportunity to process the buffers before reprocessing.
 *
 *   S8 gives the stream configurations for the typical reprocessing uses cases,
 *   this section specifies the buffer flow and controls in more details.
 *
 *   S10.1 OPAQUE (typically for ZSL use case) reprocessing flow and controls
 *
 *   For OPAQUE reprocessing (e.g. ZSL) use case, after the application creates the specific
 *   output and input streams, runtime buffer flow and controls are specified as below:
 *
 *   1. Application starts output streaming by sending repeating requests for output
 *      opaque buffers and preview. The buffers are held by an application
 *      maintained circular buffer. The requests are based on CAMERA3_TEMPLATE_ZERO_SHUTTER_LAG
 *      capture template, which should have all necessary settings that guarantee output
 *      frame rate is not slowed down relative to sensor output frame rate.
 *
 *   2. When a capture is issued, the application selects one output buffer based
 *      on application buffer selection logic, e.g. good AE and AF statistics etc.
 *      Application then creates an reprocess request based on the capture result associated
 *      with this selected buffer. The selected output buffer is now added to this reprocess
 *      request as an input buffer, the output buffer of this reprocess request should be
 *      either JPEG output buffer or YUV output buffer, or both, depending on the application
 *      choice.
 *
 *   3. Application then alters the reprocess settings to get best image quality. The HAL must
 *      support and only support below controls if the HAL support OPAQUE_REPROCESSING capability:
 *          - android.jpeg.* (if JPEG buffer is included as one of the output)
 *          - android.noiseReduction.mode (change to HIGH_QUALITY if it is supported)
 *          - android.edge.mode (change to HIGH_QUALITY if it is supported)
 *       All other controls must be ignored by the HAL.
 *   4. HAL processed the input buffer and return the output buffers in the capture results
 *      as normal.
 *
 *   S10.2 YUV reprocessing flow and controls
 *
 *   The YUV reprocessing buffer flow is similar as OPAQUE reprocessing, with below difference:
 *
 *   1. Application may want to have finer granularity control of the intermediate YUV images
 *      (before reprocessing). For example, application may choose
 *          - android.noiseReduction.mode == MINIMAL
 *      to make sure the no YUV domain noise reduction has applied to the output YUV buffers,
 *      then it can do its own advanced noise reduction on them. For OPAQUE reprocessing case, this
 *      doesn't matter, as long as the final reprocessed image has the best quality.
 *   2. Application may modify the YUV output buffer data. For example, for image fusion use
 *      case, where multiple output images are merged together to improve the signal-to-noise
 *      ratio (SNR). The input buffer may be generated from multiple buffers by the application.
 *      To avoid excessive amount of noise reduction and insufficient amount of edge enhancement
 *      being applied to the input buffer, the application can hint the HAL  how much effective
 *      exposure time improvement has been done by the application, then the HAL can adjust the
 *      noise reduction and edge enhancement paramters to get best reprocessed image quality.
 *      Below tag can be used for this purpose:
 *          - android.reprocess.effectiveExposureFactor
 *      The value would be exposure time increase factor applied to the original output image,
 *      for example, if there are N image merged, the exposure time increase factor would be up
 *      to sqrt(N). See this tag spec for more details.
 *
 *   S10.3 Reprocessing pipeline characteristics
 *
 *   Reprocessing pipeline has below different characteristics comparing with normal output
 *   pipeline:
 *
 *   1. The reprocessing result can be returned ahead of the pending normal output results. But
 *      the FIFO ordering must be maintained for all reprocessing results. For example, there are
 *      below requests (A stands for output requests, B stands for reprocessing requests)
 *      being processed by the HAL:
 *          A1, A2, A3, A4, B1, A5, B2, A6...
 *      result of B1 can be returned before A1-A4, but result of B2 must be returned after B1.
 *   2. Single input rule: For a given reprocessing request, all output buffers must be from the
 *      input buffer, rather than sensor output. For example, if a reprocess request include both
 *      JPEG and preview buffers, all output buffers must be produced from the input buffer
 *      included by the reprocessing request, rather than sensor. The HAL must not output preview
 *      buffers from sensor, while output JPEG buffer from the input buffer.
 *   3. Input buffer will be from camera output directly (ZSL case) or indirectly(image fusion
 *      case). For the case where buffer is modified, the size will remain same. The HAL can
 *      notify CAMERA3_MSG_ERROR_REQUEST if buffer from unknown source is sent.
 *   4. Result as reprocessing request: The HAL can expect that a reprocessing request is a copy
 *      of one of the output results with minor allowed setting changes. The HAL can notify
 *      CAMERA3_MSG_ERROR_REQUEST if a request from unknown source is issued.
 *   5. Output buffers may not be used as inputs across the configure stream boundary, This is
 *      because an opaque stream like the ZSL output stream may have different actual image size
 *      inside of the ZSL buffer to save power and bandwidth for smaller resolution JPEG capture.
 *      The HAL may notify CAMERA3_MSG_ERROR_REQUEST if this case occurs.
 *   6. HAL Reprocess requests error reporting during flush should follow the same rule specified
 *      by flush() method.
 *
 */

__BEGIN_DECLS

struct camera3_device;

/**********************************************************************
 *
 * Camera3 stream and stream buffer definitions.
 *
 * These structs and enums define the handles and contents of the input and
 * output streams connecting the HAL to various framework and application buffer
 * consumers. Each stream is backed by a gralloc buffer queue.
 *
 */

/**
 * camera3_stream_type_t:
 *
 * The type of the camera stream, which defines whether the camera HAL device is
 * the producer or the consumer for that stream, and how the buffers of the
 * stream relate to the other streams.
 */
typedef enum camera3_stream_type {
    /**
     * This stream is an output stream; the camera HAL device will be
     * responsible for filling buffers from this stream with newly captured or
     * reprocessed image data.
     */
    CAMERA3_STREAM_OUTPUT = 0,

    /**
     * This stream is an input stream; the camera HAL device will be responsible
     * for reading buffers from this stream and sending them through the camera
     * processing pipeline, as if the buffer was a newly captured image from the
     * imager.
     *
     * The pixel format for input stream can be any format reported by
     * android.scaler.availableInputOutputFormatsMap. The pixel format of the
     * output stream that is used to produce the reprocessing data may be any
     * format reported by android.scaler.availableStreamConfigurations. The
     * supported input/output stream combinations depends the camera device
     * capabilities, see android.scaler.availableInputOutputFormatsMap for
     * stream map details.
     *
     * This kind of stream is generally used to reprocess data into higher
     * quality images (that otherwise would cause a frame rate performance
     * loss), or to do off-line reprocessing.
     *
     * CAMERA_DEVICE_API_VERSION_3_3:
     *    The typical use cases are OPAQUE (typically ZSL) and YUV reprocessing,
     *    see S8.2, S8.3 and S10 for more details.
     */
    CAMERA3_STREAM_INPUT = 1,

    /**
     * This stream can be used for input and output. Typically, the stream is
     * used as an output stream, but occasionally one already-filled buffer may
     * be sent back to the HAL device for reprocessing.
     *
     * This kind of stream is meant generally for Zero Shutter Lag (ZSL)
     * features, where copying the captured image from the output buffer to the
     * reprocessing input buffer would be expensive. See S8.1 for more details.
     *
     * Note that the HAL will always be reprocessing data it produced.
     *
     */
    CAMERA3_STREAM_BIDIRECTIONAL = 2,

    /**
     * Total number of framework-defined stream types
     */
    CAMERA3_NUM_STREAM_TYPES

} camera3_stream_type_t;

/**
 * camera3_stream_rotation_t:
 *
 * The required counterclockwise rotation of camera stream.
 */
typedef enum camera3_stream_rotation {
    /* No rotation */
    CAMERA3_STREAM_ROTATION_0 = 0,

    /* Rotate by 90 degree counterclockwise */
    CAMERA3_STREAM_ROTATION_90 = 1,

    /* Rotate by 180 degree counterclockwise */
    CAMERA3_STREAM_ROTATION_180 = 2,

    /* Rotate by 270 degree counterclockwise */
    CAMERA3_STREAM_ROTATION_270 = 3
} camera3_stream_rotation_t;

/**
 * camera3_stream_configuration_mode_t:
 *
 * This defines the general operation mode for the HAL (for a given stream configuration), where
 * modes besides NORMAL have different semantics, and usually limit the generality of the API in
 * exchange for higher performance in some particular area.
 */
typedef enum camera3_stream_configuration_mode {
    /**
     * Normal stream configuration operation mode. This is the default camera operation mode,
     * where all semantics of HAL APIs and metadata controls apply.
     */
    CAMERA3_STREAM_CONFIGURATION_NORMAL_MODE = 0,

    /**
     * Special constrained high speed operation mode for devices that can not support high
     * speed output in NORMAL mode. All streams in this configuration are operating at high speed
     * mode and have different characteristics and limitations to achieve high speed output.
     * The NORMAL mode can still be used for high speed output if the HAL can support high speed
     * output while satisfying all the semantics of HAL APIs and metadata controls. It is
     * recommended for the HAL to support high speed output in NORMAL mode (by advertising the high
     * speed FPS ranges in android.control.aeAvailableTargetFpsRanges) if possible.
     *
     * This mode has below limitations/requirements:
     *
     *   1. The HAL must support up to 2 streams with sizes reported by
     *      android.control.availableHighSpeedVideoConfigurations.
     *   2. In this mode, the HAL is expected to output up to 120fps or higher. This mode must
     *      support the targeted FPS range and size configurations reported by
     *      android.control.availableHighSpeedVideoConfigurations.
     *   3. The HAL must support HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED output stream format.
     *   4. To achieve efficient high speed streaming, the HAL may have to aggregate
     *      multiple frames together and send to camera device for processing where the request
     *      controls are same for all the frames in this batch (batch mode). The HAL must support
     *      max batch size and the max batch size requirements defined by
     *      android.control.availableHighSpeedVideoConfigurations.
     *   5. In this mode, the HAL must override aeMode, awbMode, and afMode to ON, ON, and
     *      CONTINUOUS_VIDEO, respectively. All post-processing block mode controls must be
     *      overridden to be FAST. Therefore, no manual control of capture and post-processing
     *      parameters is possible. All other controls operate the same as when
     *      android.control.mode == AUTO. This means that all other android.control.* fields
     *      must continue to work, such as
     *
     *      android.control.aeTargetFpsRange
     *      android.control.aeExposureCompensation
     *      android.control.aeLock
     *      android.control.awbLock
     *      android.control.effectMode
     *      android.control.aeRegions
     *      android.control.afRegions
     *      android.control.awbRegions
     *      android.control.afTrigger
     *      android.control.aePrecaptureTrigger
     *
     *      Outside of android.control.*, the following controls must work:
     *
     *      android.flash.mode (TORCH mode only, automatic flash for still capture will not work
     *      since aeMode is ON)
     *      android.lens.opticalStabilizationMode (if it is supported)
     *      android.scaler.cropRegion
     *      android.statistics.faceDetectMode (if it is supported)
     *
     * For more details about high speed stream requirements, see
     * android.control.availableHighSpeedVideoConfigurations and CONSTRAINED_HIGH_SPEED_VIDEO
     * capability defined in android.request.availableCapabilities.
     *
     * This mode only needs to be supported by HALs that include CONSTRAINED_HIGH_SPEED_VIDEO in
     * the android.request.availableCapabilities static metadata.
     */
    CAMERA3_STREAM_CONFIGURATION_CONSTRAINED_HIGH_SPEED_MODE = 1,

    /**
     * First value for vendor-defined stream configuration modes.
     */
    CAMERA3_VENDOR_STREAM_CONFIGURATION_MODE_START = 0x8000
} camera3_stream_configuration_mode_t;

/**
 * camera3_stream_t:
 *
 * A handle to a single camera input or output stream. A stream is defined by
 * the framework by its buffer resolution and format, and additionally by the
 * HAL with the gralloc usage flags and the maximum in-flight buffer count.
 *
 * The stream structures are owned by the framework, but pointers to a
 * camera3_stream passed into the HAL by configure_streams() are valid until the
 * end of the first subsequent configure_streams() call that _does not_ include
 * that camera3_stream as an argument, or until the end of the close() call.
 *
 * All camera3_stream framework-controlled members are immutable once the
 * camera3_stream is passed into configure_streams().  The HAL may only change
 * the HAL-controlled parameters during a configure_streams() call, except for
 * the contents of the private pointer.
 *
 * If a configure_streams() call returns a non-fatal error, all active streams
 * remain valid as if configure_streams() had not been called.
 *
 * The endpoint of the stream is not visible to the camera HAL device.
 * In DEVICE_API_VERSION_3_1, this was changed to share consumer usage flags
 * on streams where the camera is a producer (OUTPUT and BIDIRECTIONAL stream
 * types) see the usage field below.
 */
typedef struct camera3_stream {

    /*****
     * Set by framework before configure_streams()
     */

    /**
     * The type of the stream, one of the camera3_stream_type_t values.
     */
    int stream_type;

    /**
     * The width in pixels of the buffers in this stream
     */
    uint32_t width;

    /**
     * The height in pixels of the buffers in this stream
     */
    uint32_t height;

    /**
     * The pixel format for the buffers in this stream. Format is a value from
     * the HAL_PIXEL_FORMAT_* list in system/core/include/system/graphics.h, or
     * from device-specific headers.
     *
     * If HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED is used, then the platform
     * gralloc module will select a format based on the usage flags provided by
     * the camera device and the other endpoint of the stream.
     *
     * <= CAMERA_DEVICE_API_VERSION_3_1:
     *
     * The camera HAL device must inspect the buffers handed to it in the
     * subsequent register_stream_buffers() call to obtain the
     * implementation-specific format details, if necessary.
     *
     * >= CAMERA_DEVICE_API_VERSION_3_2:
     *
     * register_stream_buffers() won't be called by the framework, so the HAL
     * should configure the ISP and sensor pipeline based purely on the sizes,
     * usage flags, and formats for the configured streams.
     */
    int format;

    /*****
     * Set by HAL during configure_streams().
     */

    /**
     * The gralloc usage flags for this stream, as needed by the HAL. The usage
     * flags are defined in gralloc.h (GRALLOC_USAGE_*), or in device-specific
     * headers.
     *
     * For output streams, these are the HAL's producer usage flags. For input
     * streams, these are the HAL's consumer usage flags. The usage flags from
     * the producer and the consumer will be combined together and then passed
     * to the platform gralloc HAL module for allocating the gralloc buffers for
     * each stream.
     *
     * Version information:
     *
     * == CAMERA_DEVICE_API_VERSION_3_0:
     *
     *   No initial value guaranteed when passed via configure_streams().
     *   HAL may not use this field as input, and must write over this field
     *   with its usage flags.
     *
     * >= CAMERA_DEVICE_API_VERSION_3_1:
     *
     *   For stream_type OUTPUT and BIDIRECTIONAL, when passed via
     *   configure_streams(), the initial value of this is the consumer's
     *   usage flags.  The HAL may use these consumer flags to decide stream
     *   configuration.
     *   For stream_type INPUT, when passed via configure_streams(), the initial
     *   value of this is 0.
     *   For all streams passed via configure_streams(), the HAL must write
     *   over this field with its usage flags.
     */
    uint32_t usage;

    /**
     * The maximum number of buffers the HAL device may need to have dequeued at
     * the same time. The HAL device may not have more buffers in-flight from
     * this stream than this value.
     */
    uint32_t max_buffers;

    /**
     * A handle to HAL-private information for the stream. Will not be inspected
     * by the framework code.
     */
    void *priv;

    /**
     * A field that describes the contents of the buffer. The format and buffer
     * dimensions define the memory layout and structure of the stream buffers,
     * while dataSpace defines the meaning of the data within the buffer.
     *
     * For most formats, dataSpace defines the color space of the image data.
     * In addition, for some formats, dataSpace indicates whether image- or
     * depth-based data is requested.  See system/core/include/system/graphics.h
     * for details of formats and valid dataSpace values for each format.
     *
     * Version information:
     *
     * < CAMERA_DEVICE_API_VERSION_3_3:
     *
     *   Not defined and should not be accessed. dataSpace should be assumed to
     *   be HAL_DATASPACE_UNKNOWN, and the appropriate color space, etc, should
     *   be determined from the usage flags and the format.
     *
     * >= CAMERA_DEVICE_API_VERSION_3_3:
     *
     *   Always set by the camera service. HAL must use this dataSpace to
     *   configure the stream to the correct colorspace, or to select between
     *   color and depth outputs if supported.
     */
    android_dataspace_t data_space;

    /**
     * The required output rotation of the stream, one of
     * the camera3_stream_rotation_t values. This must be inspected by HAL along
     * with stream width and height. For example, if the rotation is 90 degree
     * and the stream width and height is 720 and 1280 respectively, camera service
     * will supply buffers of size 720x1280, and HAL should capture a 1280x720 image
     * and rotate the image by 90 degree counterclockwise. The rotation field is
     * no-op when the stream type is input. Camera HAL must ignore the rotation
     * field for an input stream.
     *
     * <= CAMERA_DEVICE_API_VERSION_3_2:
     *
     *    Not defined and must not be accessed. HAL must not apply any rotation
     *    on output images.
     *
     * >= CAMERA_DEVICE_API_VERSION_3_3:
     *
     *    Always set by camera service. HAL must inspect this field during stream
     *    configuration and returns -EINVAL if HAL cannot perform such rotation.
     *    HAL must always support CAMERA3_STREAM_ROTATION_0, so a
     *    configure_streams() call must not fail for unsupported rotation if
     *    rotation field of all streams is CAMERA3_STREAM_ROTATION_0.
     *
     */
    int rotation;

    /* reserved for future use */
    void *reserved[7];

} camera3_stream_t;

/**
 * camera3_stream_configuration_t:
 *
 * A structure of stream definitions, used by configure_streams(). This
 * structure defines all the output streams and the reprocessing input
 * stream for the current camera use case.
 */
typedef struct camera3_stream_configuration {
    /**
     * The total number of streams requested by the framework.  This includes
     * both input and output streams. The number of streams will be at least 1,
     * and there will be at least one output-capable stream.
     */
    uint32_t num_streams;

    /**
     * An array of camera stream pointers, defining the input/output
     * configuration for the camera HAL device.
     *
     * At most one input-capable stream may be defined (INPUT or BIDIRECTIONAL)
     * in a single configuration.
     *
     * At least one output-capable stream must be defined (OUTPUT or
     * BIDIRECTIONAL).
     */
    camera3_stream_t **streams;

    /**
     * >= CAMERA_DEVICE_API_VERSION_3_3:
     *
     * The operation mode of streams in this configuration, one of the value defined in
     * camera3_stream_configuration_mode_t.
     * The HAL can use this mode as an indicator to set the stream property (e.g.,
     * camera3_stream->max_buffers) appropriately. For example, if the configuration is
     * CAMERA3_STREAM_CONFIGURATION_CONSTRAINED_HIGH_SPEED_MODE, the HAL may want to set aside more
     * buffers for batch mode operation (see android.control.availableHighSpeedVideoConfigurations
     * for batch mode definition).
     *
     */
    uint32_t operation_mode;
} camera3_stream_configuration_t;

/**
 * camera3_buffer_status_t:
 *
 * The current status of a single stream buffer.
 */
typedef enum camera3_buffer_status {
    /**
     * The buffer is in a normal state, and can be used after waiting on its
     * sync fence.
     */
    CAMERA3_BUFFER_STATUS_OK = 0,

    /**
     * The buffer does not contain valid data, and the data in it should not be
     * used. The sync fence must still be waited on before reusing the buffer.
     */
    CAMERA3_BUFFER_STATUS_ERROR = 1

} camera3_buffer_status_t;

/**
 * camera3_stream_buffer_t:
 *
 * A single buffer from a camera3 stream. It includes a handle to its parent
 * stream, the handle to the gralloc buffer itself, and sync fences
 *
 * The buffer does not specify whether it is to be used for input or output;
 * that is determined by its parent stream type and how the buffer is passed to
 * the HAL device.
 */
typedef struct camera3_stream_buffer {
    /**
     * The handle of the stream this buffer is associated with
     */
    camera3_stream_t *stream;

    /**
     * The native handle to the buffer
     */
    buffer_handle_t *buffer;

    /**
     * Current state of the buffer, one of the camera3_buffer_status_t
     * values. The framework will not pass buffers to the HAL that are in an
     * error state. In case a buffer could not be filled by the HAL, it must
     * have its status set to CAMERA3_BUFFER_STATUS_ERROR when returned to the
     * framework with process_capture_result().
     */
    int status;

    /**
     * The acquire sync fence for this buffer. The HAL must wait on this fence
     * fd before attempting to read from or write to this buffer.
     *
     * The framework may be set to -1 to indicate that no waiting is necessary
     * for this buffer.
     *
     * When the HAL returns an output buffer to the framework with
     * process_capture_result(), the acquire_fence must be set to -1. If the HAL
     * never waits on the acquire_fence due to an error in filling a buffer,
     * when calling process_capture_result() the HAL must set the release_fence
     * of the buffer to be the acquire_fence passed to it by the framework. This
     * will allow the framework to wait on the fence before reusing the buffer.
     *
     * For input buffers, the HAL must not change the acquire_fence field during
     * the process_capture_request() call.
     *
     * >= CAMERA_DEVICE_API_VERSION_3_2:
     *
     * When the HAL returns an input buffer to the framework with
     * process_capture_result(), the acquire_fence must be set to -1. If the HAL
     * never waits on input buffer acquire fence due to an error, the sync
     * fences should be handled similarly to the way they are handled for output
     * buffers.
     */
     int acquire_fence;

    /**
     * The release sync fence for this buffer. The HAL must set this fence when
     * returning buffers to the framework, or write -1 to indicate that no
     * waiting is required for this buffer.
     *
     * For the output buffers, the fences must be set in the output_buffers
     * array passed to process_capture_result().
     *
     * <= CAMERA_DEVICE_API_VERSION_3_1:
     *
     * For the input buffer, the release fence must be set by the
     * process_capture_request() call.
     *
     * >= CAMERA_DEVICE_API_VERSION_3_2:
     *
     * For the input buffer, the fences must be set in the input_buffer
     * passed to process_capture_result().
     *
     * After signaling the release_fence for this buffer, the HAL
     * should not make any further attempts to access this buffer as the
     * ownership has been fully transferred back to the framework.
     *
     * If a fence of -1 was specified then the ownership of this buffer
     * is transferred back immediately upon the call of process_capture_result.
     */
    int release_fence;

} camera3_stream_buffer_t;

/**
 * camera3_stream_buffer_set_t:
 *
 * The complete set of gralloc buffers for a stream. This structure is given to
 * register_stream_buffers() to allow the camera HAL device to register/map/etc
 * newly allocated stream buffers.
 *
 * >= CAMERA_DEVICE_API_VERSION_3_2:
 *
 * Deprecated (and not used). In particular,
 * register_stream_buffers is also deprecated and will never be invoked.
 *
 */
typedef struct camera3_stream_buffer_set {
    /**
     * The stream handle for the stream these buffers belong to
     */
    camera3_stream_t *stream;

    /**
     * The number of buffers in this stream. It is guaranteed to be at least
     * stream->max_buffers.
     */
    uint32_t num_buffers;

    /**
     * The array of gralloc buffer handles for this stream. If the stream format
     * is set to HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED, the camera HAL device
     * should inspect the passed-in buffers to determine any platform-private
     * pixel format information.
     */
    buffer_handle_t **buffers;

} camera3_stream_buffer_set_t;

/**
 * camera3_jpeg_blob:
 *
 * Transport header for compressed JPEG buffers in output streams.
 *
 * To capture JPEG images, a stream is created using the pixel format
 * HAL_PIXEL_FORMAT_BLOB. The buffer size for the stream is calculated by the
 * framework, based on the static metadata field android.jpeg.maxSize. Since
 * compressed JPEG images are of variable size, the HAL needs to include the
 * final size of the compressed image using this structure inside the output
 * stream buffer. The JPEG blob ID field must be set to CAMERA3_JPEG_BLOB_ID.
 *
 * Transport header should be at the end of the JPEG output stream buffer. That
 * means the jpeg_blob_id must start at byte[buffer_size -
 * sizeof(camera3_jpeg_blob)], where the buffer_size is the size of gralloc buffer.
 * Any HAL using this transport header must account for it in android.jpeg.maxSize
 * The JPEG data itself starts at the beginning of the buffer and should be
 * jpeg_size bytes long.
 */
typedef struct camera3_jpeg_blob {
    uint16_t jpeg_blob_id;
    uint32_t jpeg_size;
} camera3_jpeg_blob_t;

enum {
    CAMERA3_JPEG_BLOB_ID = 0x00FF
};

/**********************************************************************
 *
 * Message definitions for the HAL notify() callback.
 *
 * These definitions are used for the HAL notify callback, to signal
 * asynchronous events from the HAL device to the Android framework.
 *
 */

/**
 * camera3_msg_type:
 *
 * Indicates the type of message sent, which specifies which member of the
 * message union is valid.
 *
 */
typedef enum camera3_msg_type {
    /**
     * An error has occurred. camera3_notify_msg.message.error contains the
     * error information.
     */
    CAMERA3_MSG_ERROR = 1,

    /**
     * The exposure of a given request or processing a reprocess request has
     * begun. camera3_notify_msg.message.shutter contains the information
     * the capture.
     */
    CAMERA3_MSG_SHUTTER = 2,

    /**
     * Number of framework message types
     */
    CAMERA3_NUM_MESSAGES

} camera3_msg_type_t;

/**
 * Defined error codes for CAMERA_MSG_ERROR
 */
typedef enum camera3_error_msg_code {
    /**
     * A serious failure occured. No further frames or buffer streams will
     * be produced by the device. Device should be treated as closed. The
     * client must reopen the device to use it again. The frame_number field
     * is unused.
     */
    CAMERA3_MSG_ERROR_DEVICE = 1,

    /**
     * An error has occurred in processing a request. No output (metadata or
     * buffers) will be produced for this request. The frame_number field
     * specifies which request has been dropped. Subsequent requests are
     * unaffected, and the device remains operational.
     */
    CAMERA3_MSG_ERROR_REQUEST = 2,

    /**
     * An error has occurred in producing an output result metadata buffer
     * for a request, but output stream buffers for it will still be
     * available. Subsequent requests are unaffected, and the device remains
     * operational.  The frame_number field specifies the request for which
     * result metadata won't be available.
     */
    CAMERA3_MSG_ERROR_RESULT = 3,

    /**
     * An error has occurred in placing an output buffer into a stream for a
     * request. The frame metadata and other buffers may still be
     * available. Subsequent requests are unaffected, and the device remains
     * operational. The frame_number field specifies the request for which the
     * buffer was dropped, and error_stream contains a pointer to the stream
     * that dropped the frame.u
     */
    CAMERA3_MSG_ERROR_BUFFER = 4,

    /**
     * Number of error types
     */
    CAMERA3_MSG_NUM_ERRORS

} camera3_error_msg_code_t;

/**
 * camera3_error_msg_t:
 *
 * Message contents for CAMERA3_MSG_ERROR
 */
typedef struct camera3_error_msg {
    /**
     * Frame number of the request the error applies to. 0 if the frame number
     * isn't applicable to the error.
     */
    uint32_t frame_number;

    /**
     * Pointer to the stream that had a failure. NULL if the stream isn't
     * applicable to the error.
     */
    camera3_stream_t *error_stream;

    /**
     * The code for this error; one of the CAMERA_MSG_ERROR enum values.
     */
    int error_code;

} camera3_error_msg_t;

/**
 * camera3_shutter_msg_t:
 *
 * Message contents for CAMERA3_MSG_SHUTTER
 */
typedef struct camera3_shutter_msg {
    /**
     * Frame number of the request that has begun exposure or reprocessing.
     */
    uint32_t frame_number;

    /**
     * Timestamp for the start of capture. For a reprocess request, this must
     * be input image's start of capture. This must match the capture result
     * metadata's sensor exposure start timestamp.
     */
    uint64_t timestamp;

} camera3_shutter_msg_t;

/**
 * camera3_notify_msg_t:
 *
 * The message structure sent to camera3_callback_ops_t.notify()
 */
typedef struct camera3_notify_msg {

    /**
     * The message type. One of camera3_notify_msg_type, or a private extension.
     */
    int type;

    union {
        /**
         * Error message contents. Valid if type is CAMERA3_MSG_ERROR
         */
        camera3_error_msg_t error;

        /**
         * Shutter message contents. Valid if type is CAMERA3_MSG_SHUTTER
         */
        camera3_shutter_msg_t shutter;

        /**
         * Generic message contents. Used to ensure a minimum size for custom
         * message types.
         */
        uint8_t generic[32];
    } message;

} camera3_notify_msg_t;

/**********************************************************************
 *
 * Capture request/result definitions for the HAL process_capture_request()
 * method, and the process_capture_result() callback.
 *
 */

/**
 * camera3_request_template_t:
 *
 * Available template types for
 * camera3_device_ops.construct_default_request_settings()
 */
typedef enum camera3_request_template {
    /**
     * Standard camera preview operation with 3A on auto.
     */
    CAMERA3_TEMPLATE_PREVIEW = 1,

    /**
     * Standard camera high-quality still capture with 3A and flash on auto.
     */
    CAMERA3_TEMPLATE_STILL_CAPTURE = 2,

    /**
     * Standard video recording plus preview with 3A on auto, torch off.
     */
    CAMERA3_TEMPLATE_VIDEO_RECORD = 3,

    /**
     * High-quality still capture while recording video. Application will
     * include preview, video record, and full-resolution YUV or JPEG streams in
     * request. Must not cause stuttering on video stream. 3A on auto.
     */
    CAMERA3_TEMPLATE_VIDEO_SNAPSHOT = 4,

    /**
     * Zero-shutter-lag mode. Application will request preview and
     * full-resolution data for each frame, and reprocess it to JPEG when a
     * still image is requested by user. Settings should provide highest-quality
     * full-resolution images without compromising preview frame rate. 3A on
     * auto.
     */
    CAMERA3_TEMPLATE_ZERO_SHUTTER_LAG = 5,

    /**
     * A basic template for direct application control of capture
     * parameters. All automatic control is disabled (auto-exposure, auto-white
     * balance, auto-focus), and post-processing parameters are set to preview
     * quality. The manual capture parameters (exposure, sensitivity, etc.)
     * are set to reasonable defaults, but should be overridden by the
     * application depending on the intended use case.
     */
    CAMERA3_TEMPLATE_MANUAL = 6,

    /* Total number of templates */
    CAMERA3_TEMPLATE_COUNT,

    /**
     * First value for vendor-defined request templates
     */
    CAMERA3_VENDOR_TEMPLATE_START = 0x40000000

} camera3_request_template_t;

/**
 * camera3_capture_request_t:
 *
 * A single request for image capture/buffer reprocessing, sent to the Camera
 * HAL device by the framework in process_capture_request().
 *
 * The request contains the settings to be used for this capture, and the set of
 * output buffers to write the resulting image data in. It may optionally
 * contain an input buffer, in which case the request is for reprocessing that
 * input buffer instead of capturing a new image with the camera sensor. The
 * capture is identified by the frame_number.
 *
 * In response, the camera HAL device must send a camera3_capture_result
 * structure asynchronously to the framework, using the process_capture_result()
 * callback.
 */
typedef struct camera3_capture_request {
    /**
     * The frame number is an incrementing integer set by the framework to
     * uniquely identify this capture. It needs to be returned in the result
     * call, and is also used to identify the request in asynchronous
     * notifications sent to camera3_callback_ops_t.notify().
     */
    uint32_t frame_number;

    /**
     * The settings buffer contains the capture and processing parameters for
     * the request. As a special case, a NULL settings buffer indicates that the
     * settings are identical to the most-recently submitted capture request. A
     * NULL buffer cannot be used as the first submitted request after a
     * configure_streams() call.
     */
    const camera_metadata_t *settings;

    /**
     * The input stream buffer to use for this request, if any.
     *
     * If input_buffer is NULL, then the request is for a new capture from the
     * imager. If input_buffer is valid, the request is for reprocessing the
     * image contained in input_buffer.
     *
     * In the latter case, the HAL must set the release_fence of the
     * input_buffer to a valid sync fence, or to -1 if the HAL does not support
     * sync, before process_capture_request() returns.
     *
     * The HAL is required to wait on the acquire sync fence of the input buffer
     * before accessing it.
     *
     * <= CAMERA_DEVICE_API_VERSION_3_1:
     *
     * Any input buffer included here will have been registered with the HAL
     * through register_stream_buffers() before its inclusion in a request.
     *
     * >= CAMERA_DEVICE_API_VERSION_3_2:
     *
     * The buffers will not have been pre-registered with the HAL.
     * Subsequent requests may reuse buffers, or provide entirely new buffers.
     */
    camera3_stream_buffer_t *input_buffer;

    /**
     * The number of output buffers for this capture request. Must be at least
     * 1.
     */
    uint32_t num_output_buffers;

    /**
     * An array of num_output_buffers stream buffers, to be filled with image
     * data from this capture/reprocess. The HAL must wait on the acquire fences
     * of each stream buffer before writing to them.
     *
     * The HAL takes ownership of the actual buffer_handle_t entries in
     * output_buffers; the framework does not access them until they are
     * returned in a camera3_capture_result_t.
     *
     * <= CAMERA_DEVICE_API_VERSION_3_1:
     *
     * All the buffers included  here will have been registered with the HAL
     * through register_stream_buffers() before their inclusion in a request.
     *
     * >= CAMERA_DEVICE_API_VERSION_3_2:
     *
     * Any or all of the buffers included here may be brand new in this
     * request (having never before seen by the HAL).
     */
    const camera3_stream_buffer_t *output_buffers;

} camera3_capture_request_t;

/**
 * camera3_capture_result_t:
 *
 * The result of a single capture/reprocess by the camera HAL device. This is
 * sent to the framework asynchronously with process_capture_result(), in
 * response to a single capture request sent to the HAL with
 * process_capture_request(). Multiple process_capture_result() calls may be
 * performed by the HAL for each request.
 *
 * Each call, all with the same frame
 * number, may contain some subset of the output buffers, and/or the result
 * metadata. The metadata may only be provided once for a given frame number;
 * all other calls must set the result metadata to NULL.
 *
 * The result structure contains the output metadata from this capture, and the
 * set of output buffers that have been/will be filled for this capture. Each
 * output buffer may come with a release sync fence that the framework will wait
 * on before reading, in case the buffer has not yet been filled by the HAL.
 *
 * >= CAMERA_DEVICE_API_VERSION_3_2:
 *
 * The metadata may be provided multiple times for a single frame number. The
 * framework will accumulate together the final result set by combining each
 * partial result together into the total result set.
 *
 * If an input buffer is given in a request, the HAL must return it in one of
 * the process_capture_result calls, and the call may be to just return the input
 * buffer, without metadata and output buffers; the sync fences must be handled
 * the same way they are done for output buffers.
 *
 *
 * Performance considerations:
 *
 * Applications will also receive these partial results immediately, so sending
 * partial results is a highly recommended performance optimization to avoid
 * the total pipeline latency before sending the results for what is known very
 * early on in the pipeline.
 *
 * A typical use case might be calculating the AF state halfway through the
 * pipeline; by sending the state back to the framework immediately, we get a
 * 50% performance increase and perceived responsiveness of the auto-focus.
 *
 */
typedef struct camera3_capture_result {
    /**
     * The frame number is an incrementing integer set by the framework in the
     * submitted request to uniquely identify this capture. It is also used to
     * identify the request in asynchronous notifications sent to
     * camera3_callback_ops_t.notify().
    */
    uint32_t frame_number;

    /**
     * The result metadata for this capture. This contains information about the
     * final capture parameters, the state of the capture and post-processing
     * hardware, the state of the 3A algorithms, if enabled, and the output of
     * any enabled statistics units.
     *
     * Only one call to process_capture_result() with a given frame_number may
     * include the result metadata. All other calls for the same frame_number
     * must set this to NULL.
     *
     * If there was an error producing the result metadata, result must be an
     * empty metadata buffer, and notify() must be called with ERROR_RESULT.
     *
     * >= CAMERA_DEVICE_API_VERSION_3_2:
     *
     * Multiple calls to process_capture_result() with a given frame_number
     * may include the result metadata.
     *
     * Partial metadata submitted should not include any metadata key returned
     * in a previous partial result for a given frame. Each new partial result
     * for that frame must also set a distinct partial_result value.
     *
     * If notify has been called with ERROR_RESULT, all further partial
     * results for that frame are ignored by the framework.
     */
    const camera_metadata_t *result;

    /**
     * The number of output buffers returned in this result structure. Must be
     * less than or equal to the matching capture request's count. If this is
     * less than the buffer count in the capture request, at least one more call
     * to process_capture_result with the same frame_number must be made, to
     * return the remaining output buffers to the framework. This may only be
     * zero if the structure includes valid result metadata or an input buffer
     * is returned in this result.
     */
    uint32_t num_output_buffers;

    /**
     * The handles for the output stream buffers for this capture. They may not
     * yet be filled at the time the HAL calls process_capture_result(); the
     * framework will wait on the release sync fences provided by the HAL before
     * reading the buffers.
     *
     * The HAL must set the stream buffer's release sync fence to a valid sync
     * fd, or to -1 if the buffer has already been filled.
     *
     * If the HAL encounters an error while processing the buffer, and the
     * buffer is not filled, the buffer's status field must be set to
     * CAMERA3_BUFFER_STATUS_ERROR. If the HAL did not wait on the acquire fence
     * before encountering the error, the acquire fence should be copied into
     * the release fence, to allow the framework to wait on the fence before
     * reusing the buffer.
     *
     * The acquire fence must be set to -1 for all output buffers.  If
     * num_output_buffers is zero, this may be NULL. In that case, at least one
     * more process_capture_result call must be made by the HAL to provide the
     * output buffers.
     *
     * When process_capture_result is called with a new buffer for a frame,
     * all previous frames' buffers for that corresponding stream must have been
     * already delivered (the fences need not have yet been signaled).
     *
     * >= CAMERA_DEVICE_API_VERSION_3_2:
     *
     * Gralloc buffers for a frame may be sent to framework before the
     * corresponding SHUTTER-notify.
     *
     * Performance considerations:
     *
     * Buffers delivered to the framework will not be dispatched to the
     * application layer until a start of exposure timestamp has been received
     * via a SHUTTER notify() call. It is highly recommended to
     * dispatch that call as early as possible.
     */
     const camera3_stream_buffer_t *output_buffers;

     /**
      * >= CAMERA_DEVICE_API_VERSION_3_2:
      *
      * The handle for the input stream buffer for this capture. It may not
      * yet be consumed at the time the HAL calls process_capture_result(); the
      * framework will wait on the release sync fences provided by the HAL before
      * reusing the buffer.
      *
      * The HAL should handle the sync fences the same way they are done for
      * output_buffers.
      *
      * Only one input buffer is allowed to be sent per request. Similarly to
      * output buffers, the ordering of returned input buffers must be
      * maintained by the HAL.
      *
      * Performance considerations:
      *
      * The input buffer should be returned as early as possible. If the HAL
      * supports sync fences, it can call process_capture_result to hand it back
      * with sync fences being set appropriately. If the sync fences are not
      * supported, the buffer can only be returned when it is consumed, which
      * may take long time; the HAL may choose to copy this input buffer to make
      * the buffer return sooner.
      */
      const camera3_stream_buffer_t *input_buffer;

     /**
      * >= CAMERA_DEVICE_API_VERSION_3_2:
      *
      * In order to take advantage of partial results, the HAL must set the
      * static metadata android.request.partialResultCount to the number of
      * partial results it will send for each frame.
      *
      * Each new capture result with a partial result must set
      * this field (partial_result) to a distinct inclusive value between
      * 1 and android.request.partialResultCount.
      *
      * HALs not wishing to take advantage of this feature must not
      * set an android.request.partialResultCount or partial_result to a value
      * other than 1.
      *
      * This value must be set to 0 when a capture result contains buffers only
      * and no metadata.
      */
     uint32_t partial_result;

} camera3_capture_result_t;

/**********************************************************************
 *
 * Callback methods for the HAL to call into the framework.
 *
 * These methods are used to return metadata and image buffers for a completed
 * or failed captures, and to notify the framework of asynchronous events such
 * as errors.
 *
 * The framework will not call back into the HAL from within these callbacks,
 * and these calls will not block for extended periods.
 *
 */
typedef struct camera3_callback_ops {

    /**
     * process_capture_result:
     *
     * Send results from a completed capture to the framework.
     * process_capture_result() may be invoked multiple times by the HAL in
     * response to a single capture request. This allows, for example, the
     * metadata and low-resolution buffers to be returned in one call, and
     * post-processed JPEG buffers in a later call, once it is available. Each
     * call must include the frame number of the request it is returning
     * metadata or buffers for.
     *
     * A component (buffer or metadata) of the complete result may only be
     * included in one process_capture_result call. A buffer for each stream,
     * and the result metadata, must be returned by the HAL for each request in
     * one of the process_capture_result calls, even in case of errors producing
     * some of the output. A call to process_capture_result() with neither
     * output buffers or result metadata is not allowed.
     *
     * The order of returning metadata and buffers for a single result does not
     * matter, but buffers for a given stream must be returned in FIFO order. So
     * the buffer for request 5 for stream A must always be returned before the
     * buffer for request 6 for stream A. This also applies to the result
     * metadata; the metadata for request 5 must be returned before the metadata
     * for request 6.
     *
     * However, different streams are independent of each other, so it is
     * acceptable and expected that the buffer for request 5 for stream A may be
     * returned after the buffer for request 6 for stream B is. And it is
     * acceptable that the result metadata for request 6 for stream B is
     * returned before the buffer for request 5 for stream A is.
     *
     * The HAL retains ownership of result structure, which only needs to be
     * valid to access during this call. The framework will copy whatever it
     * needs before this call returns.
     *
     * The output buffers do not need to be filled yet; the framework will wait
     * on the stream buffer release sync fence before reading the buffer
     * data. Therefore, this method should be called by the HAL as soon as
     * possible, even if some or all of the output buffers are still in
     * being filled. The HAL must include valid release sync fences into each
     * output_buffers stream buffer entry, or -1 if that stream buffer is
     * already filled.
     *
     * If the result buffer cannot be constructed for a request, the HAL should
     * return an empty metadata buffer, but still provide the output buffers and
     * their sync fences. In addition, notify() must be called with an
     * ERROR_RESULT message.
     *
     * If an output buffer cannot be filled, its status field must be set to
     * STATUS_ERROR. In addition, notify() must be called with a ERROR_BUFFER
     * message.
     *
     * If the entire capture has failed, then this method still needs to be
     * called to return the output buffers to the framework. All the buffer
     * statuses should be STATUS_ERROR, and the result metadata should be an
     * empty buffer. In addition, notify() must be called with a ERROR_REQUEST
     * message. In this case, individual ERROR_RESULT/ERROR_BUFFER messages
     * should not be sent.
     *
     * Performance requirements:
     *
     * This is a non-blocking call. The framework will return this call in 5ms.
     *
     * The pipeline latency (see S7 for definition) should be less than or equal to
     * 4 frame intervals, and must be less than or equal to 8 frame intervals.
     *
     */
    void (*process_capture_result)(const struct camera3_callback_ops *,
            const camera3_capture_result_t *result);

    /**
     * notify:
     *
     * Asynchronous notification callback from the HAL, fired for various
     * reasons. Only for information independent of frame capture, or that
     * require specific timing. The ownership of the message structure remains
     * with the HAL, and the msg only needs to be valid for the duration of this
     * call.
     *
     * Multiple threads may call notify() simultaneously.
     *
     * <= CAMERA_DEVICE_API_VERSION_3_1:
     *
     * The notification for the start of exposure for a given request must be
     * sent by the HAL before the first call to process_capture_result() for
     * that request is made.
     *
     * >= CAMERA_DEVICE_API_VERSION_3_2:
     *
     * Buffers delivered to the framework will not be dispatched to the
     * application layer until a start of exposure timestamp (or input image's
     * start of exposure timestamp for a reprocess request) has been received
     * via a SHUTTER notify() call. It is highly recommended to dispatch this
     * call as early as possible.
     *
     * ------------------------------------------------------------------------
     * Performance requirements:
     *
     * This is a non-blocking call. The framework will return this call in 5ms.
     */
    void (*notify)(const struct camera3_callback_ops *,
            const camera3_notify_msg_t *msg);

} camera3_callback_ops_t;

/**********************************************************************
 *
 * Camera device operations
 *
 */
typedef struct camera3_device_ops {

    /**
     * initialize:
     *
     * One-time initialization to pass framework callback function pointers to
     * the HAL. Will be called once after a successful open() call, before any
     * other functions are called on the camera3_device_ops structure.
     *
     * Performance requirements:
     *
     * This should be a non-blocking call. The HAL should return from this call
     * in 5ms, and must return from this call in 10ms.
     *
     * Return values:
     *
     *  0:     On successful initialization
     *
     * -ENODEV: If initialization fails. Only close() can be called successfully
     *          by the framework after this.
     */
    int (*initialize)(const struct camera3_device *,
            const camera3_callback_ops_t *callback_ops);

    /**********************************************************************
     * Stream management
     */

    /**
     * configure_streams:
     *
     * CAMERA_DEVICE_API_VERSION_3_0 only:
     *
     * Reset the HAL camera device processing pipeline and set up new input and
     * output streams. This call replaces any existing stream configuration with
     * the streams defined in the stream_list. This method will be called at
     * least once after initialize() before a request is submitted with
     * process_capture_request().
     *
     * The stream_list must contain at least one output-capable stream, and may
     * not contain more than one input-capable stream.
     *
     * The stream_list may contain streams that are also in the currently-active
     * set of streams (from the previous call to configure_stream()). These
     * streams will already have valid values for usage, max_buffers, and the
     * private pointer.
     *
     * If such a stream has already had its buffers registered,
     * register_stream_buffers() will not be called again for the stream, and
     * buffers from the stream can be immediately included in input requests.
     *
     * If the HAL needs to change the stream configuration for an existing
     * stream due to the new configuration, it may rewrite the values of usage
     * and/or max_buffers during the configure call.
     *
     * The framework will detect such a change, and will then reallocate the
     * stream buffers, and call register_stream_buffers() again before using
     * buffers from that stream in a request.
     *
     * If a currently-active stream is not included in stream_list, the HAL may
     * safely remove any references to that stream. It will not be reused in a
     * later configure() call by the framework, and all the gralloc buffers for
     * it will be freed after the configure_streams() call returns.
     *
     * The stream_list structure is owned by the framework, and may not be
     * accessed once this call completes. The address of an individual
     * camera3_stream_t structure will remain valid for access by the HAL until
     * the end of the first configure_stream() call which no longer includes
     * that camera3_stream_t in the stream_list argument. The HAL may not change
     * values in the stream structure outside of the private pointer, except for
     * the usage and max_buffers members during the configure_streams() call
     * itself.
     *
     * If the stream is new, the usage, max_buffer, and private pointer fields
     * of the stream structure will all be set to 0. The HAL device must set
     * these fields before the configure_streams() call returns. These fields
     * are then used by the framework and the platform gralloc module to
     * allocate the gralloc buffers for each stream.
     *
     * Before such a new stream can have its buffers included in a capture
     * request, the framework will call register_stream_buffers() with that
     * stream. However, the framework is not required to register buffers for
     * _all_ streams before submitting a request. This allows for quick startup
     * of (for example) a preview stream, with allocation for other streams
     * happening later or concurrently.
     *
     * ------------------------------------------------------------------------
     * CAMERA_DEVICE_API_VERSION_3_1 only:
     *
     * Reset the HAL camera device processing pipeline and set up new input and
     * output streams. This call replaces any existing stream configuration with
     * the streams defined in the stream_list. This method will be called at
     * least once after initialize() before a request is submitted with
     * process_capture_request().
     *
     * The stream_list must contain at least one output-capable stream, and may
     * not contain more than one input-capable stream.
     *
     * The stream_list may contain streams that are also in the currently-active
     * set of streams (from the previous call to configure_stream()). These
     * streams will already have valid values for usage, max_buffers, and the
     * private pointer.
     *
     * If such a stream has already had its buffers registered,
     * register_stream_buffers() will not be called again for the stream, and
     * buffers from the stream can be immediately included in input requests.
     *
     * If the HAL needs to change the stream configuration for an existing
     * stream due to the new configuration, it may rewrite the values of usage
     * and/or max_buffers during the configure call.
     *
     * The framework will detect such a change, and will then reallocate the
     * stream buffers, and call register_stream_buffers() again before using
     * buffers from that stream in a request.
     *
     * If a currently-active stream is not included in stream_list, the HAL may
     * safely remove any references to that stream. It will not be reused in a
     * later configure() call by the framework, and all the gralloc buffers for
     * it will be freed after the configure_streams() call returns.
     *
     * The stream_list structure is owned by the framework, and may not be
     * accessed once this call completes. The address of an individual
     * camera3_stream_t structure will remain valid for access by the HAL until
     * the end of the first configure_stream() call which no longer includes
     * that camera3_stream_t in the stream_list argument. The HAL may not change
     * values in the stream structure outside of the private pointer, except for
     * the usage and max_buffers members during the configure_streams() call
     * itself.
     *
     * If the stream is new, max_buffer, and private pointer fields of the
     * stream structure will all be set to 0. The usage will be set to the
     * consumer usage flags. The HAL device must set these fields before the
     * configure_streams() call returns. These fields are then used by the
     * framework and the platform gralloc module to allocate the gralloc
     * buffers for each stream.
     *
     * Before such a new stream can have its buffers included in a capture
     * request, the framework will call register_stream_buffers() with that
     * stream. However, the framework is not required to register buffers for
     * _all_ streams before submitting a request. This allows for quick startup
     * of (for example) a preview stream, with allocation for other streams
     * happening later or concurrently.
     *
     * ------------------------------------------------------------------------
     * >= CAMERA_DEVICE_API_VERSION_3_2:
     *
     * Reset the HAL camera device processing pipeline and set up new input and
     * output streams. This call replaces any existing stream configuration with
     * the streams defined in the stream_list. This method will be called at
     * least once after initialize() before a request is submitted with
     * process_capture_request().
     *
     * The stream_list must contain at least one output-capable stream, and may
     * not contain more than one input-capable stream.
     *
     * The stream_list may contain streams that are also in the currently-active
     * set of streams (from the previous call to configure_stream()). These
     * streams will already have valid values for usage, max_buffers, and the
     * private pointer.
     *
     * If the HAL needs to change the stream configuration for an existing
     * stream due to the new configuration, it may rewrite the values of usage
     * and/or max_buffers during the configure call.
     *
     * The framework will detect such a change, and may then reallocate the
     * stream buffers before using buffers from that stream in a request.
     *
     * If a currently-active stream is not included in stream_list, the HAL may
     * safely remove any references to that stream. It will not be reused in a
     * later configure() call by the framework, and all the gralloc buffers for
     * it will be freed after the configure_streams() call returns.
     *
     * The stream_list structure is owned by the framework, and may not be
     * accessed once this call completes. The address of an individual
     * camera3_stream_t structure will remain valid for access by the HAL until
     * the end of the first configure_stream() call which no longer includes
     * that camera3_stream_t in the stream_list argument. The HAL may not change
     * values in the stream structure outside of the private pointer, except for
     * the usage and max_buffers members during the configure_streams() call
     * itself.
     *
     * If the stream is new, max_buffer, and private pointer fields of the
     * stream structure will all be set to 0. The usage will be set to the
     * consumer usage flags. The HAL device must set these fields before the
     * configure_streams() call returns. These fields are then used by the
     * framework and the platform gralloc module to allocate the gralloc
     * buffers for each stream.
     *
     * Newly allocated buffers may be included in a capture request at any time
     * by the framework. Once a gralloc buffer is returned to the framework
     * with process_capture_result (and its respective release_fence has been
     * signaled) the framework may free or reuse it at any time.
     *
     * ------------------------------------------------------------------------
     *
     * Preconditions:
     *
     * The framework will only call this method when no captures are being
     * processed. That is, all results have been returned to the framework, and
     * all in-flight input and output buffers have been returned and their
     * release sync fences have been signaled by the HAL. The framework will not
     * submit new requests for capture while the configure_streams() call is
     * underway.
     *
     * Postconditions:
     *
     * The HAL device must configure itself to provide maximum possible output
     * frame rate given the sizes and formats of the output streams, as
     * documented in the camera device's static metadata.
     *
     * Performance requirements:
     *
     * This call is expected to be heavyweight and possibly take several hundred
     * milliseconds to complete, since it may require resetting and
     * reconfiguring the image sensor and the camera processing pipeline.
     * Nevertheless, the HAL device should attempt to minimize the
     * reconfiguration delay to minimize the user-visible pauses during
     * application operational mode changes (such as switching from still
     * capture to video recording).
     *
     * The HAL should return from this call in 500ms, and must return from this
     * call in 1000ms.
     *
     * Return values:
     *
     *  0:      On successful stream configuration
     *
     * -EINVAL: If the requested stream configuration is invalid. Some examples
     *          of invalid stream configurations include:
     *
     *          - Including more than 1 input-capable stream (INPUT or
     *            BIDIRECTIONAL)
     *
     *          - Not including any output-capable streams (OUTPUT or
     *            BIDIRECTIONAL)
     *
     *          - Including streams with unsupported formats, or an unsupported
     *            size for that format.
     *
     *          - Including too many output streams of a certain format.
     *
     *          - Unsupported rotation configuration (only applies to
     *            devices with version >= CAMERA_DEVICE_API_VERSION_3_3)
     *
     *          - Stream sizes/formats don't satisfy the
     *            camera3_stream_configuration_t->operation_mode requirements for non-NORMAL mode,
     *            or the requested operation_mode is not supported by the HAL.
     *            (only applies to devices with version >= CAMERA_DEVICE_API_VERSION_3_3)
     *
     *          Note that the framework submitting an invalid stream
     *          configuration is not normal operation, since stream
     *          configurations are checked before configure. An invalid
     *          configuration means that a bug exists in the framework code, or
     *          there is a mismatch between the HAL's static metadata and the
     *          requirements on streams.
     *
     * -ENODEV: If there has been a fatal error and the device is no longer
     *          operational. Only close() can be called successfully by the
     *          framework after this error is returned.
     */
    int (*configure_streams)(const struct camera3_device *,
            camera3_stream_configuration_t *stream_list);

    /**
     * register_stream_buffers:
     *
     * >= CAMERA_DEVICE_API_VERSION_3_2:
     *
     * DEPRECATED. This will not be called and must be set to NULL.
     *
     * <= CAMERA_DEVICE_API_VERSION_3_1:
     *
     * Register buffers for a given stream with the HAL device. This method is
     * called by the framework after a new stream is defined by
     * configure_streams, and before buffers from that stream are included in a
     * capture request. If the same stream is listed in a subsequent
     * configure_streams() call, register_stream_buffers will _not_ be called
     * again for that stream.
     *
     * The framework does not need to register buffers for all configured
     * streams before it submits the first capture request. This allows quick
     * startup for preview (or similar use cases) while other streams are still
     * being allocated.
     *
     * This method is intended to allow the HAL device to map or otherwise
     * prepare the buffers for later use. The buffers passed in will already be
     * locked for use. At the end of the call, all the buffers must be ready to
     * be returned to the stream.  The buffer_set argument is only valid for the
     * duration of this call.
     *
     * If the stream format was set to HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED,
     * the camera HAL should inspect the passed-in buffers here to determine any
     * platform-private pixel format information.
     *
     * Performance requirements:
     *
     * This should be a non-blocking call. The HAL should return from this call
     * in 1ms, and must return from this call in 5ms.
     *
     * Return values:
     *
     *  0:      On successful registration of the new stream buffers
     *
     * -EINVAL: If the stream_buffer_set does not refer to a valid active
     *          stream, or if the buffers array is invalid.
     *
     * -ENOMEM: If there was a failure in registering the buffers. The framework
     *          must consider all the stream buffers to be unregistered, and can
     *          try to register again later.
     *
     * -ENODEV: If there is a fatal error, and the device is no longer
     *          operational. Only close() can be called successfully by the
     *          framework after this error is returned.
     */
    int (*register_stream_buffers)(const struct camera3_device *,
            const camera3_stream_buffer_set_t *buffer_set);

    /**********************************************************************
     * Request creation and submission
     */

    /**
     * construct_default_request_settings:
     *
     * Create capture settings for standard camera use cases.
     *
     * The device must return a settings buffer that is configured to meet the
     * requested use case, which must be one of the CAMERA3_TEMPLATE_*
     * enums. All request control fields must be included.
     *
     * The HAL retains ownership of this structure, but the pointer to the
     * structure must be valid until the device is closed. The framework and the
     * HAL may not modify the buffer once it is returned by this call. The same
     * buffer may be returned for subsequent calls for the same template, or for
     * other templates.
     *
     * Performance requirements:
     *
     * This should be a non-blocking call. The HAL should return from this call
     * in 1ms, and must return from this call in 5ms.
     *
     * Return values:
     *
     *   Valid metadata: On successful creation of a default settings
     *                   buffer.
     *
     *   NULL:           In case of a fatal error. After this is returned, only
     *                   the close() method can be called successfully by the
     *                   framework.
     */
    const camera_metadata_t* (*construct_default_request_settings)(
            const struct camera3_device *,
            int type);

    /**
     * process_capture_request:
     *
     * Send a new capture request to the HAL. The HAL should not return from
     * this call until it is ready to accept the next request to process. Only
     * one call to process_capture_request() will be made at a time by the
     * framework, and the calls will all be from the same thread. The next call
     * to process_capture_request() will be made as soon as a new request and
     * its associated buffers are available. In a normal preview scenario, this
     * means the function will be called again by the framework almost
     * instantly.
     *
     * The actual request processing is asynchronous, with the results of
     * capture being returned by the HAL through the process_capture_result()
     * call. This call requires the result metadata to be available, but output
     * buffers may simply provide sync fences to wait on. Multiple requests are
     * expected to be in flight at once, to maintain full output frame rate.
     *
     * The framework retains ownership of the request structure. It is only
     * guaranteed to be valid during this call. The HAL device must make copies
     * of the information it needs to retain for the capture processing. The HAL
     * is responsible for waiting on and closing the buffers' fences and
     * returning the buffer handles to the framework.
     *
     * The HAL must write the file descriptor for the input buffer's release
     * sync fence into input_buffer->release_fence, if input_buffer is not
     * NULL. If the HAL returns -1 for the input buffer release sync fence, the
     * framework is free to immediately reuse the input buffer. Otherwise, the
     * framework will wait on the sync fence before refilling and reusing the
     * input buffer.
     *
     * >= CAMERA_DEVICE_API_VERSION_3_2:
     *
     * The input/output buffers provided by the framework in each request
     * may be brand new (having never before seen by the HAL).
     *
     * ------------------------------------------------------------------------
     * Performance considerations:
     *
     * Handling a new buffer should be extremely lightweight and there should be
     * no frame rate degradation or frame jitter introduced.
     *
     * This call must return fast enough to ensure that the requested frame
     * rate can be sustained, especially for streaming cases (post-processing
     * quality settings set to FAST). The HAL should return this call in 1
     * frame interval, and must return from this call in 4 frame intervals.
     *
     * Return values:
     *
     *  0:      On a successful start to processing the capture request
     *
     * -EINVAL: If the input is malformed (the settings are NULL when not
     *          allowed, there are 0 output buffers, etc) and capture processing
     *          cannot start. Failures during request processing should be
     *          handled by calling camera3_callback_ops_t.notify(). In case of
     *          this error, the framework will retain responsibility for the
     *          stream buffers' fences and the buffer handles; the HAL should
     *          not close the fences or return these buffers with
     *          process_capture_result.
     *
     * -ENODEV: If the camera device has encountered a serious error. After this
     *          error is returned, only the close() method can be successfully
     *          called by the framework.
     *
     */
    int (*process_capture_request)(const struct camera3_device *,
            camera3_capture_request_t *request);

    /**********************************************************************
     * Miscellaneous methods
     */

    /**
     * get_metadata_vendor_tag_ops:
     *
     * Get methods to query for vendor extension metadata tag information. The
     * HAL should fill in all the vendor tag operation methods, or leave ops
     * unchanged if no vendor tags are defined.
     *
     * The definition of vendor_tag_query_ops_t can be found in
     * system/media/camera/include/system/camera_metadata.h.
     *
     * >= CAMERA_DEVICE_API_VERSION_3_2:
     *    DEPRECATED. This function has been deprecated and should be set to
     *    NULL by the HAL.  Please implement get_vendor_tag_ops in camera_common.h
     *    instead.
     */
    void (*get_metadata_vendor_tag_ops)(const struct camera3_device*,
            vendor_tag_query_ops_t* ops);

    /**
     * dump:
     *
     * Print out debugging state for the camera device. This will be called by
     * the framework when the camera service is asked for a debug dump, which
     * happens when using the dumpsys tool, or when capturing a bugreport.
     *
     * The passed-in file descriptor can be used to write debugging text using
     * dprintf() or write(). The text should be in ASCII encoding only.
     *
     * Performance requirements:
     *
     * This must be a non-blocking call. The HAL should return from this call
     * in 1ms, must return from this call in 10ms. This call must avoid
     * deadlocks, as it may be called at any point during camera operation.
     * Any synchronization primitives used (such as mutex locks or semaphores)
     * should be acquired with a timeout.
     */
    void (*dump)(const struct camera3_device *, int fd);

    /**
     * flush:
     *
     * Flush all currently in-process captures and all buffers in the pipeline
     * on the given device. The framework will use this to dump all state as
     * quickly as possible in order to prepare for a configure_streams() call.
     *
     * No buffers are required to be successfully returned, so every buffer
     * held at the time of flush() (whether successfully filled or not) may be
     * returned with CAMERA3_BUFFER_STATUS_ERROR. Note the HAL is still allowed
     * to return valid (CAMERA3_BUFFER_STATUS_OK) buffers during this call,
     * provided they are successfully filled.
     *
     * All requests currently in the HAL are expected to be returned as soon as
     * possible.  Not-in-process requests should return errors immediately. Any
     * interruptible hardware blocks should be stopped, and any uninterruptible
     * blocks should be waited on.
     *
     * flush() may be called concurrently to process_capture_request(), with the expectation that
     * process_capture_request will return quickly and the request submitted in that
     * process_capture_request call is treated like all other in-flight requests.  Due to
     * concurrency issues, it is possible that from the HAL's point of view, a
     * process_capture_request() call may be started after flush has been invoked but has not
     * returned yet. If such a call happens before flush() returns, the HAL should treat the new
     * capture request like other in-flight pending requests (see #4 below).
     *
     * More specifically, the HAL must follow below requirements for various cases:
     *
     * 1. For captures that are too late for the HAL to cancel/stop, and will be
     *    completed normally by the HAL; i.e. the HAL can send shutter/notify and
     *    process_capture_result and buffers as normal.
     *
     * 2. For pending requests that have not done any processing, the HAL must call notify
     *    CAMERA3_MSG_ERROR_REQUEST, and return all the output buffers with
     *    process_capture_result in the error state (CAMERA3_BUFFER_STATUS_ERROR).
     *    The HAL must not place the release fence into an error state, instead,
     *    the release fences must be set to the acquire fences passed by the framework,
     *    or -1 if they have been waited on by the HAL already. This is also the path
     *    to follow for any captures for which the HAL already called notify() with
     *    CAMERA3_MSG_SHUTTER but won't be producing any metadata/valid buffers for.
     *    After CAMERA3_MSG_ERROR_REQUEST, for a given frame, only process_capture_results with
     *    buffers in CAMERA3_BUFFER_STATUS_ERROR are allowed. No further notifys or
     *    process_capture_result with non-null metadata is allowed.
     *
     * 3. For partially completed pending requests that will not have all the output
     *    buffers or perhaps missing metadata, the HAL should follow below:
     *
     *    3.1. Call notify with CAMERA3_MSG_ERROR_RESULT if some of the expected result
     *    metadata (i.e. one or more partial metadata) won't be available for the capture.
     *
     *    3.2. Call notify with CAMERA3_MSG_ERROR_BUFFER for every buffer that won't
     *         be produced for the capture.
     *
     *    3.3  Call notify with CAMERA3_MSG_SHUTTER with the capture timestamp before
     *         any buffers/metadata are returned with process_capture_result.
     *
     *    3.4 For captures that will produce some results, the HAL must not call
     *        CAMERA3_MSG_ERROR_REQUEST, since that indicates complete failure.
     *
     *    3.5. Valid buffers/metadata should be passed to the framework as normal.
     *
     *    3.6. Failed buffers should be returned to the framework as described for case 2.
     *         But failed buffers do not have to follow the strict ordering valid buffers do,
     *         and may be out-of-order with respect to valid buffers. For example, if buffers
     *         A, B, C, D, E are sent, D and E are failed, then A, E, B, D, C is an acceptable
     *         return order.
     *
     *    3.7. For fully-missing metadata, calling CAMERA3_MSG_ERROR_RESULT is sufficient, no
     *         need to call process_capture_result with NULL metadata or equivalent.
     *
     * 4. If a flush() is invoked while a process_capture_request() invocation is active, that
     *    process call should return as soon as possible. In addition, if a process_capture_request()
     *    call is made after flush() has been invoked but before flush() has returned, the
     *    capture request provided by the late process_capture_request call should be treated like
     *    a pending request in case #2 above.
     *
     * flush() should only return when there are no more outstanding buffers or
     * requests left in the HAL. The framework may call configure_streams (as
     * the HAL state is now quiesced) or may issue new requests.
     *
     * Note that it's sufficient to only support fully-succeeded and fully-failed result cases.
     * However, it is highly desirable to support the partial failure cases as well, as it
     * could help improve the flush call overall performance.
     *
     * Performance requirements:
     *
     * The HAL should return from this call in 100ms, and must return from this
     * call in 1000ms. And this call must not be blocked longer than pipeline
     * latency (see S7 for definition).
     *
     * Version information:
     *
     *   only available if device version >= CAMERA_DEVICE_API_VERSION_3_1.
     *
     * Return values:
     *
     *  0:      On a successful flush of the camera HAL.
     *
     * -EINVAL: If the input is malformed (the device is not valid).
     *
     * -ENODEV: If the camera device has encountered a serious error. After this
     *          error is returned, only the close() method can be successfully
     *          called by the framework.
     */
    int (*flush)(const struct camera3_device *);

    /* reserved for future use */
    void *reserved[8];
} camera3_device_ops_t;

/**********************************************************************
 *
 * Camera device definition
 *
 */
typedef struct camera3_device {
    /**
     * common.version must equal CAMERA_DEVICE_API_VERSION_3_0 to identify this
     * device as implementing version 3.0 of the camera device HAL.
     *
     * Performance requirements:
     *
     * Camera open (common.module->common.methods->open) should return in 200ms, and must return
     * in 500ms.
     * Camera close (common.close) should return in 200ms, and must return in 500ms.
     *
     */
    hw_device_t common;
    camera3_device_ops_t *ops;
    void *priv;
} camera3_device_t;

__END_DECLS

#endif /* #ifdef ANDROID_INCLUDE_CAMERA3_H */
