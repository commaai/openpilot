/*
 * Copyright (C) 2011 The Android Open Source Project
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

#ifndef SYSTEM_CORE_INCLUDE_ANDROID_CAMERA_H
#define SYSTEM_CORE_INCLUDE_ANDROID_CAMERA_H

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <cutils/native_handle.h>
#include <hardware/hardware.h>
#include <hardware/gralloc.h>

__BEGIN_DECLS

/**
 * A set of bit masks for specifying how the received preview frames are
 * handled before the previewCallback() call.
 *
 * The least significant 3 bits of an "int" value are used for this purpose:
 *
 * ..... 0 0 0
 *       ^ ^ ^
 *       | | |---------> determine whether the callback is enabled or not
 *       | |-----------> determine whether the callback is one-shot or not
 *       |-------------> determine whether the frame is copied out or not
 *
 * WARNING: When a frame is sent directly without copying, it is the frame
 * receiver's responsiblity to make sure that the frame data won't get
 * corrupted by subsequent preview frames filled by the camera. This flag is
 * recommended only when copying out data brings significant performance price
 * and the handling/processing of the received frame data is always faster than
 * the preview frame rate so that data corruption won't occur.
 *
 * For instance,
 * 1. 0x00 disables the callback. In this case, copy out and one shot bits
 *    are ignored.
 * 2. 0x01 enables a callback without copying out the received frames. A
 *    typical use case is the Camcorder application to avoid making costly
 *    frame copies.
 * 3. 0x05 is enabling a callback with frame copied out repeatedly. A typical
 *    use case is the Camera application.
 * 4. 0x07 is enabling a callback with frame copied out only once. A typical
 *    use case is the Barcode scanner application.
 */

enum {
    CAMERA_FRAME_CALLBACK_FLAG_ENABLE_MASK = 0x01,
    CAMERA_FRAME_CALLBACK_FLAG_ONE_SHOT_MASK = 0x02,
    CAMERA_FRAME_CALLBACK_FLAG_COPY_OUT_MASK = 0x04,
    /** Typical use cases */
    CAMERA_FRAME_CALLBACK_FLAG_NOOP = 0x00,
    CAMERA_FRAME_CALLBACK_FLAG_CAMCORDER = 0x01,
    CAMERA_FRAME_CALLBACK_FLAG_CAMERA = 0x05,
    CAMERA_FRAME_CALLBACK_FLAG_BARCODE_SCANNER = 0x07
};

/** msgType in notifyCallback and dataCallback functions */
enum {
    CAMERA_MSG_ERROR = 0x0001,            // notifyCallback
    CAMERA_MSG_SHUTTER = 0x0002,          // notifyCallback
    CAMERA_MSG_FOCUS = 0x0004,            // notifyCallback
    CAMERA_MSG_ZOOM = 0x0008,             // notifyCallback
    CAMERA_MSG_PREVIEW_FRAME = 0x0010,    // dataCallback
    CAMERA_MSG_VIDEO_FRAME = 0x0020,      // data_timestamp_callback
    CAMERA_MSG_POSTVIEW_FRAME = 0x0040,   // dataCallback
    CAMERA_MSG_RAW_IMAGE = 0x0080,        // dataCallback
    CAMERA_MSG_COMPRESSED_IMAGE = 0x0100, // dataCallback
    CAMERA_MSG_RAW_IMAGE_NOTIFY = 0x0200, // dataCallback
    // Preview frame metadata. This can be combined with
    // CAMERA_MSG_PREVIEW_FRAME in dataCallback. For example, the apps can
    // request FRAME and METADATA. Or the apps can request only FRAME or only
    // METADATA.
    CAMERA_MSG_PREVIEW_METADATA = 0x0400, // dataCallback
    // Notify on autofocus start and stop. This is useful in continuous
    // autofocus - FOCUS_MODE_CONTINUOUS_VIDEO and FOCUS_MODE_CONTINUOUS_PICTURE.
    CAMERA_MSG_FOCUS_MOVE = 0x0800,       // notifyCallback
    CAMERA_MSG_ALL_MSGS = 0xFFFF
};

/** cmdType in sendCommand functions */
enum {
    CAMERA_CMD_START_SMOOTH_ZOOM = 1,
    CAMERA_CMD_STOP_SMOOTH_ZOOM = 2,

    /**
     * Set the clockwise rotation of preview display (setPreviewDisplay) in
     * degrees. This affects the preview frames and the picture displayed after
     * snapshot. This method is useful for portrait mode applications. Note
     * that preview display of front-facing cameras is flipped horizontally
     * before the rotation, that is, the image is reflected along the central
     * vertical axis of the camera sensor. So the users can see themselves as
     * looking into a mirror.
     *
     * This does not affect the order of byte array of
     * CAMERA_MSG_PREVIEW_FRAME, CAMERA_MSG_VIDEO_FRAME,
     * CAMERA_MSG_POSTVIEW_FRAME, CAMERA_MSG_RAW_IMAGE, or
     * CAMERA_MSG_COMPRESSED_IMAGE. This is allowed to be set during preview
     * since API level 14.
     */
    CAMERA_CMD_SET_DISPLAY_ORIENTATION = 3,

    /**
     * cmdType to disable/enable shutter sound. In sendCommand passing arg1 =
     * 0 will disable, while passing arg1 = 1 will enable the shutter sound.
     */
    CAMERA_CMD_ENABLE_SHUTTER_SOUND = 4,

    /* cmdType to play recording sound */
    CAMERA_CMD_PLAY_RECORDING_SOUND = 5,

    /**
     * Start the face detection. This should be called after preview is started.
     * The camera will notify the listener of CAMERA_MSG_FACE and the detected
     * faces in the preview frame. The detected faces may be the same as the
     * previous ones. Apps should call CAMERA_CMD_STOP_FACE_DETECTION to stop
     * the face detection. This method is supported if CameraParameters
     * KEY_MAX_NUM_HW_DETECTED_FACES or KEY_MAX_NUM_SW_DETECTED_FACES is
     * bigger than 0. Hardware and software face detection should not be running
     * at the same time. If the face detection has started, apps should not send
     * this again.
     *
     * In hardware face detection mode, CameraParameters KEY_WHITE_BALANCE,
     * KEY_FOCUS_AREAS and KEY_METERING_AREAS have no effect.
     *
     * arg1 is the face detection type. It can be CAMERA_FACE_DETECTION_HW or
     * CAMERA_FACE_DETECTION_SW. If the type of face detection requested is not
     * supported, the HAL must return BAD_VALUE.
     */
    CAMERA_CMD_START_FACE_DETECTION = 6,

    /**
     * Stop the face detection.
     */
    CAMERA_CMD_STOP_FACE_DETECTION = 7,

    /**
     * Enable/disable focus move callback (CAMERA_MSG_FOCUS_MOVE). Passing
     * arg1 = 0 will disable, while passing arg1 = 1 will enable the callback.
     */
    CAMERA_CMD_ENABLE_FOCUS_MOVE_MSG = 8,

    /**
     * Ping camera service to see if camera hardware is released.
     *
     * When any camera method returns error, the client can use ping command
     * to see if the camera has been taken away by other clients. If the result
     * is NO_ERROR, it means the camera hardware is not released. If the result
     * is not NO_ERROR, the camera has been released and the existing client
     * can silently finish itself or show a dialog.
     */
    CAMERA_CMD_PING = 9,

    /**
     * Configure the number of video buffers used for recording. The intended
     * video buffer count for recording is passed as arg1, which must be
     * greater than 0. This command must be sent before recording is started.
     * This command returns INVALID_OPERATION error if it is sent after video
     * recording is started, or the command is not supported at all. This
     * command also returns a BAD_VALUE error if the intended video buffer
     * count is non-positive or too big to be realized.
     */
    CAMERA_CMD_SET_VIDEO_BUFFER_COUNT = 10,

    /**
     * Configure an explicit format to use for video recording metadata mode.
     * This can be used to switch the format from the
     * default IMPLEMENTATION_DEFINED gralloc format to some other
     * device-supported format, and the default dataspace from the BT_709 color
     * space to some other device-supported dataspace. arg1 is the HAL pixel
     * format, and arg2 is the HAL dataSpace. This command returns
     * INVALID_OPERATION error if it is sent after video recording is started,
     * or the command is not supported at all.
     *
     * If the gralloc format is set to a format other than
     * IMPLEMENTATION_DEFINED, then HALv3 devices will use gralloc usage flags
     * of SW_READ_OFTEN.
     */
    CAMERA_CMD_SET_VIDEO_FORMAT = 11
};

/** camera fatal errors */
enum {
    CAMERA_ERROR_UNKNOWN = 1,
    /**
     * Camera was released because another client has connected to the camera.
     * The original client should call Camera::disconnect immediately after
     * getting this notification. Otherwise, the camera will be released by
     * camera service in a short time. The client should not call any method
     * (except disconnect and sending CAMERA_CMD_PING) after getting this.
     */
    CAMERA_ERROR_RELEASED = 2,

    /**
     * Camera was released because device policy change or the client application
     * is going to background. The client should call Camera::disconnect
     * immediately after getting this notification. Otherwise, the camera will be
     * released by camera service in a short time. The client should not call any
     * method (except disconnect and sending CAMERA_CMD_PING) after getting this.
     */
    CAMERA_ERROR_DISABLED = 3,
    CAMERA_ERROR_SERVER_DIED = 100
};

enum {
    /** The facing of the camera is opposite to that of the screen. */
    CAMERA_FACING_BACK = 0,
    /** The facing of the camera is the same as that of the screen. */
    CAMERA_FACING_FRONT = 1,
    /**
     * The facing of the camera is not fixed relative to the screen.
     * The cameras with this facing are external cameras, e.g. USB cameras.
     */
    CAMERA_FACING_EXTERNAL = 2
};

enum {
    /** Hardware face detection. It does not use much CPU. */
    CAMERA_FACE_DETECTION_HW = 0,
    /**
     * Software face detection. It uses some CPU. Applications must use
     * Camera.setPreviewTexture for preview in this mode.
     */
    CAMERA_FACE_DETECTION_SW = 1
};

/**
 * The information of a face from camera face detection.
 */
typedef struct camera_face {
    /**
     * Bounds of the face [left, top, right, bottom]. (-1000, -1000) represents
     * the top-left of the camera field of view, and (1000, 1000) represents the
     * bottom-right of the field of view. The width and height cannot be 0 or
     * negative. This is supported by both hardware and software face detection.
     *
     * The direction is relative to the sensor orientation, that is, what the
     * sensor sees. The direction is not affected by the rotation or mirroring
     * of CAMERA_CMD_SET_DISPLAY_ORIENTATION.
     */
    int32_t rect[4];

    /**
     * The confidence level of the face. The range is 1 to 100. 100 is the
     * highest confidence. This is supported by both hardware and software
     * face detection.
     */
    int32_t score;

    /**
     * An unique id per face while the face is visible to the tracker. If
     * the face leaves the field-of-view and comes back, it will get a new
     * id. If the value is 0, id is not supported.
     */
    int32_t id;

    /**
     * The coordinates of the center of the left eye. The range is -1000 to
     * 1000. -2000, -2000 if this is not supported.
     */
    int32_t left_eye[2];

    /**
     * The coordinates of the center of the right eye. The range is -1000 to
     * 1000. -2000, -2000 if this is not supported.
     */
    int32_t right_eye[2];

    /**
     * The coordinates of the center of the mouth. The range is -1000 to 1000.
     * -2000, -2000 if this is not supported.
     */
    int32_t mouth[2];

} camera_face_t;

/**
 * The metadata of the frame data.
 */
typedef struct camera_frame_metadata {
    /**
     * The number of detected faces in the frame.
     */
    int32_t number_of_faces;

    /**
     * An array of the detected faces. The length is number_of_faces.
     */
    camera_face_t *faces;
} camera_frame_metadata_t;

__END_DECLS

#endif /* SYSTEM_CORE_INCLUDE_ANDROID_CAMERA_H */
