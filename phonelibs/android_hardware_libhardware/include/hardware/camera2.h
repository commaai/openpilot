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

#ifndef ANDROID_INCLUDE_CAMERA2_H
#define ANDROID_INCLUDE_CAMERA2_H

#include "camera_common.h"
#include "system/camera_metadata.h"

/**
 * Camera device HAL 2.1 [ CAMERA_DEVICE_API_VERSION_2_0, CAMERA_DEVICE_API_VERSION_2_1 ]
 *
 * DEPRECATED. New devices should use Camera HAL v3.2 or newer.
 *
 * Supports the android.hardware.Camera API, and the android.hardware.camera2
 * API in legacy mode only.
 *
 * Camera devices that support this version of the HAL must return
 * CAMERA_DEVICE_API_VERSION_2_1 in camera_device_t.common.version and in
 * camera_info_t.device_version (from camera_module_t.get_camera_info).
 *
 * Camera modules that may contain version 2.x devices must implement at least
 * version 2.0 of the camera module interface (as defined by
 * camera_module_t.common.module_api_version).
 *
 * See camera_common.h for more versioning details.
 *
 * Version history:
 *
 * 2.0: CAMERA_DEVICE_API_VERSION_2_0. Initial release (Android 4.2):
 *      - Sufficient for implementing existing android.hardware.Camera API.
 *      - Allows for ZSL queue in camera service layer
 *      - Not tested for any new features such manual capture control,
 *        Bayer RAW capture, reprocessing of RAW data.
 *
 * 2.1: CAMERA_DEVICE_API_VERSION_2_1. Support per-device static metadata:
 *      - Add get_instance_metadata() method to retrieve metadata that is fixed
 *        after device open, but may be variable between open() calls.
 */

__BEGIN_DECLS

struct camera2_device;

/**********************************************************************
 *
 * Input/output stream buffer queue interface definitions
 *
 */

/**
 * Output image stream queue interface. A set of these methods is provided to
 * the HAL device in allocate_stream(), and are used to interact with the
 * gralloc buffer queue for that stream. They may not be called until after
 * allocate_stream returns.
 */
typedef struct camera2_stream_ops {
    /**
     * Get a buffer to fill from the queue. The size and format of the buffer
     * are fixed for a given stream (defined in allocate_stream), and the stride
     * should be queried from the platform gralloc module. The gralloc buffer
     * will have been allocated based on the usage flags provided by
     * allocate_stream, and will be locked for use.
     */
    int (*dequeue_buffer)(const struct camera2_stream_ops* w,
            buffer_handle_t** buffer);

    /**
     * Push a filled buffer to the stream to be used by the consumer.
     *
     * The timestamp represents the time at start of exposure of the first row
     * of the image; it must be from a monotonic clock, and is measured in
     * nanoseconds. The timestamps do not need to be comparable between
     * different cameras, or consecutive instances of the same camera. However,
     * they must be comparable between streams from the same camera. If one
     * capture produces buffers for multiple streams, each stream must have the
     * same timestamp for that buffer, and that timestamp must match the
     * timestamp in the output frame metadata.
     */
    int (*enqueue_buffer)(const struct camera2_stream_ops* w,
            int64_t timestamp,
            buffer_handle_t* buffer);
    /**
     * Return a buffer to the queue without marking it as filled.
     */
    int (*cancel_buffer)(const struct camera2_stream_ops* w,
            buffer_handle_t* buffer);
    /**
     * Set the crop window for subsequently enqueued buffers. The parameters are
     * measured in pixels relative to the buffer width and height.
     */
    int (*set_crop)(const struct camera2_stream_ops *w,
            int left, int top, int right, int bottom);

} camera2_stream_ops_t;

/**
 * Temporary definition during transition.
 *
 * These formats will be removed and replaced with
 * HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED.  To maximize forward compatibility,
 * HAL implementations are strongly recommended to treat FORMAT_OPAQUE and
 * FORMAT_ZSL as equivalent to HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED, and
 * return HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED in the format_actual output
 * parameter of allocate_stream, allowing the gralloc module to select the
 * specific format based on the usage flags from the camera and the stream
 * consumer.
 */
enum {
    CAMERA2_HAL_PIXEL_FORMAT_OPAQUE = HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED,
    CAMERA2_HAL_PIXEL_FORMAT_ZSL = -1
};

/**
 * Transport header for compressed JPEG buffers in output streams.
 *
 * To capture JPEG images, a stream is created using the pixel format
 * HAL_PIXEL_FORMAT_BLOB, and the static metadata field android.jpeg.maxSize is
 * used as the buffer size. Since compressed JPEG images are of variable size,
 * the HAL needs to include the final size of the compressed image using this
 * structure inside the output stream buffer. The JPEG blob ID field must be set
 * to CAMERA2_JPEG_BLOB_ID.
 *
 * Transport header should be at the end of the JPEG output stream buffer.  That
 * means the jpeg_blob_id must start at byte[android.jpeg.maxSize -
 * sizeof(camera2_jpeg_blob)].  Any HAL using this transport header must
 * account for it in android.jpeg.maxSize.  The JPEG data itself starts at
 * byte[0] and should be jpeg_size bytes long.
 */
typedef struct camera2_jpeg_blob {
    uint16_t jpeg_blob_id;
    uint32_t jpeg_size;
};

enum {
    CAMERA2_JPEG_BLOB_ID = 0x00FF
};

/**
 * Input reprocess stream queue management. A set of these methods is provided
 * to the HAL device in allocate_reprocess_stream(); they are used to interact
 * with the reprocess stream's input gralloc buffer queue.
 */
typedef struct camera2_stream_in_ops {
    /**
     * Get the next buffer of image data to reprocess. The width, height, and
     * format of the buffer is fixed in allocate_reprocess_stream(), and the
     * stride and other details should be queried from the platform gralloc
     * module as needed. The buffer will already be locked for use.
     */
    int (*acquire_buffer)(const struct camera2_stream_in_ops *w,
            buffer_handle_t** buffer);
    /**
     * Return a used buffer to the buffer queue for reuse.
     */
    int (*release_buffer)(const struct camera2_stream_in_ops *w,
            buffer_handle_t* buffer);

} camera2_stream_in_ops_t;

/**********************************************************************
 *
 * Metadata queue management, used for requests sent to HAL module, and for
 * frames produced by the HAL.
 *
 */

enum {
    CAMERA2_REQUEST_QUEUE_IS_BOTTOMLESS = -1
};

/**
 * Request input queue protocol:
 *
 * The framework holds the queue and its contents. At start, the queue is empty.
 *
 * 1. When the first metadata buffer is placed into the queue, the framework
 *    signals the device by calling notify_request_queue_not_empty().
 *
 * 2. After receiving notify_request_queue_not_empty, the device must call
 *    dequeue() once it's ready to handle the next buffer.
 *
 * 3. Once the device has processed a buffer, and is ready for the next buffer,
 *    it must call dequeue() again instead of waiting for a notification. If
 *    there are no more buffers available, dequeue() will return NULL. After
 *    this point, when a buffer becomes available, the framework must call
 *    notify_request_queue_not_empty() again. If the device receives a NULL
 *    return from dequeue, it does not need to query the queue again until a
 *    notify_request_queue_not_empty() call is received from the source.
 *
 * 4. If the device calls buffer_count() and receives 0, this does not mean that
 *    the framework will provide a notify_request_queue_not_empty() call. The
 *    framework will only provide such a notification after the device has
 *    received a NULL from dequeue, or on initial startup.
 *
 * 5. The dequeue() call in response to notify_request_queue_not_empty() may be
 *    on the same thread as the notify_request_queue_not_empty() call, and may
 *    be performed from within the notify call.
 *
 * 6. All dequeued request buffers must be returned to the framework by calling
 *    free_request, including when errors occur, a device flush is requested, or
 *    when the device is shutting down.
 */
typedef struct camera2_request_queue_src_ops {
    /**
     * Get the count of request buffers pending in the queue. May return
     * CAMERA2_REQUEST_QUEUE_IS_BOTTOMLESS if a repeating request (stream
     * request) is currently configured. Calling this method has no effect on
     * whether the notify_request_queue_not_empty() method will be called by the
     * framework.
     */
    int (*request_count)(const struct camera2_request_queue_src_ops *q);

    /**
     * Get a metadata buffer from the framework. Returns OK if there is no
     * error. If the queue is empty, returns NULL in buffer. In that case, the
     * device must wait for a notify_request_queue_not_empty() message before
     * attempting to dequeue again. Buffers obtained in this way must be
     * returned to the framework with free_request().
     */
    int (*dequeue_request)(const struct camera2_request_queue_src_ops *q,
            camera_metadata_t **buffer);
    /**
     * Return a metadata buffer to the framework once it has been used, or if
     * an error or shutdown occurs.
     */
    int (*free_request)(const struct camera2_request_queue_src_ops *q,
            camera_metadata_t *old_buffer);

} camera2_request_queue_src_ops_t;

/**
 * Frame output queue protocol:
 *
 * The framework holds the queue and its contents. At start, the queue is empty.
 *
 * 1. When the device is ready to fill an output metadata frame, it must dequeue
 *    a metadata buffer of the required size.
 *
 * 2. It should then fill the metadata buffer, and place it on the frame queue
 *    using enqueue_frame. The framework takes ownership of the frame.
 *
 * 3. In case of an error, a request to flush the pipeline, or shutdown, the
 *    device must return any affected dequeued frames to the framework by
 *    calling cancel_frame.
 */
typedef struct camera2_frame_queue_dst_ops {
    /**
     * Get an empty metadata buffer to fill from the framework. The new metadata
     * buffer will have room for entries number of metadata entries, plus
     * data_bytes worth of extra storage. Frames dequeued here must be returned
     * to the framework with either cancel_frame or enqueue_frame.
     */
    int (*dequeue_frame)(const struct camera2_frame_queue_dst_ops *q,
            size_t entries, size_t data_bytes,
            camera_metadata_t **buffer);

    /**
     * Return a dequeued metadata buffer to the framework for reuse; do not mark it as
     * filled. Use when encountering errors, or flushing the internal request queue.
     */
    int (*cancel_frame)(const struct camera2_frame_queue_dst_ops *q,
            camera_metadata_t *buffer);

    /**
     * Place a completed metadata frame on the frame output queue.
     */
    int (*enqueue_frame)(const struct camera2_frame_queue_dst_ops *q,
            camera_metadata_t *buffer);

} camera2_frame_queue_dst_ops_t;

/**********************************************************************
 *
 * Notification callback and message definition, and trigger definitions
 *
 */

/**
 * Asynchronous notification callback from the HAL, fired for various
 * reasons. Only for information independent of frame capture, or that require
 * specific timing. The user pointer must be the same one that was passed to the
 * device in set_notify_callback().
 */
typedef void (*camera2_notify_callback)(int32_t msg_type,
        int32_t ext1,
        int32_t ext2,
        int32_t ext3,
        void *user);

/**
 * Possible message types for camera2_notify_callback
 */
enum {
    /**
     * An error has occurred. Argument ext1 contains the error code, and
     * ext2 and ext3 contain any error-specific information.
     */
    CAMERA2_MSG_ERROR   = 0x0001,
    /**
     * The exposure of a given request has begun. Argument ext1 contains the
     * frame number, and ext2 and ext3 contain the low-order and high-order
     * bytes of the timestamp for when exposure began.
     * (timestamp = (ext3 << 32 | ext2))
     */
    CAMERA2_MSG_SHUTTER = 0x0010,
    /**
     * The autofocus routine has changed state. Argument ext1 contains the new
     * state; the values are the same as those for the metadata field
     * android.control.afState. Ext2 contains the latest trigger ID passed to
     * trigger_action(CAMERA2_TRIGGER_AUTOFOCUS) or
     * trigger_action(CAMERA2_TRIGGER_CANCEL_AUTOFOCUS), or 0 if trigger has not
     * been called with either of those actions.
     */
    CAMERA2_MSG_AUTOFOCUS = 0x0020,
    /**
     * The autoexposure routine has changed state. Argument ext1 contains the
     * new state; the values are the same as those for the metadata field
     * android.control.aeState. Ext2 contains the latest trigger ID value passed to
     * trigger_action(CAMERA2_TRIGGER_PRECAPTURE_METERING), or 0 if that method
     * has not been called.
     */
    CAMERA2_MSG_AUTOEXPOSURE = 0x0021,
    /**
     * The auto-whitebalance routine has changed state. Argument ext1 contains
     * the new state; the values are the same as those for the metadata field
     * android.control.awbState. Ext2 contains the latest trigger ID passed to
     * trigger_action(CAMERA2_TRIGGER_PRECAPTURE_METERING), or 0 if that method
     * has not been called.
     */
    CAMERA2_MSG_AUTOWB = 0x0022
};

/**
 * Error codes for CAMERA_MSG_ERROR
 */
enum {
    /**
     * A serious failure occured. Camera device may not work without reboot, and
     * no further frames or buffer streams will be produced by the
     * device. Device should be treated as closed.
     */
    CAMERA2_MSG_ERROR_HARDWARE = 0x0001,
    /**
     * A serious failure occured. No further frames or buffer streams will be
     * produced by the device. Device should be treated as closed. The client
     * must reopen the device to use it again.
     */
    CAMERA2_MSG_ERROR_DEVICE,
    /**
     * An error has occurred in processing a request. No output (metadata or
     * buffers) will be produced for this request. ext2 contains the frame
     * number of the request. Subsequent requests are unaffected, and the device
     * remains operational.
     */
    CAMERA2_MSG_ERROR_REQUEST,
    /**
     * An error has occurred in producing an output frame metadata buffer for a
     * request, but image buffers for it will still be available. Subsequent
     * requests are unaffected, and the device remains operational. ext2
     * contains the frame number of the request.
     */
    CAMERA2_MSG_ERROR_FRAME,
    /**
     * An error has occurred in placing an output buffer into a stream for a
     * request. The frame metadata and other buffers may still be
     * available. Subsequent requests are unaffected, and the device remains
     * operational. ext2 contains the frame number of the request, and ext3
     * contains the stream id.
     */
    CAMERA2_MSG_ERROR_STREAM,
    /**
     * Number of error types
     */
    CAMERA2_MSG_NUM_ERRORS
};

/**
 * Possible trigger ids for trigger_action()
 */
enum {
    /**
     * Trigger an autofocus cycle. The effect of the trigger depends on the
     * autofocus mode in effect when the trigger is received, which is the mode
     * listed in the latest capture request to be dequeued by the HAL. If the
     * mode is OFF, EDOF, or FIXED, the trigger has no effect. In AUTO, MACRO,
     * or CONTINUOUS_* modes, see below for the expected behavior. The state of
     * the autofocus cycle can be tracked in android.control.afMode and the
     * corresponding notifications.
     *
     **
     * In AUTO or MACRO mode, the AF state transitions (and notifications)
     * when calling with trigger ID = N with the previous ID being K are:
     *
     * Initial state       Transitions
     * INACTIVE (K)         -> ACTIVE_SCAN (N) -> AF_FOCUSED (N) or AF_NOT_FOCUSED (N)
     * AF_FOCUSED (K)       -> ACTIVE_SCAN (N) -> AF_FOCUSED (N) or AF_NOT_FOCUSED (N)
     * AF_NOT_FOCUSED (K)   -> ACTIVE_SCAN (N) -> AF_FOCUSED (N) or AF_NOT_FOCUSED (N)
     * ACTIVE_SCAN (K)      -> AF_FOCUSED(N) or AF_NOT_FOCUSED(N)
     * PASSIVE_SCAN (K)      Not used in AUTO/MACRO mode
     * PASSIVE_FOCUSED (K)   Not used in AUTO/MACRO mode
     *
     **
     * In CONTINUOUS_PICTURE mode, triggering AF must lock the AF to the current
     * lens position and transition the AF state to either AF_FOCUSED or
     * NOT_FOCUSED. If a passive scan is underway, that scan must complete and
     * then lock the lens position and change AF state. TRIGGER_CANCEL_AUTOFOCUS
     * will allow the AF to restart its operation.
     *
     * Initial state      Transitions
     * INACTIVE (K)        -> immediate AF_FOCUSED (N) or AF_NOT_FOCUSED (N)
     * PASSIVE_FOCUSED (K) -> immediate AF_FOCUSED (N) or AF_NOT_FOCUSED (N)
     * PASSIVE_SCAN (K)    -> AF_FOCUSED (N) or AF_NOT_FOCUSED (N)
     * AF_FOCUSED (K)      no effect except to change next notification ID to N
     * AF_NOT_FOCUSED (K)  no effect except to change next notification ID to N
     *
     **
     * In CONTINUOUS_VIDEO mode, triggering AF must lock the AF to the current
     * lens position and transition the AF state to either AF_FOCUSED or
     * NOT_FOCUSED. If a passive scan is underway, it must immediately halt, in
     * contrast with CONTINUOUS_PICTURE mode. TRIGGER_CANCEL_AUTOFOCUS will
     * allow the AF to restart its operation.
     *
     * Initial state      Transitions
     * INACTIVE (K)        -> immediate AF_FOCUSED (N) or AF_NOT_FOCUSED (N)
     * PASSIVE_FOCUSED (K) -> immediate AF_FOCUSED (N) or AF_NOT_FOCUSED (N)
     * PASSIVE_SCAN (K)    -> immediate AF_FOCUSED (N) or AF_NOT_FOCUSED (N)
     * AF_FOCUSED (K)      no effect except to change next notification ID to N
     * AF_NOT_FOCUSED (K)  no effect except to change next notification ID to N
     *
     * Ext1 is an ID that must be returned in subsequent auto-focus state change
     * notifications through camera2_notify_callback() and stored in
     * android.control.afTriggerId.
     */
    CAMERA2_TRIGGER_AUTOFOCUS = 0x0001,
    /**
     * Send a cancel message to the autofocus algorithm. The effect of the
     * cancellation depends on the autofocus mode in effect when the trigger is
     * received, which is the mode listed in the latest capture request to be
     * dequeued by the HAL. If the AF mode is OFF or EDOF, the cancel has no
     * effect.  For other modes, the lens should return to its default position,
     * any current autofocus scan must be canceled, and the AF state should be
     * set to INACTIVE.
     *
     * The state of the autofocus cycle can be tracked in android.control.afMode
     * and the corresponding notification. Continuous autofocus modes may resume
     * focusing operations thereafter exactly as if the camera had just been set
     * to a continuous AF mode.
     *
     * Ext1 is an ID that must be returned in subsequent auto-focus state change
     * notifications through camera2_notify_callback() and stored in
     * android.control.afTriggerId.
     */
    CAMERA2_TRIGGER_CANCEL_AUTOFOCUS,
    /**
     * Trigger a pre-capture metering cycle, which may include firing the flash
     * to determine proper capture parameters. Typically, this trigger would be
     * fired for a half-depress of a camera shutter key, or before a snapshot
     * capture in general. The state of the metering cycle can be tracked in
     * android.control.aeMode and the corresponding notification.  If the
     * auto-exposure mode is OFF, the trigger does nothing.
     *
     * Ext1 is an ID that must be returned in subsequent
     * auto-exposure/auto-white balance state change notifications through
     * camera2_notify_callback() and stored in android.control.aePrecaptureId.
     */
     CAMERA2_TRIGGER_PRECAPTURE_METERING
};

/**
 * Possible template types for construct_default_request()
 */
enum {
    /**
     * Standard camera preview operation with 3A on auto.
     */
    CAMERA2_TEMPLATE_PREVIEW = 1,
    /**
     * Standard camera high-quality still capture with 3A and flash on auto.
     */
    CAMERA2_TEMPLATE_STILL_CAPTURE,
    /**
     * Standard video recording plus preview with 3A on auto, torch off.
     */
    CAMERA2_TEMPLATE_VIDEO_RECORD,
    /**
     * High-quality still capture while recording video. Application will
     * include preview, video record, and full-resolution YUV or JPEG streams in
     * request. Must not cause stuttering on video stream. 3A on auto.
     */
    CAMERA2_TEMPLATE_VIDEO_SNAPSHOT,
    /**
     * Zero-shutter-lag mode. Application will request preview and
     * full-resolution data for each frame, and reprocess it to JPEG when a
     * still image is requested by user. Settings should provide highest-quality
     * full-resolution images without compromising preview frame rate. 3A on
     * auto.
     */
    CAMERA2_TEMPLATE_ZERO_SHUTTER_LAG,

    /* Total number of templates */
    CAMERA2_TEMPLATE_COUNT
};


/**********************************************************************
 *
 * Camera device operations
 *
 */
typedef struct camera2_device_ops {

    /**********************************************************************
     * Request and frame queue setup and management methods
     */

    /**
     * Pass in input request queue interface methods.
     */
    int (*set_request_queue_src_ops)(const struct camera2_device *,
            const camera2_request_queue_src_ops_t *request_src_ops);

    /**
     * Notify device that the request queue is no longer empty. Must only be
     * called when the first buffer is added a new queue, or after the source
     * has returned NULL in response to a dequeue call.
     */
    int (*notify_request_queue_not_empty)(const struct camera2_device *);

    /**
     * Pass in output frame queue interface methods
     */
    int (*set_frame_queue_dst_ops)(const struct camera2_device *,
            const camera2_frame_queue_dst_ops_t *frame_dst_ops);

    /**
     * Number of camera requests being processed by the device at the moment
     * (captures/reprocesses that have had their request dequeued, but have not
     * yet been enqueued onto output pipeline(s) ). No streams may be released
     * by the framework until the in-progress count is 0.
     */
    int (*get_in_progress_count)(const struct camera2_device *);

    /**
     * Flush all in-progress captures. This includes all dequeued requests
     * (regular or reprocessing) that have not yet placed any outputs into a
     * stream or the frame queue. Partially completed captures must be completed
     * normally. No new requests may be dequeued from the request queue until
     * the flush completes.
     */
    int (*flush_captures_in_progress)(const struct camera2_device *);

    /**
     * Create a filled-in default request for standard camera use cases.
     *
     * The device must return a complete request that is configured to meet the
     * requested use case, which must be one of the CAMERA2_TEMPLATE_*
     * enums. All request control fields must be included, except for
     * android.request.outputStreams.
     *
     * The metadata buffer returned must be allocated with
     * allocate_camera_metadata. The framework takes ownership of the buffer.
     */
    int (*construct_default_request)(const struct camera2_device *,
            int request_template,
            camera_metadata_t **request);

    /**********************************************************************
     * Stream management
     */

    /**
     * allocate_stream:
     *
     * Allocate a new output stream for use, defined by the output buffer width,
     * height, target, and possibly the pixel format.  Returns the new stream's
     * ID, gralloc usage flags, minimum queue buffer count, and possibly the
     * pixel format, on success. Error conditions:
     *
     *  - Requesting a width/height/format combination not listed as
     *    supported by the sensor's static characteristics
     *
     *  - Asking for too many streams of a given format type (2 bayer raw
     *    streams, for example).
     *
     * Input parameters:
     *
     * - width, height, format: Specification for the buffers to be sent through
     *   this stream. Format is a value from the HAL_PIXEL_FORMAT_* list. If
     *   HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED is used, then the platform
     *   gralloc module will select a format based on the usage flags provided
     *   by the camera HAL and the consumer of the stream. The camera HAL should
     *   inspect the buffers handed to it in the register_stream_buffers call to
     *   obtain the implementation-specific format if necessary.
     *
     * - stream_ops: A structure of function pointers for obtaining and queuing
     *   up buffers for this stream. The underlying stream will be configured
     *   based on the usage and max_buffers outputs. The methods in this
     *   structure may not be called until after allocate_stream returns.
     *
     * Output parameters:
     *
     * - stream_id: An unsigned integer identifying this stream. This value is
     *   used in incoming requests to identify the stream, and in releasing the
     *   stream.
     *
     * - usage: The gralloc usage mask needed by the HAL device for producing
     *   the requested type of data. This is used in allocating new gralloc
     *   buffers for the stream buffer queue.
     *
     * - max_buffers: The maximum number of buffers the HAL device may need to
     *   have dequeued at the same time. The device may not dequeue more buffers
     *   than this value at the same time.
     *
     */
    int (*allocate_stream)(
            const struct camera2_device *,
            // inputs
            uint32_t width,
            uint32_t height,
            int      format,
            const camera2_stream_ops_t *stream_ops,
            // outputs
            uint32_t *stream_id,
            uint32_t *format_actual, // IGNORED, will be removed
            uint32_t *usage,
            uint32_t *max_buffers);

    /**
     * Register buffers for a given stream. This is called after a successful
     * allocate_stream call, and before the first request referencing the stream
     * is enqueued. This method is intended to allow the HAL device to map or
     * otherwise prepare the buffers for later use. num_buffers is guaranteed to
     * be at least max_buffers (from allocate_stream), but may be larger. The
     * buffers will already be locked for use. At the end of the call, all the
     * buffers must be ready to be returned to the queue. If the stream format
     * was set to HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED, the camera HAL should
     * inspect the passed-in buffers here to determine any platform-private
     * pixel format information.
     */
    int (*register_stream_buffers)(
            const struct camera2_device *,
            uint32_t stream_id,
            int num_buffers,
            buffer_handle_t *buffers);

    /**
     * Release a stream. Returns an error if called when get_in_progress_count
     * is non-zero, or if the stream id is invalid.
     */
    int (*release_stream)(
            const struct camera2_device *,
            uint32_t stream_id);

    /**
     * allocate_reprocess_stream:
     *
     * Allocate a new input stream for use, defined by the output buffer width,
     * height, and the pixel format.  Returns the new stream's ID, gralloc usage
     * flags, and required simultaneously acquirable buffer count, on
     * success. Error conditions:
     *
     *  - Requesting a width/height/format combination not listed as
     *    supported by the sensor's static characteristics
     *
     *  - Asking for too many reprocessing streams to be configured at once.
     *
     * Input parameters:
     *
     * - width, height, format: Specification for the buffers to be sent through
     *   this stream. Format must be a value from the HAL_PIXEL_FORMAT_* list.
     *
     * - reprocess_stream_ops: A structure of function pointers for acquiring
     *   and releasing buffers for this stream. The underlying stream will be
     *   configured based on the usage and max_buffers outputs.
     *
     * Output parameters:
     *
     * - stream_id: An unsigned integer identifying this stream. This value is
     *   used in incoming requests to identify the stream, and in releasing the
     *   stream. These ids are numbered separately from the input stream ids.
     *
     * - consumer_usage: The gralloc usage mask needed by the HAL device for
     *   consuming the requested type of data. This is used in allocating new
     *   gralloc buffers for the stream buffer queue.
     *
     * - max_buffers: The maximum number of buffers the HAL device may need to
     *   have acquired at the same time. The device may not have more buffers
     *   acquired at the same time than this value.
     *
     */
    int (*allocate_reprocess_stream)(const struct camera2_device *,
            uint32_t width,
            uint32_t height,
            uint32_t format,
            const camera2_stream_in_ops_t *reprocess_stream_ops,
            // outputs
            uint32_t *stream_id,
            uint32_t *consumer_usage,
            uint32_t *max_buffers);

    /**
     * allocate_reprocess_stream_from_stream:
     *
     * Allocate a new input stream for use, which will use the buffers allocated
     * for an existing output stream. That is, after the HAL enqueues a buffer
     * onto the output stream, it may see that same buffer handed to it from
     * this input reprocessing stream. After the HAL releases the buffer back to
     * the reprocessing stream, it will be returned to the output queue for
     * reuse.
     *
     * Error conditions:
     *
     * - Using an output stream of unsuitable size/format for the basis of the
     *   reprocessing stream.
     *
     * - Attempting to allocatee too many reprocessing streams at once.
     *
     * Input parameters:
     *
     * - output_stream_id: The ID of an existing output stream which has
     *   a size and format suitable for reprocessing.
     *
     * - reprocess_stream_ops: A structure of function pointers for acquiring
     *   and releasing buffers for this stream. The underlying stream will use
     *   the same graphics buffer handles as the output stream uses.
     *
     * Output parameters:
     *
     * - stream_id: An unsigned integer identifying this stream. This value is
     *   used in incoming requests to identify the stream, and in releasing the
     *   stream. These ids are numbered separately from the input stream ids.
     *
     * The HAL client must always release the reprocessing stream before it
     * releases the output stream it is based on.
     *
     */
    int (*allocate_reprocess_stream_from_stream)(const struct camera2_device *,
            uint32_t output_stream_id,
            const camera2_stream_in_ops_t *reprocess_stream_ops,
            // outputs
            uint32_t *stream_id);

    /**
     * Release a reprocessing stream. Returns an error if called when
     * get_in_progress_count is non-zero, or if the stream id is not
     * valid.
     */
    int (*release_reprocess_stream)(
            const struct camera2_device *,
            uint32_t stream_id);

    /**********************************************************************
     * Miscellaneous methods
     */

    /**
     * Trigger asynchronous activity. This is used for triggering special
     * behaviors of the camera 3A routines when they are in use. See the
     * documentation for CAMERA2_TRIGGER_* above for details of the trigger ids
     * and their arguments.
     */
    int (*trigger_action)(const struct camera2_device *,
            uint32_t trigger_id,
            int32_t ext1,
            int32_t ext2);

    /**
     * Notification callback setup
     */
    int (*set_notify_callback)(const struct camera2_device *,
            camera2_notify_callback notify_cb,
            void *user);

    /**
     * Get methods to query for vendor extension metadata tag infomation. May
     * set ops to NULL if no vendor extension tags are defined.
     */
    int (*get_metadata_vendor_tag_ops)(const struct camera2_device*,
            vendor_tag_query_ops_t **ops);

    /**
     * Dump state of the camera hardware
     */
    int (*dump)(const struct camera2_device *, int fd);

    /**
     * Get device-instance-specific metadata. This metadata must be constant for
     * a single instance of the camera device, but may be different between
     * open() calls. The returned camera_metadata pointer must be valid until
     * the device close() method is called.
     *
     * Version information:
     *
     * CAMERA_DEVICE_API_VERSION_2_0:
     *
     *   Not available. Framework may not access this function pointer.
     *
     * CAMERA_DEVICE_API_VERSION_2_1:
     *
     *   Valid. Can be called by the framework.
     *
     */
    int (*get_instance_metadata)(const struct camera2_device *,
            camera_metadata **instance_metadata);

} camera2_device_ops_t;

/**********************************************************************
 *
 * Camera device definition
 *
 */
typedef struct camera2_device {
    /**
     * common.version must equal CAMERA_DEVICE_API_VERSION_2_0 to identify
     * this device as implementing version 2.0 of the camera device HAL.
     */
    hw_device_t common;
    camera2_device_ops_t *ops;
    void *priv;
} camera2_device_t;

__END_DECLS

#endif /* #ifdef ANDROID_INCLUDE_CAMERA2_H */
