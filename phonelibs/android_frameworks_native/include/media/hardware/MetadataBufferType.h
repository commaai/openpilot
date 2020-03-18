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

#ifndef METADATA_BUFFER_TYPE_H
#define METADATA_BUFFER_TYPE_H

#ifdef __cplusplus
extern "C" {
namespace android {
#endif

/*
 * MetadataBufferType defines the type of the metadata buffers that
 * can be passed to video encoder component for encoding, via Stagefright
 * media recording framework. To see how to work with the metadata buffers
 * in media recording framework, please consult HardwareAPI.h
 *
 * The creator of metadata buffers and video encoder share common knowledge
 * on what is actually being stored in these metadata buffers, and
 * how the information can be used by the video encoder component
 * to locate the actual pixel data as the source input for video
 * encoder, plus whatever other information that is necessary. Stagefright
 * media recording framework does not need to know anything specific about the
 * metadata buffers, except for receving each individual metadata buffer
 * as the source input, making a copy of the metadata buffer, and passing the
 * copy via OpenMAX API to the video encoder component.
 *
 * The creator of the metadata buffers must ensure that the first
 * 4 bytes in every metadata buffer indicates its buffer type,
 * and the rest of the metadata buffer contains the
 * actual metadata information. When a video encoder component receives
 * a metadata buffer, it uses the first 4 bytes in that buffer to find
 * out the type of the metadata buffer, and takes action appropriate
 * to that type of metadata buffers (for instance, locate the actual
 * pixel data input and then encoding the input data to produce a
 * compressed output buffer).
 *
 * The following shows the layout of a metadata buffer,
 * where buffer type is a 4-byte field of MetadataBufferType,
 * and the payload is the metadata information.
 *
 * --------------------------------------------------------------
 * |  buffer type  |          payload                           |
 * --------------------------------------------------------------
 *
 */
typedef enum {

    /*
     * kMetadataBufferTypeCameraSource is used to indicate that
     * the source of the metadata buffer is the camera component.
     */
    kMetadataBufferTypeCameraSource  = 0,

    /*
     * kMetadataBufferTypeGrallocSource is used to indicate that
     * the payload of the metadata buffers can be interpreted as
     * a buffer_handle_t.
     * So in this case,the metadata that the encoder receives
     * will have a byte stream that consists of two parts:
     * 1. First, there is an integer indicating that it is a GRAlloc
     * source (kMetadataBufferTypeGrallocSource)
     * 2. This is followed by the buffer_handle_t that is a handle to the
     * GRalloc buffer. The encoder needs to interpret this GRalloc handle
     * and encode the frames.
     * --------------------------------------------------------------
     * |  kMetadataBufferTypeGrallocSource | buffer_handle_t buffer |
     * --------------------------------------------------------------
     *
     * See the VideoGrallocMetadata structure.
     */
    kMetadataBufferTypeGrallocSource = 1,

    /*
     * kMetadataBufferTypeGraphicBuffer is used to indicate that
     * the payload of the metadata buffers can be interpreted as
     * an ANativeWindowBuffer, and that a fence is provided.
     *
     * In this case, the metadata will have a byte stream that consists of three parts:
     * 1. First, there is an integer indicating that the metadata
     * contains an ANativeWindowBuffer (kMetadataBufferTypeANWBuffer)
     * 2. This is followed by the pointer to the ANativeWindowBuffer.
     * Codec must not free this buffer as it does not actually own this buffer.
     * 3. Finally, there is an integer containing a fence file descriptor.
     * The codec must wait on the fence before encoding or decoding into this
     * buffer. When the buffer is returned, codec must replace this file descriptor
     * with a new fence, that will be waited on before the buffer is replaced
     * (encoder) or read (decoder).
     * ---------------------------------
     * |  kMetadataBufferTypeANWBuffer |
     * ---------------------------------
     * |  ANativeWindowBuffer *buffer  |
     * ---------------------------------
     * |  int fenceFd                  |
     * ---------------------------------
     *
     * See the VideoNativeMetadata structure.
     */
    kMetadataBufferTypeANWBuffer = 2,

    /* This value is used by framework, but is never used inside a metadata buffer  */
    kMetadataBufferTypeInvalid = -1,


    // Add more here...

} MetadataBufferType;

#ifdef __cplusplus
}  // namespace android
}
#endif

#endif  // METADATA_BUFFER_TYPE_H
