/*
 * Copyright (C) 2018 The Android Open Source Project
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

/**
 * @addtogroup SurfaceTexture
 * @{
 */

/**
 * @file surface_texture.h
 */

#ifndef ANDROID_NATIVE_SURFACE_TEXTURE_H
#define ANDROID_NATIVE_SURFACE_TEXTURE_H

/******************************************************************
 *
 * IMPORTANT NOTICE:
 *
 *   This file is part of Android's set of stable system headers
 *   exposed by the Android NDK (Native Development Kit).
 *
 *   Third-party source AND binary code relies on the definitions
 *   here to be FROZEN ON ALL UPCOMING PLATFORM RELEASES.
 *
 *   - DO NOT MODIFY ENUMS (EXCEPT IF YOU ADD NEW 32-BIT VALUES)
 *   - DO NOT MODIFY CONSTANTS OR FUNCTIONAL MACROS
 *   - DO NOT CHANGE THE SIGNATURE OF FUNCTIONS IN ANY WAY
 *   - DO NOT CHANGE THE LAYOUT OR SIZE OF STRUCTURES
 */

#include <stdint.h>

#include <android/native_window.h>

__BEGIN_DECLS

struct ASurfaceTexture;

/**
 * {@link ASurfaceTexture} is an opaque type to manage SurfaceTexture from native code
 *
 * {@link ASurfaceTexture} can be obtained from an android.graphics.SurfaceTexture object using
 * ASurfaceTexture_fromSurfaceTexture().
 */
typedef struct ASurfaceTexture ASurfaceTexture;

/**
 * Release the reference to the native ASurfaceTexture acquired with
 * ASurfaceTexture_fromSurfaceTexture().
 * Failing to do so will result in leaked memory and graphic resources.
 * \param st A ASurfaceTexture reference acquired with ASurfaceTexture_fromSurfaceTexture()
 */
void ASurfaceTexture_release(ASurfaceTexture* st);

/**
 * Returns a reference to an ANativeWindow (i.e. the Producer) for this SurfaceTexture.
 * This is equivalent to Java's: Surface sur = new Surface(surfaceTexture);
 *
 * \param st A ASurfaceTexture reference acquired with ASurfaceTexture_fromSurfaceTexture()
 * @return A reference to an ANativeWindow. This reference MUST BE released when no longer needed
 * using ANativeWindow_release(). Failing to do so will result in leaked resources. nullptr is
 * returned if \st is null or if it's not an instance of android.graphics.SurfaceTexture
 */
ANativeWindow* ASurfaceTexture_acquireANativeWindow(ASurfaceTexture* st);

/**
 * Attach the SurfaceTexture to the OpenGL ES context that is current on the calling thread.  A
 * new OpenGL ES texture object is created and populated with the SurfaceTexture image frame
 * that was current at the time of the last call to {@link #detachFromGLContext}.  This new
 * texture is bound to the GL_TEXTURE_EXTERNAL_OES texture target.
 *
 * This can be used to access the SurfaceTexture image contents from multiple OpenGL ES
 * contexts.  Note, however, that the image contents are only accessible from one OpenGL ES
 * context at a time.
 *
 * \param st A ASurfaceTexture reference acquired with ASurfaceTexture_fromSurfaceTexture()
 * \param texName The name of the OpenGL ES texture that will be created.  This texture name
 * must be unusued in the OpenGL ES context that is current on the calling thread.
 * \return 0 on success, negative posix error code otherwise (see <errno.h>)
 */
int ASurfaceTexture_attachToGLContext(ASurfaceTexture* st, uint32_t texName);

/**
 * Detach the SurfaceTexture from the OpenGL ES context that owns the OpenGL ES texture object.
 * This call must be made with the OpenGL ES context current on the calling thread.  The OpenGL
 * ES texture object will be deleted as a result of this call.  After calling this method all
 * calls to {@link #updateTexImage} will fail until a successful call to {@link #attachToGLContext}
 * is made.
 *
 * This can be used to access the SurfaceTexture image contents from multiple OpenGL ES
 * contexts.  Note, however, that the image contents are only accessible from one OpenGL ES
 * context at a time.
 *
 * \param st A ASurfaceTexture reference acquired with ASurfaceTexture_fromSurfaceTexture()
 * \return 0 on success, negative posix error code otherwise (see <errno.h>)
 */
int ASurfaceTexture_detachFromGLContext(ASurfaceTexture* st);

/**
 * Update the texture image to the most recent frame from the image stream.  This may only be
 * called while the OpenGL ES context that owns the texture is current on the calling thread.
 * It will implicitly bind its texture to the GL_TEXTURE_EXTERNAL_OES texture target.
 *
 * \param st A ASurfaceTexture reference acquired with ASurfaceTexture_fromSurfaceTexture()
 * \return 0 on success, negative posix error code otherwise (see <errno.h>)
 */
int ASurfaceTexture_updateTexImage(ASurfaceTexture* st);

/**
 * Retrieve the 4x4 texture coordinate transform matrix associated with the texture image set by
 * the most recent call to updateTexImage.
 *
 * This transform matrix maps 2D homogeneous texture coordinates of the form (s, t, 0, 1) with s
 * and t in the inclusive range [0, 1] to the texture coordinate that should be used to sample
 * that location from the texture.  Sampling the texture outside of the range of this transform
 * is undefined.
 *
 * The matrix is stored in column-major order so that it may be passed directly to OpenGL ES via
 * the glLoadMatrixf or glUniformMatrix4fv functions.
 *
 * \param st A ASurfaceTexture reference acquired with ASurfaceTexture_fromSurfaceTexture()
 * \param mtx the array into which the 4x4 matrix will be stored.  The array must have exactly
 *     16 elements.
 */
void ASurfaceTexture_getTransformMatrix(ASurfaceTexture* st, float mtx[16]);

/**
 * Retrieve the timestamp associated with the texture image set by the most recent call to
 * updateTexImage.
 *
 * This timestamp is in nanoseconds, and is normally monotonically increasing. The timestamp
 * should be unaffected by time-of-day adjustments, and for a camera should be strictly
 * monotonic but for a MediaPlayer may be reset when the position is set.  The
 * specific meaning and zero point of the timestamp depends on the source providing images to
 * the SurfaceTexture. Unless otherwise specified by the image source, timestamps cannot
 * generally be compared across SurfaceTexture instances, or across multiple program
 * invocations. It is mostly useful for determining time offsets between subsequent frames.
 *
 * For EGL/Vulkan producers, this timestamp is the desired present time set with the
 * EGL_ANDROID_presentation_time or VK_GOOGLE_display_timing extensions
 *
 * \param st A ASurfaceTexture reference acquired with ASurfaceTexture_fromSurfaceTexture()
 */
int64_t ASurfaceTexture_getTimestamp(ASurfaceTexture* st);

__END_DECLS

#endif /* ANDROID_NATIVE_SURFACE_TEXTURE_H */
