/*
 * Copyright (C) 2010 The Android Open Source Project
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
 * @addtogroup NativeActivity Native Activity
 * @{
 */

/**
 * @file native_window_jni.h
 */

#ifndef ANDROID_NATIVE_WINDOW_JNI_H
#define ANDROID_NATIVE_WINDOW_JNI_H

#include <sys/cdefs.h>

#include <android/native_window.h>

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Return the ANativeWindow associated with a Java Surface object,
 * for interacting with it through native code.  This acquires a reference
 * on the ANativeWindow that is returned; be sure to use ANativeWindow_release()
 * when done with it so that it doesn't leak.
 */
ANativeWindow* ANativeWindow_fromSurface(JNIEnv* env, jobject surface);

#if __ANDROID_API__ >= 13
/**
 * Return the ANativeWindow associated with a Java SurfaceTexture object,
 * for interacting with it through native code.  This acquires a reference
 * on the ANativeWindow that is returned; be sure to use ANativeWindow_release()
 * when done with it so that it doesn't leak.
 */
ANativeWindow* ANativeWindow_fromSurfaceTexture(JNIEnv* env, jobject surfaceTexture);
#endif

#if __ANDROID_API__ >= 26
/**
 * Return a Java Surface object derived from the ANativeWindow, for interacting
 * with it through Java code. The returned Java object acquires a reference on
 * the ANativeWindow; maintains it through general Java object's life cycle;
 * and will automatically release the reference when the Java object gets garbage
 * collected.
 */
jobject ANativeWindow_toSurface(JNIEnv* env, ANativeWindow* window);
#endif

#ifdef __cplusplus
};
#endif

#endif // ANDROID_NATIVE_WINDOW_H

/** @} */
