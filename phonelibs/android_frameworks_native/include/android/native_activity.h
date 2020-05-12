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
 * @file native_activity.h
 */

#ifndef ANDROID_NATIVE_ACTIVITY_H
#define ANDROID_NATIVE_ACTIVITY_H

#include <stdint.h>
#include <sys/types.h>

#include <jni.h>

#include <android/asset_manager.h>
#include <android/input.h>
#include <android/native_window.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * {@link ANativeActivityCallbacks}
 */
struct ANativeActivityCallbacks;

/**
 * This structure defines the native side of an android.app.NativeActivity.
 * It is created by the framework, and handed to the application's native
 * code as it is being launched.
 */
typedef struct ANativeActivity {
    /**
     * Pointer to the callback function table of the native application.
     * You can set the functions here to your own callbacks.  The callbacks
     * pointer itself here should not be changed; it is allocated and managed
     * for you by the framework.
     */
    struct ANativeActivityCallbacks* callbacks;

    /**
     * The global handle on the process's Java VM.
     */
    JavaVM* vm;

    /**
     * JNI context for the main thread of the app.  Note that this field
     * can ONLY be used from the main thread of the process; that is, the
     * thread that calls into the ANativeActivityCallbacks.
     */
    JNIEnv* env;

    /**
     * The NativeActivity object handle.
     *
     * IMPORTANT NOTE: This member is mis-named. It should really be named
     * 'activity' instead of 'clazz', since it's a reference to the
     * NativeActivity instance created by the system for you.
     *
     * We unfortunately cannot change this without breaking NDK
     * source-compatibility.
     */
    jobject clazz;

    /**
     * Path to this application's internal data directory.
     */
    const char* internalDataPath;

    /**
     * Path to this application's external (removable/mountable) data directory.
     */
    const char* externalDataPath;

    /**
     * The platform's SDK version code.
     */
    int32_t sdkVersion;

    /**
     * This is the native instance of the application.  It is not used by
     * the framework, but can be set by the application to its own instance
     * state.
     */
    void* instance;

    /**
     * Pointer to the Asset Manager instance for the application.  The application
     * uses this to access binary assets bundled inside its own .apk file.
     */
    AAssetManager* assetManager;

    /**
     * Available starting with Honeycomb: path to the directory containing
     * the application's OBB files (if any).  If the app doesn't have any
     * OBB files, this directory may not exist.
     */
    const char* obbPath;
} ANativeActivity;

/**
 * These are the callbacks the framework makes into a native application.
 * All of these callbacks happen on the main thread of the application.
 * By default, all callbacks are NULL; set to a pointer to your own function
 * to have it called.
 */
typedef struct ANativeActivityCallbacks {
    /**
     * NativeActivity has started.  See Java documentation for Activity.onStart()
     * for more information.
     */
    void (*onStart)(ANativeActivity* activity);

    /**
     * NativeActivity has resumed.  See Java documentation for Activity.onResume()
     * for more information.
     */
    void (*onResume)(ANativeActivity* activity);

    /**
     * Framework is asking NativeActivity to save its current instance state.
     * See Java documentation for Activity.onSaveInstanceState() for more
     * information.  The returned pointer needs to be created with malloc();
     * the framework will call free() on it for you.  You also must fill in
     * outSize with the number of bytes in the allocation.  Note that the
     * saved state will be persisted, so it can not contain any active
     * entities (pointers to memory, file descriptors, etc).
     */
    void* (*onSaveInstanceState)(ANativeActivity* activity, size_t* outSize);

    /**
     * NativeActivity has paused.  See Java documentation for Activity.onPause()
     * for more information.
     */
    void (*onPause)(ANativeActivity* activity);

    /**
     * NativeActivity has stopped.  See Java documentation for Activity.onStop()
     * for more information.
     */
    void (*onStop)(ANativeActivity* activity);

    /**
     * NativeActivity is being destroyed.  See Java documentation for Activity.onDestroy()
     * for more information.
     */
    void (*onDestroy)(ANativeActivity* activity);

    /**
     * Focus has changed in this NativeActivity's window.  This is often used,
     * for example, to pause a game when it loses input focus.
     */
    void (*onWindowFocusChanged)(ANativeActivity* activity, int hasFocus);

    /**
     * The drawing window for this native activity has been created.  You
     * can use the given native window object to start drawing.
     */
    void (*onNativeWindowCreated)(ANativeActivity* activity, ANativeWindow* window);

    /**
     * The drawing window for this native activity has been resized.  You should
     * retrieve the new size from the window and ensure that your rendering in
     * it now matches.
     */
    void (*onNativeWindowResized)(ANativeActivity* activity, ANativeWindow* window);

    /**
     * The drawing window for this native activity needs to be redrawn.  To avoid
     * transient artifacts during screen changes (such resizing after rotation),
     * applications should not return from this function until they have finished
     * drawing their window in its current state.
     */
    void (*onNativeWindowRedrawNeeded)(ANativeActivity* activity, ANativeWindow* window);

    /**
     * The drawing window for this native activity is going to be destroyed.
     * You MUST ensure that you do not touch the window object after returning
     * from this function: in the common case of drawing to the window from
     * another thread, that means the implementation of this callback must
     * properly synchronize with the other thread to stop its drawing before
     * returning from here.
     */
    void (*onNativeWindowDestroyed)(ANativeActivity* activity, ANativeWindow* window);

    /**
     * The input queue for this native activity's window has been created.
     * You can use the given input queue to start retrieving input events.
     */
    void (*onInputQueueCreated)(ANativeActivity* activity, AInputQueue* queue);

    /**
     * The input queue for this native activity's window is being destroyed.
     * You should no longer try to reference this object upon returning from this
     * function.
     */
    void (*onInputQueueDestroyed)(ANativeActivity* activity, AInputQueue* queue);

    /**
     * The rectangle in the window in which content should be placed has changed.
     */
    void (*onContentRectChanged)(ANativeActivity* activity, const ARect* rect);

    /**
     * The current device AConfiguration has changed.  The new configuration can
     * be retrieved from assetManager.
     */
    void (*onConfigurationChanged)(ANativeActivity* activity);

    /**
     * The system is running low on memory.  Use this callback to release
     * resources you do not need, to help the system avoid killing more
     * important processes.
     */
    void (*onLowMemory)(ANativeActivity* activity);
} ANativeActivityCallbacks;

/**
 * This is the function that must be in the native code to instantiate the
 * application's native activity.  It is called with the activity instance (see
 * above); if the code is being instantiated from a previously saved instance,
 * the savedState will be non-NULL and point to the saved data.  You must make
 * any copy of this data you need -- it will be released after you return from
 * this function.
 */
typedef void ANativeActivity_createFunc(ANativeActivity* activity,
        void* savedState, size_t savedStateSize);

/**
 * The name of the function that NativeInstance looks for when launching its
 * native code.  This is the default function that is used, you can specify
 * "android.app.func_name" string meta-data in your manifest to use a different
 * function.
 */
extern ANativeActivity_createFunc ANativeActivity_onCreate;

/**
 * Finish the given activity.  Its finish() method will be called, causing it
 * to be stopped and destroyed.  Note that this method can be called from
 * *any* thread; it will send a message to the main thread of the process
 * where the Java finish call will take place.
 */
void ANativeActivity_finish(ANativeActivity* activity);

/**
 * Change the window format of the given activity.  Calls getWindow().setFormat()
 * of the given activity.  Note that this method can be called from
 * *any* thread; it will send a message to the main thread of the process
 * where the Java finish call will take place.
 */
void ANativeActivity_setWindowFormat(ANativeActivity* activity, int32_t format);

/**
 * Change the window flags of the given activity.  Calls getWindow().setFlags()
 * of the given activity.  Note that this method can be called from
 * *any* thread; it will send a message to the main thread of the process
 * where the Java finish call will take place.  See window.h for flag constants.
 */
void ANativeActivity_setWindowFlags(ANativeActivity* activity,
        uint32_t addFlags, uint32_t removeFlags);

/**
 * Flags for ANativeActivity_showSoftInput; see the Java InputMethodManager
 * API for documentation.
 */
enum {
    /**
     * Implicit request to show the input window, not as the result
     * of a direct request by the user.
     */
    ANATIVEACTIVITY_SHOW_SOFT_INPUT_IMPLICIT = 0x0001,

    /**
     * The user has forced the input method open (such as by
     * long-pressing menu) so it should not be closed until they
     * explicitly do so.
     */
    ANATIVEACTIVITY_SHOW_SOFT_INPUT_FORCED = 0x0002,
};

/**
 * Show the IME while in the given activity.  Calls InputMethodManager.showSoftInput()
 * for the given activity.  Note that this method can be called from
 * *any* thread; it will send a message to the main thread of the process
 * where the Java finish call will take place.
 */
void ANativeActivity_showSoftInput(ANativeActivity* activity, uint32_t flags);

/**
 * Flags for ANativeActivity_hideSoftInput; see the Java InputMethodManager
 * API for documentation.
 */
enum {
    /**
     * The soft input window should only be hidden if it was not
     * explicitly shown by the user.
     */
    ANATIVEACTIVITY_HIDE_SOFT_INPUT_IMPLICIT_ONLY = 0x0001,
    /**
     * The soft input window should normally be hidden, unless it was
     * originally shown with {@link ANATIVEACTIVITY_SHOW_SOFT_INPUT_FORCED}.
     */
    ANATIVEACTIVITY_HIDE_SOFT_INPUT_NOT_ALWAYS = 0x0002,
};

/**
 * Hide the IME while in the given activity.  Calls InputMethodManager.hideSoftInput()
 * for the given activity.  Note that this method can be called from
 * *any* thread; it will send a message to the main thread of the process
 * where the Java finish call will take place.
 */
void ANativeActivity_hideSoftInput(ANativeActivity* activity, uint32_t flags);

#ifdef __cplusplus
};
#endif

#endif // ANDROID_NATIVE_ACTIVITY_H

/** @} */
