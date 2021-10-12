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
 * @file window.h
 */

#ifndef ANDROID_WINDOW_H
#define ANDROID_WINDOW_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Window flags, as per the Java API at android.view.WindowManager.LayoutParams.
 */
enum {
    /**
     * As long as this window is visible to the user, allow the lock
     * screen to activate while the screen is on.  This can be used
     * independently, or in combination with {@link
     * AWINDOW_FLAG_KEEP_SCREEN_ON} and/or {@link
     * AWINDOW_FLAG_SHOW_WHEN_LOCKED}
     */
    AWINDOW_FLAG_ALLOW_LOCK_WHILE_SCREEN_ON = 0x00000001,
    /** Everything behind this window will be dimmed. */
    AWINDOW_FLAG_DIM_BEHIND                 = 0x00000002,
    /**
     * Blur everything behind this window.
     * @deprecated Blurring is no longer supported.
     */
    AWINDOW_FLAG_BLUR_BEHIND                = 0x00000004,
    /**
     * This window won't ever get key input focus, so the
     * user can not send key or other button events to it.  Those will
     * instead go to whatever focusable window is behind it.  This flag
     * will also enable {@link AWINDOW_FLAG_NOT_TOUCH_MODAL} whether or not that
     * is explicitly set.
     *
     * Setting this flag also implies that the window will not need to
     * interact with
     * a soft input method, so it will be Z-ordered and positioned
     * independently of any active input method (typically this means it
     * gets Z-ordered on top of the input method, so it can use the full
     * screen for its content and cover the input method if needed.  You
     * can use {@link AWINDOW_FLAG_ALT_FOCUSABLE_IM} to modify this behavior.
     */
    AWINDOW_FLAG_NOT_FOCUSABLE              = 0x00000008,
    /** this window can never receive touch events. */
    AWINDOW_FLAG_NOT_TOUCHABLE              = 0x00000010,
    /**
     * Even when this window is focusable (its
     * {@link AWINDOW_FLAG_NOT_FOCUSABLE} is not set), allow any pointer events
     * outside of the window to be sent to the windows behind it.  Otherwise
     * it will consume all pointer events itself, regardless of whether they
     * are inside of the window.
     */
    AWINDOW_FLAG_NOT_TOUCH_MODAL            = 0x00000020,
    /**
     * When set, if the device is asleep when the touch
     * screen is pressed, you will receive this first touch event.  Usually
     * the first touch event is consumed by the system since the user can
     * not see what they are pressing on.
     *
     * @deprecated This flag has no effect.
     */
    AWINDOW_FLAG_TOUCHABLE_WHEN_WAKING      = 0x00000040,
    /**
     * As long as this window is visible to the user, keep
     * the device's screen turned on and bright.
     */
    AWINDOW_FLAG_KEEP_SCREEN_ON             = 0x00000080,
    /**
     * Place the window within the entire screen, ignoring
     * decorations around the border (such as the status bar).  The
     * window must correctly position its contents to take the screen
     * decoration into account.
     */
    AWINDOW_FLAG_LAYOUT_IN_SCREEN           = 0x00000100,
    /** allow window to extend outside of the screen. */
    AWINDOW_FLAG_LAYOUT_NO_LIMITS           = 0x00000200,
    /**
     * Hide all screen decorations (such as the status
     * bar) while this window is displayed.  This allows the window to
     * use the entire display space for itself -- the status bar will
     * be hidden when an app window with this flag set is on the top
     * layer. A fullscreen window will ignore a value of {@link
     * AWINDOW_SOFT_INPUT_ADJUST_RESIZE}; the window will stay
     * fullscreen and will not resize.
     */
    AWINDOW_FLAG_FULLSCREEN                 = 0x00000400,
    /**
     * Override {@link AWINDOW_FLAG_FULLSCREEN} and force the
     * screen decorations (such as the status bar) to be shown.
     */
    AWINDOW_FLAG_FORCE_NOT_FULLSCREEN       = 0x00000800,
    /**
     * Turn on dithering when compositing this window to
     * the screen.
     * @deprecated This flag is no longer used.
     */
    AWINDOW_FLAG_DITHER                     = 0x00001000,
    /**
     * Treat the content of the window as secure, preventing
     * it from appearing in screenshots or from being viewed on non-secure
     * displays.
     */
    AWINDOW_FLAG_SECURE                     = 0x00002000,
    /**
     * A special mode where the layout parameters are used
     * to perform scaling of the surface when it is composited to the
     * screen.
     */
    AWINDOW_FLAG_SCALED                     = 0x00004000,
    /**
     * Intended for windows that will often be used when the user is
     * holding the screen against their face, it will aggressively
     * filter the event stream to prevent unintended presses in this
     * situation that may not be desired for a particular window, when
     * such an event stream is detected, the application will receive
     * a {@link AMOTION_EVENT_ACTION_CANCEL} to indicate this so
     * applications can handle this accordingly by taking no action on
     * the event until the finger is released.
     */
    AWINDOW_FLAG_IGNORE_CHEEK_PRESSES       = 0x00008000,
    /**
     * A special option only for use in combination with
     * {@link AWINDOW_FLAG_LAYOUT_IN_SCREEN}.  When requesting layout in the
     * screen your window may appear on top of or behind screen decorations
     * such as the status bar.  By also including this flag, the window
     * manager will report the inset rectangle needed to ensure your
     * content is not covered by screen decorations.
     */
    AWINDOW_FLAG_LAYOUT_INSET_DECOR         = 0x00010000,
    /**
     * Invert the state of {@link AWINDOW_FLAG_NOT_FOCUSABLE} with
     * respect to how this window interacts with the current method.
     * That is, if FLAG_NOT_FOCUSABLE is set and this flag is set,
     * then the window will behave as if it needs to interact with the
     * input method and thus be placed behind/away from it; if {@link
     * AWINDOW_FLAG_NOT_FOCUSABLE} is not set and this flag is set,
     * then the window will behave as if it doesn't need to interact
     * with the input method and can be placed to use more space and
     * cover the input method.
     */
    AWINDOW_FLAG_ALT_FOCUSABLE_IM           = 0x00020000,
    /**
     * If you have set {@link AWINDOW_FLAG_NOT_TOUCH_MODAL}, you
     * can set this flag to receive a single special MotionEvent with
     * the action
     * {@link AMOTION_EVENT_ACTION_OUTSIDE} for
     * touches that occur outside of your window.  Note that you will not
     * receive the full down/move/up gesture, only the location of the
     * first down as an {@link AMOTION_EVENT_ACTION_OUTSIDE}.
     */
    AWINDOW_FLAG_WATCH_OUTSIDE_TOUCH        = 0x00040000,
    /**
     * Special flag to let windows be shown when the screen
     * is locked. This will let application windows take precedence over
     * key guard or any other lock screens. Can be used with
     * {@link AWINDOW_FLAG_KEEP_SCREEN_ON} to turn screen on and display windows
     * directly before showing the key guard window.  Can be used with
     * {@link AWINDOW_FLAG_DISMISS_KEYGUARD} to automatically fully dismisss
     * non-secure keyguards.  This flag only applies to the top-most
     * full-screen window.
     */
    AWINDOW_FLAG_SHOW_WHEN_LOCKED           = 0x00080000,
    /**
     * Ask that the system wallpaper be shown behind
     * your window.  The window surface must be translucent to be able
     * to actually see the wallpaper behind it; this flag just ensures
     * that the wallpaper surface will be there if this window actually
     * has translucent regions.
     */
    AWINDOW_FLAG_SHOW_WALLPAPER             = 0x00100000,
    /**
     * When set as a window is being added or made
     * visible, once the window has been shown then the system will
     * poke the power manager's user activity (as if the user had woken
     * up the device) to turn the screen on.
     */
    AWINDOW_FLAG_TURN_SCREEN_ON             = 0x00200000,
    /**
     * When set the window will cause the keyguard to
     * be dismissed, only if it is not a secure lock keyguard.  Because such
     * a keyguard is not needed for security, it will never re-appear if
     * the user navigates to another window (in contrast to
     * {@link AWINDOW_FLAG_SHOW_WHEN_LOCKED}, which will only temporarily
     * hide both secure and non-secure keyguards but ensure they reappear
     * when the user moves to another UI that doesn't hide them).
     * If the keyguard is currently active and is secure (requires an
     * unlock pattern) than the user will still need to confirm it before
     * seeing this window, unless {@link AWINDOW_FLAG_SHOW_WHEN_LOCKED} has
     * also been set.
     */
    AWINDOW_FLAG_DISMISS_KEYGUARD           = 0x00400000,
};

#ifdef __cplusplus
};
#endif

#endif // ANDROID_WINDOW_H

/** @} */
