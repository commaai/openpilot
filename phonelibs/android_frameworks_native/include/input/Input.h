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

#ifndef _LIBINPUT_INPUT_H
#define _LIBINPUT_INPUT_H

/**
 * Native input event structures.
 */

#include <android/input.h>
#include <utils/BitSet.h>
#include <utils/KeyedVector.h>
#include <utils/RefBase.h>
#include <utils/String8.h>
#include <utils/Timers.h>
#include <utils/Vector.h>
#include <stdint.h>

/*
 * Additional private constants not defined in ndk/ui/input.h.
 */
enum {
    /* Signifies that the key is being predispatched */
    AKEY_EVENT_FLAG_PREDISPATCH = 0x20000000,

    /* Private control to determine when an app is tracking a key sequence. */
    AKEY_EVENT_FLAG_START_TRACKING = 0x40000000,

    /* Key event is inconsistent with previously sent key events. */
    AKEY_EVENT_FLAG_TAINTED = 0x80000000,
};

enum {

    /**
     * This flag indicates that the window that received this motion event is partly
     * or wholly obscured by another visible window above it.  This flag is set to true
     * even if the event did not directly pass through the obscured area.
     * A security sensitive application can check this flag to identify situations in which
     * a malicious application may have covered up part of its content for the purpose
     * of misleading the user or hijacking touches.  An appropriate response might be
     * to drop the suspect touches or to take additional precautions to confirm the user's
     * actual intent.
     */
    AMOTION_EVENT_FLAG_WINDOW_IS_PARTIALLY_OBSCURED = 0x2,

    /* Motion event is inconsistent with previously sent motion events. */
    AMOTION_EVENT_FLAG_TAINTED = 0x80000000,
};

enum {
    /* Used when a motion event is not associated with any display.
     * Typically used for non-pointer events. */
    ADISPLAY_ID_NONE = -1,

    /* The default display id. */
    ADISPLAY_ID_DEFAULT = 0,
};

enum {
    /*
     * Indicates that an input device has switches.
     * This input source flag is hidden from the API because switches are only used by the system
     * and applications have no way to interact with them.
     */
    AINPUT_SOURCE_SWITCH = 0x80000000,
};

enum {
    /**
     * Constants for LEDs. Hidden from the API since we don't actually expose a way to interact
     * with LEDs to developers
     *
     * NOTE: If you add LEDs here, you must also add them to InputEventLabels.h
     */

    ALED_NUM_LOCK = 0x00,
    ALED_CAPS_LOCK = 0x01,
    ALED_SCROLL_LOCK = 0x02,
    ALED_COMPOSE = 0x03,
    ALED_KANA = 0x04,
    ALED_SLEEP = 0x05,
    ALED_SUSPEND = 0x06,
    ALED_MUTE = 0x07,
    ALED_MISC = 0x08,
    ALED_MAIL = 0x09,
    ALED_CHARGING = 0x0a,
    ALED_CONTROLLER_1 = 0x10,
    ALED_CONTROLLER_2 = 0x11,
    ALED_CONTROLLER_3 = 0x12,
    ALED_CONTROLLER_4 = 0x13,
};

/* Maximum number of controller LEDs we support */
#define MAX_CONTROLLER_LEDS 4

/*
 * SystemUiVisibility constants from View.
 */
enum {
    ASYSTEM_UI_VISIBILITY_STATUS_BAR_VISIBLE = 0,
    ASYSTEM_UI_VISIBILITY_STATUS_BAR_HIDDEN = 0x00000001,
};

/*
 * Maximum number of pointers supported per motion event.
 * Smallest number of pointers is 1.
 * (We want at least 10 but some touch controllers obstensibly configured for 10 pointers
 * will occasionally emit 11.  There is not much harm making this constant bigger.)
 */
#define MAX_POINTERS 16

/*
 * Maximum number of samples supported per motion event.
 */
#define MAX_SAMPLES UINT16_MAX

/*
 * Maximum pointer id value supported in a motion event.
 * Smallest pointer id is 0.
 * (This is limited by our use of BitSet32 to track pointer assignments.)
 */
#define MAX_POINTER_ID 31

/*
 * Declare a concrete type for the NDK's input event forward declaration.
 */
struct AInputEvent {
    virtual ~AInputEvent() { }
};

/*
 * Declare a concrete type for the NDK's input device forward declaration.
 */
struct AInputDevice {
    virtual ~AInputDevice() { }
};


namespace android {

#ifdef HAVE_ANDROID_OS
class Parcel;
#endif

/*
 * Flags that flow alongside events in the input dispatch system to help with certain
 * policy decisions such as waking from device sleep.
 *
 * These flags are also defined in frameworks/base/core/java/android/view/WindowManagerPolicy.java.
 */
enum {
    /* These flags originate in RawEvents and are generally set in the key map.
     * NOTE: If you want a flag to be able to set in a keylayout file, then you must add it to
     * InputEventLabels.h as well. */

    // Indicates that the event should wake the device.
    POLICY_FLAG_WAKE = 0x00000001,

    // Indicates that the key is virtual, such as a capacitive button, and should
    // generate haptic feedback.  Virtual keys may be suppressed for some time
    // after a recent touch to prevent accidental activation of virtual keys adjacent
    // to the touch screen during an edge swipe.
    POLICY_FLAG_VIRTUAL = 0x00000002,

    // Indicates that the key is the special function modifier.
    POLICY_FLAG_FUNCTION = 0x00000004,

    // Indicates that the key represents a special gesture that has been detected by
    // the touch firmware or driver.  Causes touch events from the same device to be canceled.
    POLICY_FLAG_GESTURE = 0x00000008,

    POLICY_FLAG_RAW_MASK = 0x0000ffff,

    /* These flags are set by the input dispatcher. */

    // Indicates that the input event was injected.
    POLICY_FLAG_INJECTED = 0x01000000,

    // Indicates that the input event is from a trusted source such as a directly attached
    // input device or an application with system-wide event injection permission.
    POLICY_FLAG_TRUSTED = 0x02000000,

    // Indicates that the input event has passed through an input filter.
    POLICY_FLAG_FILTERED = 0x04000000,

    // Disables automatic key repeating behavior.
    POLICY_FLAG_DISABLE_KEY_REPEAT = 0x08000000,

    /* These flags are set by the input reader policy as it intercepts each event. */

    // Indicates that the device was in an interactive state when the
    // event was intercepted.
    POLICY_FLAG_INTERACTIVE = 0x20000000,

    // Indicates that the event should be dispatched to applications.
    // The input event should still be sent to the InputDispatcher so that it can see all
    // input events received include those that it will not deliver.
    POLICY_FLAG_PASS_TO_USER = 0x40000000,
};

/*
 * Pointer coordinate data.
 */
struct PointerCoords {
    enum { MAX_AXES = 30 }; // 30 so that sizeof(PointerCoords) == 128

    // Bitfield of axes that are present in this structure.
    uint64_t bits __attribute__((aligned(8)));

    // Values of axes that are stored in this structure packed in order by axis id
    // for each axis that is present in the structure according to 'bits'.
    float values[MAX_AXES];

    inline void clear() {
        BitSet64::clear(bits);
    }

    bool isEmpty() const {
        return BitSet64::isEmpty(bits);
    }

    float getAxisValue(int32_t axis) const;
    status_t setAxisValue(int32_t axis, float value);

    void scale(float scale);
    void applyOffset(float xOffset, float yOffset);

    inline float getX() const {
        return getAxisValue(AMOTION_EVENT_AXIS_X);
    }

    inline float getY() const {
        return getAxisValue(AMOTION_EVENT_AXIS_Y);
    }

#ifdef HAVE_ANDROID_OS
    status_t readFromParcel(Parcel* parcel);
    status_t writeToParcel(Parcel* parcel) const;
#endif

    bool operator==(const PointerCoords& other) const;
    inline bool operator!=(const PointerCoords& other) const {
        return !(*this == other);
    }

    void copyFrom(const PointerCoords& other);

private:
    void tooManyAxes(int axis);
};

/*
 * Pointer property data.
 */
struct PointerProperties {
    // The id of the pointer.
    int32_t id;

    // The pointer tool type.
    int32_t toolType;

    inline void clear() {
        id = -1;
        toolType = 0;
    }

    bool operator==(const PointerProperties& other) const;
    inline bool operator!=(const PointerProperties& other) const {
        return !(*this == other);
    }

    void copyFrom(const PointerProperties& other);
};

/*
 * Input events.
 */
class InputEvent : public AInputEvent {
public:
    virtual ~InputEvent() { }

    virtual int32_t getType() const = 0;

    inline int32_t getDeviceId() const { return mDeviceId; }

    inline int32_t getSource() const { return mSource; }

    inline void setSource(int32_t source) { mSource = source; }

protected:
    void initialize(int32_t deviceId, int32_t source);
    void initialize(const InputEvent& from);

    int32_t mDeviceId;
    int32_t mSource;
};

/*
 * Key events.
 */
class KeyEvent : public InputEvent {
public:
    virtual ~KeyEvent() { }

    virtual int32_t getType() const { return AINPUT_EVENT_TYPE_KEY; }

    inline int32_t getAction() const { return mAction; }

    inline int32_t getFlags() const { return mFlags; }

    inline void setFlags(int32_t flags) { mFlags = flags; }

    inline int32_t getKeyCode() const { return mKeyCode; }

    inline int32_t getScanCode() const { return mScanCode; }

    inline int32_t getMetaState() const { return mMetaState; }

    inline int32_t getRepeatCount() const { return mRepeatCount; }

    inline nsecs_t getDownTime() const { return mDownTime; }

    inline nsecs_t getEventTime() const { return mEventTime; }

    static const char* getLabel(int32_t keyCode);
    static int32_t getKeyCodeFromLabel(const char* label);
    
    void initialize(
            int32_t deviceId,
            int32_t source,
            int32_t action,
            int32_t flags,
            int32_t keyCode,
            int32_t scanCode,
            int32_t metaState,
            int32_t repeatCount,
            nsecs_t downTime,
            nsecs_t eventTime);
    void initialize(const KeyEvent& from);

protected:
    int32_t mAction;
    int32_t mFlags;
    int32_t mKeyCode;
    int32_t mScanCode;
    int32_t mMetaState;
    int32_t mRepeatCount;
    nsecs_t mDownTime;
    nsecs_t mEventTime;
};

/*
 * Motion events.
 */
class MotionEvent : public InputEvent {
public:
    virtual ~MotionEvent() { }

    virtual int32_t getType() const { return AINPUT_EVENT_TYPE_MOTION; }

    inline int32_t getAction() const { return mAction; }

    inline int32_t getActionMasked() const { return mAction & AMOTION_EVENT_ACTION_MASK; }

    inline int32_t getActionIndex() const {
        return (mAction & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK)
                >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
    }

    inline void setAction(int32_t action) { mAction = action; }

    inline int32_t getFlags() const { return mFlags; }

    inline void setFlags(int32_t flags) { mFlags = flags; }

    inline int32_t getEdgeFlags() const { return mEdgeFlags; }

    inline void setEdgeFlags(int32_t edgeFlags) { mEdgeFlags = edgeFlags; }

    inline int32_t getMetaState() const { return mMetaState; }

    inline void setMetaState(int32_t metaState) { mMetaState = metaState; }

    inline int32_t getButtonState() const { return mButtonState; }

    inline int32_t setButtonState(int32_t buttonState) { mButtonState = buttonState; }

    inline int32_t getActionButton() const { return mActionButton; }

    inline void setActionButton(int32_t button) { mActionButton = button; }

    inline float getXOffset() const { return mXOffset; }

    inline float getYOffset() const { return mYOffset; }

    inline float getXPrecision() const { return mXPrecision; }

    inline float getYPrecision() const { return mYPrecision; }

    inline nsecs_t getDownTime() const { return mDownTime; }

    inline void setDownTime(nsecs_t downTime) { mDownTime = downTime; }

    inline size_t getPointerCount() const { return mPointerProperties.size(); }

    inline const PointerProperties* getPointerProperties(size_t pointerIndex) const {
        return &mPointerProperties[pointerIndex];
    }

    inline int32_t getPointerId(size_t pointerIndex) const {
        return mPointerProperties[pointerIndex].id;
    }

    inline int32_t getToolType(size_t pointerIndex) const {
        return mPointerProperties[pointerIndex].toolType;
    }

    inline nsecs_t getEventTime() const { return mSampleEventTimes[getHistorySize()]; }

    const PointerCoords* getRawPointerCoords(size_t pointerIndex) const;

    float getRawAxisValue(int32_t axis, size_t pointerIndex) const;

    inline float getRawX(size_t pointerIndex) const {
        return getRawAxisValue(AMOTION_EVENT_AXIS_X, pointerIndex);
    }

    inline float getRawY(size_t pointerIndex) const {
        return getRawAxisValue(AMOTION_EVENT_AXIS_Y, pointerIndex);
    }

    float getAxisValue(int32_t axis, size_t pointerIndex) const;

    inline float getX(size_t pointerIndex) const {
        return getAxisValue(AMOTION_EVENT_AXIS_X, pointerIndex);
    }

    inline float getY(size_t pointerIndex) const {
        return getAxisValue(AMOTION_EVENT_AXIS_Y, pointerIndex);
    }

    inline float getPressure(size_t pointerIndex) const {
        return getAxisValue(AMOTION_EVENT_AXIS_PRESSURE, pointerIndex);
    }

    inline float getSize(size_t pointerIndex) const {
        return getAxisValue(AMOTION_EVENT_AXIS_SIZE, pointerIndex);
    }

    inline float getTouchMajor(size_t pointerIndex) const {
        return getAxisValue(AMOTION_EVENT_AXIS_TOUCH_MAJOR, pointerIndex);
    }

    inline float getTouchMinor(size_t pointerIndex) const {
        return getAxisValue(AMOTION_EVENT_AXIS_TOUCH_MINOR, pointerIndex);
    }

    inline float getToolMajor(size_t pointerIndex) const {
        return getAxisValue(AMOTION_EVENT_AXIS_TOOL_MAJOR, pointerIndex);
    }

    inline float getToolMinor(size_t pointerIndex) const {
        return getAxisValue(AMOTION_EVENT_AXIS_TOOL_MINOR, pointerIndex);
    }

    inline float getOrientation(size_t pointerIndex) const {
        return getAxisValue(AMOTION_EVENT_AXIS_ORIENTATION, pointerIndex);
    }

    inline size_t getHistorySize() const { return mSampleEventTimes.size() - 1; }

    inline nsecs_t getHistoricalEventTime(size_t historicalIndex) const {
        return mSampleEventTimes[historicalIndex];
    }

    const PointerCoords* getHistoricalRawPointerCoords(
            size_t pointerIndex, size_t historicalIndex) const;

    float getHistoricalRawAxisValue(int32_t axis, size_t pointerIndex,
            size_t historicalIndex) const;

    inline float getHistoricalRawX(size_t pointerIndex, size_t historicalIndex) const {
        return getHistoricalRawAxisValue(
                AMOTION_EVENT_AXIS_X, pointerIndex, historicalIndex);
    }

    inline float getHistoricalRawY(size_t pointerIndex, size_t historicalIndex) const {
        return getHistoricalRawAxisValue(
                AMOTION_EVENT_AXIS_Y, pointerIndex, historicalIndex);
    }

    float getHistoricalAxisValue(int32_t axis, size_t pointerIndex, size_t historicalIndex) const;

    inline float getHistoricalX(size_t pointerIndex, size_t historicalIndex) const {
        return getHistoricalAxisValue(
                AMOTION_EVENT_AXIS_X, pointerIndex, historicalIndex);
    }

    inline float getHistoricalY(size_t pointerIndex, size_t historicalIndex) const {
        return getHistoricalAxisValue(
                AMOTION_EVENT_AXIS_Y, pointerIndex, historicalIndex);
    }

    inline float getHistoricalPressure(size_t pointerIndex, size_t historicalIndex) const {
        return getHistoricalAxisValue(
                AMOTION_EVENT_AXIS_PRESSURE, pointerIndex, historicalIndex);
    }

    inline float getHistoricalSize(size_t pointerIndex, size_t historicalIndex) const {
        return getHistoricalAxisValue(
                AMOTION_EVENT_AXIS_SIZE, pointerIndex, historicalIndex);
    }

    inline float getHistoricalTouchMajor(size_t pointerIndex, size_t historicalIndex) const {
        return getHistoricalAxisValue(
                AMOTION_EVENT_AXIS_TOUCH_MAJOR, pointerIndex, historicalIndex);
    }

    inline float getHistoricalTouchMinor(size_t pointerIndex, size_t historicalIndex) const {
        return getHistoricalAxisValue(
                AMOTION_EVENT_AXIS_TOUCH_MINOR, pointerIndex, historicalIndex);
    }

    inline float getHistoricalToolMajor(size_t pointerIndex, size_t historicalIndex) const {
        return getHistoricalAxisValue(
                AMOTION_EVENT_AXIS_TOOL_MAJOR, pointerIndex, historicalIndex);
    }

    inline float getHistoricalToolMinor(size_t pointerIndex, size_t historicalIndex) const {
        return getHistoricalAxisValue(
                AMOTION_EVENT_AXIS_TOOL_MINOR, pointerIndex, historicalIndex);
    }

    inline float getHistoricalOrientation(size_t pointerIndex, size_t historicalIndex) const {
        return getHistoricalAxisValue(
                AMOTION_EVENT_AXIS_ORIENTATION, pointerIndex, historicalIndex);
    }

    ssize_t findPointerIndex(int32_t pointerId) const;

    void initialize(
            int32_t deviceId,
            int32_t source,
            int32_t action,
            int32_t actionButton,
            int32_t flags,
            int32_t edgeFlags,
            int32_t metaState,
            int32_t buttonState,
            float xOffset,
            float yOffset,
            float xPrecision,
            float yPrecision,
            nsecs_t downTime,
            nsecs_t eventTime,
            size_t pointerCount,
            const PointerProperties* pointerProperties,
            const PointerCoords* pointerCoords);

    void copyFrom(const MotionEvent* other, bool keepHistory);

    void addSample(
            nsecs_t eventTime,
            const PointerCoords* pointerCoords);

    void offsetLocation(float xOffset, float yOffset);

    void scale(float scaleFactor);

    // Apply 3x3 perspective matrix transformation.
    // Matrix is in row-major form and compatible with SkMatrix.
    void transform(const float matrix[9]);

#ifdef HAVE_ANDROID_OS
    status_t readFromParcel(Parcel* parcel);
    status_t writeToParcel(Parcel* parcel) const;
#endif

    static bool isTouchEvent(int32_t source, int32_t action);
    inline bool isTouchEvent() const {
        return isTouchEvent(mSource, mAction);
    }

    // Low-level accessors.
    inline const PointerProperties* getPointerProperties() const {
        return mPointerProperties.array();
    }
    inline const nsecs_t* getSampleEventTimes() const { return mSampleEventTimes.array(); }
    inline const PointerCoords* getSamplePointerCoords() const {
            return mSamplePointerCoords.array();
    }

    static const char* getLabel(int32_t axis);
    static int32_t getAxisFromLabel(const char* label);

protected:
    int32_t mAction;
    int32_t mActionButton;
    int32_t mFlags;
    int32_t mEdgeFlags;
    int32_t mMetaState;
    int32_t mButtonState;
    float mXOffset;
    float mYOffset;
    float mXPrecision;
    float mYPrecision;
    nsecs_t mDownTime;
    Vector<PointerProperties> mPointerProperties;
    Vector<nsecs_t> mSampleEventTimes;
    Vector<PointerCoords> mSamplePointerCoords;
};

/*
 * Input event factory.
 */
class InputEventFactoryInterface {
protected:
    virtual ~InputEventFactoryInterface() { }

public:
    InputEventFactoryInterface() { }

    virtual KeyEvent* createKeyEvent() = 0;
    virtual MotionEvent* createMotionEvent() = 0;
};

/*
 * A simple input event factory implementation that uses a single preallocated instance
 * of each type of input event that are reused for each request.
 */
class PreallocatedInputEventFactory : public InputEventFactoryInterface {
public:
    PreallocatedInputEventFactory() { }
    virtual ~PreallocatedInputEventFactory() { }

    virtual KeyEvent* createKeyEvent() { return & mKeyEvent; }
    virtual MotionEvent* createMotionEvent() { return & mMotionEvent; }

private:
    KeyEvent mKeyEvent;
    MotionEvent mMotionEvent;
};

/*
 * An input event factory implementation that maintains a pool of input events.
 */
class PooledInputEventFactory : public InputEventFactoryInterface {
public:
    PooledInputEventFactory(size_t maxPoolSize = 20);
    virtual ~PooledInputEventFactory();

    virtual KeyEvent* createKeyEvent();
    virtual MotionEvent* createMotionEvent();

    void recycle(InputEvent* event);

private:
    const size_t mMaxPoolSize;

    Vector<KeyEvent*> mKeyEventPool;
    Vector<MotionEvent*> mMotionEventPool;
};

} // namespace android

#endif // _LIBINPUT_INPUT_H
