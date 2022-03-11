/*
 * Copyright (C) 2006 The Android Open Source Project
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

#ifndef ANDROID_GUI_ISURFACE_COMPOSER_H
#define ANDROID_GUI_ISURFACE_COMPOSER_H

#include <stdint.h>
#include <sys/types.h>

#include <utils/RefBase.h>
#include <utils/Errors.h>
#include <utils/Timers.h>
#include <utils/Vector.h>

#include <binder/IInterface.h>

#include <ui/FrameStats.h>
#include <ui/PixelFormat.h>
#include <ui/GraphicBuffer.h>
#include <ui/GraphicTypes.h>

#include <vector>

namespace android {
// ----------------------------------------------------------------------------

struct ComposerState;
struct DisplayState;
struct DisplayInfo;
struct DisplayStatInfo;
class LayerDebugInfo;
class HdrCapabilities;
class IDisplayEventConnection;
class IGraphicBufferProducer;
class ISurfaceComposerClient;
class Rect;
enum class FrameEvent;

/*
 * This class defines the Binder IPC interface for accessing various
 * SurfaceFlinger features.
 */
class ISurfaceComposer: public IInterface {
public:
    DECLARE_META_INTERFACE(SurfaceComposer)

    // flags for setTransactionState()
    enum {
        eSynchronous = 0x01,
        eAnimation   = 0x02,

        // Indicates that this transaction will likely result in a lot of layers being composed, and
        // thus, SurfaceFlinger should wake-up earlier to avoid missing frame deadlines. In this
        // case SurfaceFlinger will wake up at (sf vsync offset - debug.sf.early_phase_offset_ns)
        eEarlyWakeup = 0x04
    };

    enum {
        eDisplayIdMain = 0,
        eDisplayIdHdmi = 1
    };

    enum Rotation {
        eRotateNone = 0,
        eRotate90   = 1,
        eRotate180  = 2,
        eRotate270  = 3
    };

    enum VsyncSource {
        eVsyncSourceApp = 0,
        eVsyncSourceSurfaceFlinger = 1
    };

    /* create connection with surface flinger, requires
     * ACCESS_SURFACE_FLINGER permission
     */
    virtual sp<ISurfaceComposerClient> createConnection() = 0;

    /** create a scoped connection with surface flinger.
     * Surfaces produced with this connection will act
     * as children of the passed in GBP. That is to say
     * SurfaceFlinger will draw them relative and confined to
     * drawing of buffers from the layer associated with parent.
     * As this is graphically equivalent in reach to just drawing
     * pixels into the parent buffers, it requires no special permission.
     */
    virtual sp<ISurfaceComposerClient> createScopedConnection(
            const sp<IGraphicBufferProducer>& parent) = 0;

    /* return an IDisplayEventConnection */
    virtual sp<IDisplayEventConnection> createDisplayEventConnection(
            VsyncSource vsyncSource = eVsyncSourceApp) = 0;

    /* create a virtual display
     * requires ACCESS_SURFACE_FLINGER permission.
     */
    virtual sp<IBinder> createDisplay(const String8& displayName,
            bool secure) = 0;

    /* destroy a virtual display
     * requires ACCESS_SURFACE_FLINGER permission.
     */
    virtual void destroyDisplay(const sp<IBinder>& display) = 0;

    /* get the token for the existing default displays. possible values
     * for id are eDisplayIdMain and eDisplayIdHdmi.
     */
    virtual sp<IBinder> getBuiltInDisplay(int32_t id) = 0;

    /* open/close transactions. requires ACCESS_SURFACE_FLINGER permission */
    virtual void setTransactionState(const Vector<ComposerState>& state,
            const Vector<DisplayState>& displays, uint32_t flags) = 0;

    /* signal that we're done booting.
     * Requires ACCESS_SURFACE_FLINGER permission
     */
    virtual void bootFinished() = 0;

    /* verify that an IGraphicBufferProducer was created by SurfaceFlinger.
     */
    virtual bool authenticateSurfaceTexture(
            const sp<IGraphicBufferProducer>& surface) const = 0;

    /* Returns the frame timestamps supported by SurfaceFlinger.
     */
    virtual status_t getSupportedFrameTimestamps(
            std::vector<FrameEvent>* outSupported) const = 0;

    /* set display power mode. depending on the mode, it can either trigger
     * screen on, off or low power mode and wait for it to complete.
     * requires ACCESS_SURFACE_FLINGER permission.
     */
    virtual void setPowerMode(const sp<IBinder>& display, int mode) = 0;

    /* returns information for each configuration of the given display
     * intended to be used to get information about built-in displays */
    virtual status_t getDisplayConfigs(const sp<IBinder>& display,
            Vector<DisplayInfo>* configs) = 0;

    /* returns display statistics for a given display
     * intended to be used by the media framework to properly schedule
     * video frames */
    virtual status_t getDisplayStats(const sp<IBinder>& display,
            DisplayStatInfo* stats) = 0;

    /* returns display viewport information of the given display */
    virtual status_t getDisplayViewport(const sp<IBinder>& display, Rect* outViewport) = 0;

    /* indicates which of the configurations returned by getDisplayInfo is
     * currently active */
    virtual int getActiveConfig(const sp<IBinder>& display) = 0;

    /* specifies which configuration (of those returned by getDisplayInfo)
     * should be used */
    virtual status_t setActiveConfig(const sp<IBinder>& display, int id) = 0;

    virtual status_t getDisplayColorModes(const sp<IBinder>& display,
            Vector<ui::ColorMode>* outColorModes) = 0;
    virtual ui::ColorMode getActiveColorMode(const sp<IBinder>& display) = 0;
    virtual status_t setActiveColorMode(const sp<IBinder>& display,
            ui::ColorMode colorMode) = 0;

    /* Capture the specified screen. requires READ_FRAME_BUFFER permission
     * This function will fail if there is a secure window on screen.
     */
    virtual status_t captureScreen(const sp<IBinder>& display, sp<GraphicBuffer>* outBuffer,
                                   bool& outCapturedSecureLayers, Rect sourceCrop,
                                   uint32_t reqWidth, uint32_t reqHeight, int32_t minLayerZ,
                                   int32_t maxLayerZ, bool useIdentityTransform,
                                   Rotation rotation = eRotateNone,
                                   bool captureSecureLayers = false) = 0;

    virtual status_t captureScreen(const sp<IBinder>& display, sp<GraphicBuffer>* outBuffer,
                                   Rect sourceCrop,
                                   uint32_t reqWidth, uint32_t reqHeight, int32_t minLayerZ,
                                   int32_t maxLayerZ, bool useIdentityTransform,
                                   Rotation rotation = eRotateNone,
                                   bool captureSecureLayers = false) {
      bool ignored;
      return captureScreen(display, outBuffer, ignored, sourceCrop, reqWidth, reqHeight, minLayerZ,
                           maxLayerZ, useIdentityTransform, rotation, captureSecureLayers);
    }
    /**
     * Capture a subtree of the layer hierarchy, potentially ignoring the root node.
     */
    virtual status_t captureLayers(const sp<IBinder>& layerHandleBinder,
                                   sp<GraphicBuffer>* outBuffer, const Rect& sourceCrop,
                                   float frameScale = 1.0, bool childrenOnly = false) = 0;

    /* Clears the frame statistics for animations.
     *
     * Requires the ACCESS_SURFACE_FLINGER permission.
     */
    virtual status_t clearAnimationFrameStats() = 0;

    /* Gets the frame statistics for animations.
     *
     * Requires the ACCESS_SURFACE_FLINGER permission.
     */
    virtual status_t getAnimationFrameStats(FrameStats* outStats) const = 0;

    /* Gets the supported HDR capabilities of the given display.
     *
     * Requires the ACCESS_SURFACE_FLINGER permission.
     */
    virtual status_t getHdrCapabilities(const sp<IBinder>& display,
            HdrCapabilities* outCapabilities) const = 0;

    virtual status_t enableVSyncInjections(bool enable) = 0;

    virtual status_t injectVSync(nsecs_t when) = 0;

    /* Gets the list of active layers in Z order for debugging purposes
     *
     * Requires the ACCESS_SURFACE_FLINGER permission.
     */
    virtual status_t getLayerDebugInfo(std::vector<LayerDebugInfo>* outLayers) const = 0;
};

// ----------------------------------------------------------------------------

class BnSurfaceComposer: public BnInterface<ISurfaceComposer> {
public:
    enum {
        // Note: BOOT_FINISHED must remain this value, it is called from
        // Java by ActivityManagerService.
        BOOT_FINISHED = IBinder::FIRST_CALL_TRANSACTION,
        CREATE_CONNECTION,
        UNUSED, // formerly CREATE_GRAPHIC_BUFFER_ALLOC
        CREATE_DISPLAY_EVENT_CONNECTION,
        CREATE_DISPLAY,
        DESTROY_DISPLAY,
        GET_BUILT_IN_DISPLAY,
        SET_TRANSACTION_STATE,
        AUTHENTICATE_SURFACE,
        GET_SUPPORTED_FRAME_TIMESTAMPS,
        GET_DISPLAY_CONFIGS,
        GET_ACTIVE_CONFIG,
        SET_ACTIVE_CONFIG,
        CONNECT_DISPLAY,
        CAPTURE_SCREEN,
        CAPTURE_LAYERS,
        CLEAR_ANIMATION_FRAME_STATS,
        GET_ANIMATION_FRAME_STATS,
        SET_POWER_MODE,
        GET_DISPLAY_STATS,
        GET_HDR_CAPABILITIES,
        GET_DISPLAY_COLOR_MODES,
        GET_ACTIVE_COLOR_MODE,
        SET_ACTIVE_COLOR_MODE,
        ENABLE_VSYNC_INJECTIONS,
        INJECT_VSYNC,
        GET_LAYER_DEBUG_INFO,
        CREATE_SCOPED_CONNECTION,
        GET_DISPLAY_VIEWPORT
    };

    virtual status_t onTransact(uint32_t code, const Parcel& data,
            Parcel* reply, uint32_t flags = 0);
};

// ----------------------------------------------------------------------------

}; // namespace android

#endif // ANDROID_GUI_ISURFACE_COMPOSER_H
