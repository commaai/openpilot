/*
 * Copyright (C) 2007 The Android Open Source Project
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

#ifndef ANDROID_GUI_SURFACE_COMPOSER_CLIENT_H
#define ANDROID_GUI_SURFACE_COMPOSER_CLIENT_H

#include <stdint.h>
#include <sys/types.h>
#include <unordered_map>

#include <binder/IBinder.h>

#include <utils/RefBase.h>
#include <utils/Singleton.h>
#include <utils/SortedVector.h>
#include <utils/threads.h>

#include <ui/FrameStats.h>
#include <ui/GraphicTypes.h>
#include <ui/PixelFormat.h>

#include <gui/CpuConsumer.h>
#include <gui/SurfaceControl.h>
#include <math/vec3.h>
#include <gui/LayerState.h>

namespace android {

// ---------------------------------------------------------------------------

struct DisplayInfo;
class HdrCapabilities;
class ISurfaceComposerClient;
class IGraphicBufferProducer;
class Region;

// ---------------------------------------------------------------------------

class SurfaceComposerClient : public RefBase
{
    friend class Composer;
public:
                SurfaceComposerClient();
                SurfaceComposerClient(const sp<ISurfaceComposerClient>& client);
                SurfaceComposerClient(const sp<IGraphicBufferProducer>& parent);
    virtual     ~SurfaceComposerClient();

    // Always make sure we could initialize
    status_t    initCheck() const;

    // Return the connection of this client
    sp<IBinder> connection() const;

    // Forcibly remove connection before all references have gone away.
    void        dispose();

    // callback when the composer is dies
    status_t linkToComposerDeath(const sp<IBinder::DeathRecipient>& recipient,
            void* cookie = NULL, uint32_t flags = 0);

    // Get a list of supported configurations for a given display
    static status_t getDisplayConfigs(const sp<IBinder>& display,
            Vector<DisplayInfo>* configs);

    // Get the DisplayInfo for the currently-active configuration
    static status_t getDisplayInfo(const sp<IBinder>& display,
            DisplayInfo* info);

    // Get the display viewport for the given display
    static status_t getDisplayViewport(const sp<IBinder>& display, Rect* outViewport);

    // Get the index of the current active configuration (relative to the list
    // returned by getDisplayInfo)
    static int getActiveConfig(const sp<IBinder>& display);

    // Set a new active configuration using an index relative to the list
    // returned by getDisplayInfo
    static status_t setActiveConfig(const sp<IBinder>& display, int id);

    // Gets the list of supported color modes for the given display
    static status_t getDisplayColorModes(const sp<IBinder>& display,
            Vector<ui::ColorMode>* outColorModes);

    // Gets the active color mode for the given display
    static ui::ColorMode getActiveColorMode(const sp<IBinder>& display);

    // Sets the active color mode for the given display
    static status_t setActiveColorMode(const sp<IBinder>& display,
            ui::ColorMode colorMode);

    /* Triggers screen on/off or low power mode and waits for it to complete */
    static void setDisplayPowerMode(const sp<IBinder>& display, int mode);

    // ------------------------------------------------------------------------
    // surface creation / destruction

    //! Create a surface
    sp<SurfaceControl> createSurface(
            const String8& name,// name of the surface
            uint32_t w,         // width in pixel
            uint32_t h,         // height in pixel
            PixelFormat format, // pixel-format desired
            uint32_t flags = 0, // usage flags
            SurfaceControl* parent = nullptr, // parent
            int32_t windowType = -1, // from WindowManager.java (STATUS_BAR, INPUT_METHOD, etc.)
            int32_t ownerUid = -1 // UID of the task
    );

    status_t createSurfaceChecked(
            const String8& name,// name of the surface
            uint32_t w,         // width in pixel
            uint32_t h,         // height in pixel
            PixelFormat format, // pixel-format desired
            sp<SurfaceControl>* outSurface,
            uint32_t flags = 0, // usage flags
            SurfaceControl* parent = nullptr, // parent
            int32_t windowType = -1, // from WindowManager.java (STATUS_BAR, INPUT_METHOD, etc.)
            int32_t ownerUid = -1 // UID of the task
    );

    //! Create a virtual display
    static sp<IBinder> createDisplay(const String8& displayName, bool secure);

    //! Destroy a virtual display
    static void destroyDisplay(const sp<IBinder>& display);

    //! Get the token for the existing default displays.
    //! Possible values for id are eDisplayIdMain and eDisplayIdHdmi.
    static sp<IBinder> getBuiltInDisplay(int32_t id);

    static status_t enableVSyncInjections(bool enable);

    static status_t injectVSync(nsecs_t when);

    struct SCHash {
        std::size_t operator()(const sp<SurfaceControl>& sc) const {
            return std::hash<SurfaceControl *>{}(sc.get());
        }
    };

    class Transaction {
        std::unordered_map<sp<SurfaceControl>, ComposerState, SCHash> mComposerStates;
        SortedVector<DisplayState > mDisplayStates;
        uint32_t                    mForceSynchronous = 0;
        uint32_t                    mTransactionNestCount = 0;
        bool                        mAnimation = false;
        bool                        mEarlyWakeup = false;

        int mStatus = NO_ERROR;

        layer_state_t* getLayerState(const sp<SurfaceControl>& sc);
        DisplayState& getDisplayState(const sp<IBinder>& token);

    public:
        Transaction() = default;
        virtual ~Transaction() = default;
        Transaction(Transaction const& other);

        status_t apply(bool synchronous = false);
        // Merge another transaction in to this one, clearing other
        // as if it had been applied.
        Transaction& merge(Transaction&& other);
        Transaction& show(const sp<SurfaceControl>& sc);
        Transaction& hide(const sp<SurfaceControl>& sc);
        Transaction& setPosition(const sp<SurfaceControl>& sc,
                float x, float y);
        Transaction& setSize(const sp<SurfaceControl>& sc,
                uint32_t w, uint32_t h);
        Transaction& setLayer(const sp<SurfaceControl>& sc,
                int32_t z);

        // Sets a Z order relative to the Surface specified by "relativeTo" but
        // without becoming a full child of the relative. Z-ordering works exactly
        // as if it were a child however.
        //
        // As a nod to sanity, only non-child surfaces may have a relative Z-order.
        //
        // This overrides any previous call and is overriden by any future calls
        // to setLayer.
        //
        // If the relative is removed, the Surface will have no layer and be
        // invisible, until the next time set(Relative)Layer is called.
        Transaction& setRelativeLayer(const sp<SurfaceControl>& sc,
                const sp<IBinder>& relativeTo, int32_t z);
        Transaction& setFlags(const sp<SurfaceControl>& sc,
                uint32_t flags, uint32_t mask);
        Transaction& setTransparentRegionHint(const sp<SurfaceControl>& sc,
                const Region& transparentRegion);
        Transaction& setAlpha(const sp<SurfaceControl>& sc,
                float alpha);
        Transaction& setMatrix(const sp<SurfaceControl>& sc,
                float dsdx, float dtdx, float dtdy, float dsdy);
        Transaction& setCrop(const sp<SurfaceControl>& sc, const Rect& crop);
        Transaction& setFinalCrop(const sp<SurfaceControl>& sc, const Rect& crop);
        Transaction& setLayerStack(const sp<SurfaceControl>& sc, uint32_t layerStack);
        // Defers applying any changes made in this transaction until the Layer
        // identified by handle reaches the given frameNumber. If the Layer identified
        // by handle is removed, then we will apply this transaction regardless of
        // what frame number has been reached.
        Transaction& deferTransactionUntil(const sp<SurfaceControl>& sc,
                const sp<IBinder>& handle,
                uint64_t frameNumber);
        // A variant of deferTransactionUntil which identifies the Layer we wait for by
        // Surface instead of Handle. Useful for clients which may not have the
        // SurfaceControl for some of their Surfaces. Otherwise behaves identically.
        Transaction& deferTransactionUntil(const sp<SurfaceControl>& sc,
                const sp<Surface>& barrierSurface,
                uint64_t frameNumber);
        // Reparents all children of this layer to the new parent handle.
        Transaction& reparentChildren(const sp<SurfaceControl>& sc,
                const sp<IBinder>& newParentHandle);

        /// Reparents the current layer to the new parent handle. The new parent must not be null.
        // This can be used instead of reparentChildren if the caller wants to
        // only re-parent a specific child.
        Transaction& reparent(const sp<SurfaceControl>& sc,
                const sp<IBinder>& newParentHandle);

        Transaction& setColor(const sp<SurfaceControl>& sc, const half3& color);

        // Detaches all child surfaces (and their children recursively)
        // from their SurfaceControl.
        // The child SurfaceControls will not throw exceptions or return errors,
        // but transactions will have no effect.
        // The child surfaces will continue to follow their parent surfaces,
        // and remain eligible for rendering, but their relative state will be
        // frozen. We use this in the WindowManager, in app shutdown/relaunch
        // scenarios, where the app would otherwise clean up its child Surfaces.
        // Sometimes the WindowManager needs to extend their lifetime slightly
        // in order to perform an exit animation or prevent flicker.
        Transaction& detachChildren(const sp<SurfaceControl>& sc);
        // Set an override scaling mode as documented in <system/window.h>
        // the override scaling mode will take precedence over any client
        // specified scaling mode. -1 will clear the override scaling mode.
        Transaction& setOverrideScalingMode(const sp<SurfaceControl>& sc,
                int32_t overrideScalingMode);

        // If the size changes in this transaction, all geometry updates specified
        // in this transaction will not complete until a buffer of the new size
        // arrives. As some elements normally apply immediately, this enables
        // freezing the total geometry of a surface until a resize is completed.
        Transaction& setGeometryAppliesWithResize(const sp<SurfaceControl>& sc);

        Transaction& destroySurface(const sp<SurfaceControl>& sc);

        status_t setDisplaySurface(const sp<IBinder>& token,
                const sp<IGraphicBufferProducer>& bufferProducer);

        void setDisplayLayerStack(const sp<IBinder>& token, uint32_t layerStack);

        /* setDisplayProjection() defines the projection of layer stacks
         * to a given display.
         *
         * - orientation defines the display's orientation.
         * - layerStackRect defines which area of the window manager coordinate
         * space will be used.
         * - displayRect defines where on the display will layerStackRect be
         * mapped to. displayRect is specified post-orientation, that is
         * it uses the orientation seen by the end-user.
         */
        void setDisplayProjection(const sp<IBinder>& token,
                uint32_t orientation,
                const Rect& layerStackRect,
                const Rect& displayRect);
        void setDisplaySize(const sp<IBinder>& token, uint32_t width, uint32_t height);
        void setAnimationTransaction();
        void setEarlyWakeup();
    };

    status_t    destroySurface(const sp<IBinder>& id);

    status_t clearLayerFrameStats(const sp<IBinder>& token) const;
    status_t getLayerFrameStats(const sp<IBinder>& token, FrameStats* outStats) const;
    static status_t clearAnimationFrameStats();
    static status_t getAnimationFrameStats(FrameStats* outStats);

    static status_t getHdrCapabilities(const sp<IBinder>& display,
            HdrCapabilities* outCapabilities);

    static void setDisplayProjection(const sp<IBinder>& token,
            uint32_t orientation,
            const Rect& layerStackRect,
            const Rect& displayRect);

    inline sp<ISurfaceComposerClient> getClient() { return mClient; }

private:
    virtual void onFirstRef();

    mutable     Mutex                       mLock;
                status_t                    mStatus;
                sp<ISurfaceComposerClient>  mClient;
                wp<IGraphicBufferProducer>  mParent;
};

// ---------------------------------------------------------------------------

class ScreenshotClient {
public:
    // if cropping isn't required, callers may pass in a default Rect, e.g.:
    //   capture(display, producer, Rect(), reqWidth, ...);
    static status_t capture(const sp<IBinder>& display, Rect sourceCrop, uint32_t reqWidth,
                            uint32_t reqHeight, int32_t minLayerZ, int32_t maxLayerZ,
                            bool useIdentityTransform, uint32_t rotation,
                            bool captureSecureLayers, sp<GraphicBuffer>* outBuffer,
                            bool& outCapturedSecureLayers);
    static status_t capture(const sp<IBinder>& display, Rect sourceCrop, uint32_t reqWidth,
                            uint32_t reqHeight, int32_t minLayerZ, int32_t maxLayerZ,
                            bool useIdentityTransform, uint32_t rotation,
                            sp<GraphicBuffer>* outBuffer);
    static status_t captureLayers(const sp<IBinder>& layerHandle, Rect sourceCrop, float frameScale,
                                  sp<GraphicBuffer>* outBuffer);
    static status_t captureChildLayers(const sp<IBinder>& layerHandle, Rect sourceCrop,
                                       float frameScale, sp<GraphicBuffer>* outBuffer);
};

// ---------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_GUI_SURFACE_COMPOSER_CLIENT_H
