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

#include <binder/IBinder.h>
#include <binder/IMemory.h>

#include <utils/RefBase.h>
#include <utils/Singleton.h>
#include <utils/SortedVector.h>
#include <utils/threads.h>

#include <ui/FrameStats.h>
#include <ui/PixelFormat.h>

#include <gui/CpuConsumer.h>
#include <gui/SurfaceControl.h>

namespace android {

// ---------------------------------------------------------------------------

class DisplayInfo;
class Composer;
class ISurfaceComposerClient;
class IGraphicBufferProducer;
class Region;

// ---------------------------------------------------------------------------

class SurfaceComposerClient : public RefBase
{
    friend class Composer;
public:
                SurfaceComposerClient();
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

    // Get the index of the current active configuration (relative to the list
    // returned by getDisplayInfo)
    static int getActiveConfig(const sp<IBinder>& display);

    // Set a new active configuration using an index relative to the list
    // returned by getDisplayInfo
    static status_t setActiveConfig(const sp<IBinder>& display, int id);

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
            uint32_t flags = 0  // usage flags
    );

    //! Create a virtual display
    static sp<IBinder> createDisplay(const String8& displayName, bool secure);

    //! Destroy a virtual display
    static void destroyDisplay(const sp<IBinder>& display);

    //! Get the token for the existing default displays.
    //! Possible values for id are eDisplayIdMain and eDisplayIdHdmi.
    static sp<IBinder> getBuiltInDisplay(int32_t id);

    // ------------------------------------------------------------------------
    // Composer parameters
    // All composer parameters must be changed within a transaction
    // several surfaces can be updated in one transaction, all changes are
    // committed at once when the transaction is closed.
    // closeGlobalTransaction() requires an IPC with the server.

    //! Open a composer transaction on all active SurfaceComposerClients.
    static void openGlobalTransaction();

    //! Close a composer transaction on all active SurfaceComposerClients.
    static void closeGlobalTransaction(bool synchronous = false);

    //! Flag the currently open transaction as an animation transaction.
    static void setAnimationTransaction();

    status_t    hide(const sp<IBinder>& id);
    status_t    show(const sp<IBinder>& id);
    status_t    setFlags(const sp<IBinder>& id, uint32_t flags, uint32_t mask);
    status_t    setTransparentRegionHint(const sp<IBinder>& id, const Region& transparent);
    status_t    setLayer(const sp<IBinder>& id, uint32_t layer);
    status_t    setAlpha(const sp<IBinder>& id, float alpha=1.0f);
    status_t    setMatrix(const sp<IBinder>& id, float dsdx, float dtdx, float dsdy, float dtdy);
    status_t    setPosition(const sp<IBinder>& id, float x, float y);
    status_t    setSize(const sp<IBinder>& id, uint32_t w, uint32_t h);
    status_t    setCrop(const sp<IBinder>& id, const Rect& crop);
    status_t    setLayerStack(const sp<IBinder>& id, uint32_t layerStack);
    status_t    destroySurface(const sp<IBinder>& id);

    status_t clearLayerFrameStats(const sp<IBinder>& token) const;
    status_t getLayerFrameStats(const sp<IBinder>& token, FrameStats* outStats) const;

    static status_t clearAnimationFrameStats();
    static status_t getAnimationFrameStats(FrameStats* outStats);

    static void setDisplaySurface(const sp<IBinder>& token,
            const sp<IGraphicBufferProducer>& bufferProducer);
    static void setDisplayLayerStack(const sp<IBinder>& token,
            uint32_t layerStack);
    static void setDisplaySize(const sp<IBinder>& token, uint32_t width, uint32_t height);

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
    static void setDisplayProjection(const sp<IBinder>& token,
            uint32_t orientation,
            const Rect& layerStackRect,
            const Rect& displayRect);

    status_t    setBlur(const sp<IBinder>& id, float blur);
    status_t    setBlurMaskSurface(const sp<IBinder>& id, const sp<IBinder>& maskSurfaceId);
    status_t    setBlurMaskSampling(const sp<IBinder>& id, uint32_t blurMaskSampling);
    status_t    setBlurMaskAlphaThreshold(const sp<IBinder>& id, float alpha);

private:
    virtual void onFirstRef();
    Composer& getComposer();

    mutable     Mutex                       mLock;
                status_t                    mStatus;
                sp<ISurfaceComposerClient>  mClient;
                Composer&                   mComposer;
};

// ---------------------------------------------------------------------------

class ScreenshotClient
{
public:
    // if cropping isn't required, callers may pass in a default Rect, e.g.:
    //   capture(display, producer, Rect(), reqWidth, ...);
    static status_t capture(
            const sp<IBinder>& display,
            const sp<IGraphicBufferProducer>& producer,
            Rect sourceCrop, uint32_t reqWidth, uint32_t reqHeight,
            uint32_t minLayerZ, uint32_t maxLayerZ,
            bool useIdentityTransform);

private:
    mutable sp<CpuConsumer> mCpuConsumer;
    mutable sp<IGraphicBufferProducer> mProducer;
    CpuConsumer::LockedBuffer mBuffer;
    bool mHaveBuffer;

public:
    ScreenshotClient();
    ~ScreenshotClient();

    // frees the previous screenshot and captures a new one
    // if cropping isn't required, callers may pass in a default Rect, e.g.:
    //   update(display, Rect(), useIdentityTransform);
    status_t update(const sp<IBinder>& display,
            Rect sourceCrop, bool useIdentityTransform);
    status_t update(const sp<IBinder>& display,
            Rect sourceCrop, uint32_t reqWidth, uint32_t reqHeight,
            bool useIdentityTransform);
    status_t update(const sp<IBinder>& display,
            Rect sourceCrop, uint32_t reqWidth, uint32_t reqHeight,
            uint32_t minLayerZ, uint32_t maxLayerZ,
            bool useIdentityTransform);
    status_t update(const sp<IBinder>& display,
            Rect sourceCrop, uint32_t reqWidth, uint32_t reqHeight,
            uint32_t minLayerZ, uint32_t maxLayerZ,
            bool useIdentityTransform, uint32_t rotation);

    sp<CpuConsumer> getCpuConsumer() const;

    // release memory occupied by the screenshot
    void release();

    // pixels are valid until this object is freed or
    // release() or update() is called
    void const* getPixels() const;

    uint32_t getWidth() const;
    uint32_t getHeight() const;
    PixelFormat getFormat() const;
    uint32_t getStride() const;
    // size of allocated memory in bytes
    size_t getSize() const;
};

// ---------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_GUI_SURFACE_COMPOSER_CLIENT_H
