/*
 * Copyright 2016 The Android Open Source Project
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

#ifndef ANDROID_GUI_FRAMETIMESTAMPS_H
#define ANDROID_GUI_FRAMETIMESTAMPS_H

#include <ui/FenceTime.h>
#include <utils/Flattenable.h>
#include <utils/StrongPointer.h>
#include <utils/Timers.h>

#include <array>
#include <bitset>
#include <vector>

namespace android {

struct FrameEvents;
class FrameEventHistoryDelta;
class String8;


// Identifiers for all the events that may be recorded or reported.
enum class FrameEvent {
    POSTED,
    REQUESTED_PRESENT,
    LATCH,
    ACQUIRE,
    FIRST_REFRESH_START,
    LAST_REFRESH_START,
    GPU_COMPOSITION_DONE,
    DISPLAY_PRESENT,
    DEQUEUE_READY,
    RELEASE,
    EVENT_COUNT, // Not an actual event.
};


// A collection of timestamps corresponding to a single frame.
struct FrameEvents {
    static constexpr auto EVENT_COUNT =
            static_cast<size_t>(FrameEvent::EVENT_COUNT);
    static_assert(EVENT_COUNT <= 32, "Event count sanity check failed.");
    static constexpr nsecs_t TIMESTAMP_PENDING = -2;

    static inline bool isValidTimestamp(nsecs_t time) {
        return time != TIMESTAMP_PENDING;
    }

    bool hasPostedInfo() const;
    bool hasRequestedPresentInfo() const;
    bool hasLatchInfo() const;
    bool hasFirstRefreshStartInfo() const;
    bool hasLastRefreshStartInfo() const;
    bool hasAcquireInfo() const;
    bool hasGpuCompositionDoneInfo() const;
    bool hasDisplayPresentInfo() const;
    bool hasReleaseInfo() const;
    bool hasDequeueReadyInfo() const;

    void checkFencesForCompletion();
    void dump(String8& outString) const;

    bool valid{false};
    int connectId{0};
    uint64_t frameNumber{0};

    // Whether or not certain points in the frame's life cycle have been
    // encountered help us determine if timestamps aren't available because
    // a) we'll just never get them or b) they're not ready yet.
    bool addPostCompositeCalled{false};
    bool addReleaseCalled{false};

    nsecs_t postedTime{TIMESTAMP_PENDING};
    nsecs_t requestedPresentTime{TIMESTAMP_PENDING};
    nsecs_t latchTime{TIMESTAMP_PENDING};
    nsecs_t firstRefreshStartTime{TIMESTAMP_PENDING};
    nsecs_t lastRefreshStartTime{TIMESTAMP_PENDING};
    nsecs_t dequeueReadyTime{TIMESTAMP_PENDING};

    std::shared_ptr<FenceTime> acquireFence{FenceTime::NO_FENCE};
    std::shared_ptr<FenceTime> gpuCompositionDoneFence{FenceTime::NO_FENCE};
    std::shared_ptr<FenceTime> displayPresentFence{FenceTime::NO_FENCE};
    std::shared_ptr<FenceTime> releaseFence{FenceTime::NO_FENCE};
};

struct CompositorTiming {
    nsecs_t deadline{0};
    nsecs_t interval{16666667};
    nsecs_t presentLatency{16666667};
};

// A short history of frames that are synchronized between the consumer and
// producer via deltas.
class FrameEventHistory {
public:
    virtual ~FrameEventHistory();

    FrameEvents* getFrame(uint64_t frameNumber);
    FrameEvents* getFrame(uint64_t frameNumber, size_t* iHint);
    void checkFencesForCompletion();
    void dump(String8& outString) const;

    static constexpr size_t MAX_FRAME_HISTORY = 8;

protected:
    std::array<FrameEvents, MAX_FRAME_HISTORY> mFrames;

    CompositorTiming mCompositorTiming;
};


// The producer's interface to FrameEventHistory
class ProducerFrameEventHistory : public FrameEventHistory {
public:
    ~ProducerFrameEventHistory() override;

    // Public for testing.
    static nsecs_t snapToNextTick(
            nsecs_t timestamp, nsecs_t tickPhase, nsecs_t tickInterval);

    nsecs_t getNextCompositeDeadline(const nsecs_t now) const;
    nsecs_t getCompositeInterval() const { return mCompositorTiming.interval; }
    nsecs_t getCompositeToPresentLatency() const {
        return mCompositorTiming.presentLatency;
    }

    // virtual for testing.
    virtual void updateAcquireFence(
            uint64_t frameNumber, std::shared_ptr<FenceTime>&& acquire);
    void applyDelta(const FrameEventHistoryDelta& delta);

    void updateSignalTimes();

protected:
    void applyFenceDelta(FenceTimeline* timeline,
            std::shared_ptr<FenceTime>* dst,
            const FenceTime::Snapshot& src) const;

    // virtual for testing.
    virtual std::shared_ptr<FenceTime> createFenceTime(
            const sp<Fence>& fence) const;

    size_t mAcquireOffset{0};

    // The consumer updates it's timelines in Layer and SurfaceFlinger since
    // they can coordinate shared timelines better. The producer doesn't have
    // shared timelines though, so just let it own and update all of them.
    FenceTimeline mAcquireTimeline;
    FenceTimeline mGpuCompositionDoneTimeline;
    FenceTimeline mPresentTimeline;
    FenceTimeline mReleaseTimeline;
};


// Used by the consumer to create a new frame event record that is
// partially complete.
struct NewFrameEventsEntry {
    uint64_t frameNumber{0};
    nsecs_t postedTime{0};
    nsecs_t requestedPresentTime{0};
    std::shared_ptr<FenceTime> acquireFence{FenceTime::NO_FENCE};
};


// Used by the consumer to keep track of which fields it already sent to
// the producer.
class FrameEventDirtyFields {
public:
    inline void reset() { mBitset.reset(); }
    inline bool anyDirty() const { return mBitset.any(); }

    template <FrameEvent event>
    inline void setDirty() {
        constexpr size_t eventIndex = static_cast<size_t>(event);
        static_assert(eventIndex < FrameEvents::EVENT_COUNT, "Bad index.");
        mBitset.set(eventIndex);
    }

    template <FrameEvent event>
    inline bool isDirty() const {
        constexpr size_t eventIndex = static_cast<size_t>(event);
        static_assert(eventIndex < FrameEvents::EVENT_COUNT, "Bad index.");
        return mBitset[eventIndex];
    }

private:
    std::bitset<FrameEvents::EVENT_COUNT> mBitset;
};


// The consumer's interface to FrameEventHistory
class ConsumerFrameEventHistory : public FrameEventHistory {
public:
    ~ConsumerFrameEventHistory() override;

    void onDisconnect();

    void initializeCompositorTiming(const CompositorTiming& compositorTiming);

    void addQueue(const NewFrameEventsEntry& newEntry);
    void addLatch(uint64_t frameNumber, nsecs_t latchTime);
    void addPreComposition(uint64_t frameNumber, nsecs_t refreshStartTime);
    void addPostComposition(uint64_t frameNumber,
            const std::shared_ptr<FenceTime>& gpuCompositionDone,
            const std::shared_ptr<FenceTime>& displayPresent,
            const CompositorTiming& compositorTiming);
    void addRelease(uint64_t frameNumber, nsecs_t dequeueReadyTime,
            std::shared_ptr<FenceTime>&& release);

    void getAndResetDelta(FrameEventHistoryDelta* delta);

private:
    void getFrameDelta(FrameEventHistoryDelta* delta,
            const std::array<FrameEvents, MAX_FRAME_HISTORY>::iterator& frame);

    std::array<FrameEventDirtyFields, MAX_FRAME_HISTORY> mFramesDirty;

    size_t mQueueOffset{0};
    size_t mCompositionOffset{0};
    size_t mReleaseOffset{0};

    int mCurrentConnectId{0};
    bool mProducerWantsEvents{false};
};


// A single frame update from the consumer to producer that can be sent
// through Binder.
// Although this may be sent multiple times for the same frame as new
// timestamps are set, Fences only need to be sent once.
class FrameEventsDelta : public Flattenable<FrameEventsDelta> {
friend class ProducerFrameEventHistory;
public:
    FrameEventsDelta() = default;
    FrameEventsDelta(size_t index,
            const FrameEvents& frameTimestamps,
            const FrameEventDirtyFields& dirtyFields);

    // Movable.
    FrameEventsDelta(FrameEventsDelta&& src) = default;
    FrameEventsDelta& operator=(FrameEventsDelta&& src) = default;
    // Not copyable.
    FrameEventsDelta(const FrameEventsDelta& src) = delete;
    FrameEventsDelta& operator=(const FrameEventsDelta& src) = delete;

    // Flattenable implementation
    size_t getFlattenedSize() const;
    size_t getFdCount() const;
    status_t flatten(void*& buffer, size_t& size, int*& fds,
            size_t& count) const;
    status_t unflatten(void const*& buffer, size_t& size, int const*& fds,
            size_t& count);

private:
    static constexpr size_t minFlattenedSize();

    size_t mIndex{0};
    uint64_t mFrameNumber{0};

    bool mAddPostCompositeCalled{0};
    bool mAddReleaseCalled{0};

    nsecs_t mPostedTime{FrameEvents::TIMESTAMP_PENDING};
    nsecs_t mRequestedPresentTime{FrameEvents::TIMESTAMP_PENDING};
    nsecs_t mLatchTime{FrameEvents::TIMESTAMP_PENDING};
    nsecs_t mFirstRefreshStartTime{FrameEvents::TIMESTAMP_PENDING};
    nsecs_t mLastRefreshStartTime{FrameEvents::TIMESTAMP_PENDING};
    nsecs_t mDequeueReadyTime{FrameEvents::TIMESTAMP_PENDING};

    FenceTime::Snapshot mGpuCompositionDoneFence;
    FenceTime::Snapshot mDisplayPresentFence;
    FenceTime::Snapshot mReleaseFence;

    // This is a static method with an auto return value so we can call
    // it without needing const and non-const versions.
    template <typename ThisT>
    static inline auto allFences(ThisT fed) ->
            std::array<decltype(&fed->mReleaseFence), 3> {
        return {{
            &fed->mGpuCompositionDoneFence, &fed->mDisplayPresentFence,
            &fed->mReleaseFence
        }};
    }
};


// A collection of updates from consumer to producer that can be sent
// through Binder.
class FrameEventHistoryDelta
        : public Flattenable<FrameEventHistoryDelta> {

friend class ConsumerFrameEventHistory;
friend class ProducerFrameEventHistory;

public:
    FrameEventHistoryDelta() = default;

    // Movable.
    FrameEventHistoryDelta(FrameEventHistoryDelta&& src) = default;
    FrameEventHistoryDelta& operator=(FrameEventHistoryDelta&& src);
    // Not copyable.
    FrameEventHistoryDelta(const FrameEventHistoryDelta& src) = delete;
    FrameEventHistoryDelta& operator=(
            const FrameEventHistoryDelta& src) = delete;

    // Flattenable implementation.
    size_t getFlattenedSize() const;
    size_t getFdCount() const;
    status_t flatten(void*& buffer, size_t& size, int*& fds,
            size_t& count) const;
    status_t unflatten(void const*& buffer, size_t& size, int const*& fds,
            size_t& count);

private:
    static constexpr size_t minFlattenedSize();

    std::vector<FrameEventsDelta> mDeltas;
    CompositorTiming mCompositorTiming;
};


} // namespace android
#endif
