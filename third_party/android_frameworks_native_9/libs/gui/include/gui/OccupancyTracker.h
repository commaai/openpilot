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


#ifndef ANDROID_GUI_OCCUPANCYTRACKER_H
#define ANDROID_GUI_OCCUPANCYTRACKER_H

#include <binder/Parcelable.h>

#include <utils/Timers.h>

#include <deque>
#include <unordered_map>

namespace android {

class String8;

class OccupancyTracker
{
public:
    OccupancyTracker()
      : mPendingSegment(),
        mSegmentHistory(),
        mLastOccupancy(0),
        mLastOccupancyChangeTime(0) {}

    struct Segment : public Parcelable {
        Segment()
          : totalTime(0),
            numFrames(0),
            occupancyAverage(0.0f),
            usedThirdBuffer(false) {}

        Segment(nsecs_t _totalTime, size_t _numFrames, float _occupancyAverage,
                bool _usedThirdBuffer)
          : totalTime(_totalTime),
            numFrames(_numFrames),
            occupancyAverage(_occupancyAverage),
            usedThirdBuffer(_usedThirdBuffer) {}

        // Parcelable interface
        virtual status_t writeToParcel(Parcel* parcel) const override;
        virtual status_t readFromParcel(const Parcel* parcel) override;

        nsecs_t totalTime;
        size_t numFrames;

        // Average occupancy of the queue over this segment. (0.0, 1.0) implies
        // double-buffered, (1.0, 2.0) implies triple-buffered.
        float occupancyAverage;

        // Whether a third buffer was used at all during this segment (since a
        // segment could read as double-buffered on average, but still require a
        // third buffer to avoid jank for some smaller portion)
        bool usedThirdBuffer;
    };

    void registerOccupancyChange(size_t occupancy);
    std::vector<Segment> getSegmentHistory(bool forceFlush);

private:
    static constexpr size_t MAX_HISTORY_SIZE = 10;
    static constexpr nsecs_t NEW_SEGMENT_DELAY = ms2ns(100);
    static constexpr size_t LONG_SEGMENT_THRESHOLD = 3;

    struct PendingSegment {
        void clear() {
            totalTime = 0;
            numFrames = 0;
            mOccupancyTimes.clear();
        }

        nsecs_t totalTime;
        size_t numFrames;
        std::unordered_map<size_t, nsecs_t> mOccupancyTimes;
    };

    void recordPendingSegment();

    PendingSegment mPendingSegment;
    std::deque<Segment> mSegmentHistory;

    size_t mLastOccupancy;
    nsecs_t mLastOccupancyChangeTime;

}; // class OccupancyTracker

} // namespace android

#endif
