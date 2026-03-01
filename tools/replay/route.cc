#include "tools/replay/route.h"

#include <array>

#include "tools/replay/py_downloader.h"
#include "tools/replay/util.h"

// class Segment

Segment::Segment(int n, const SegmentFile &files, uint32_t flags, const std::vector<bool> &filters,
                 std::function<void(int, bool)> callback)
    : seg_num(n), flags(flags), filters_(filters), on_load_finished_(callback) {
  // [RoadCam, DriverCam, WideRoadCam, log]. fallback to qcamera/qlog
  const std::array file_list = {
      (flags & REPLAY_FLAG_QCAMERA) || files.road_cam.empty() ? files.qcamera : files.road_cam,
      flags & REPLAY_FLAG_DCAM ? files.driver_cam : "",
      flags & REPLAY_FLAG_ECAM ? files.wide_road_cam : "",
      files.rlog.empty() ? files.qlog : files.rlog,
  };
  for (int i = 0; i < file_list.size(); ++i) {
    if (!file_list[i].empty() && (!(flags & REPLAY_FLAG_NO_VIPC) || i >= MAX_CAMERAS)) {
      ++loading_;
      threads_.emplace_back(&Segment::loadFile, this, i, file_list[i]);
    }
  }
}

Segment::~Segment() {
  {
    std::lock_guard lock(mutex_);
    on_load_finished_ = nullptr;  // Prevent callback after destruction
  }
  abort_ = true;
  for (auto &thread : threads_) {
    if (thread.joinable()) thread.join();
  }
}

void Segment::loadFile(int id, const std::string file) {
  const bool local_cache = !(flags & REPLAY_FLAG_NO_FILE_CACHE);
  bool success = false;
  if (id < MAX_CAMERAS) {
    frames[id] = std::make_unique<FrameReader>();
    success = frames[id]->load((CameraType)id, file, flags & REPLAY_FLAG_NO_HW_DECODER, &abort_, local_cache);
  } else {
    log = std::make_unique<LogReader>(filters_);
    success = log->load(file, &abort_, local_cache);
  }

  if (!success) {
    // abort all loading jobs.
    abort_ = true;
  }

  if (--loading_ == 0) {
    std::lock_guard lock(mutex_);
    load_state_ = !abort_ ? LoadState::Loaded : LoadState::Failed;
    if (on_load_finished_) {
      on_load_finished_(seg_num, !abort_);
    }
  }
}

Segment::LoadState Segment::getState() {
  std::scoped_lock lock(mutex_);
  return load_state_;
}
