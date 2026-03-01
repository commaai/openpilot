#pragma once

#include <ctime>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "tools/replay/framereader.h"
#include "tools/replay/logreader.h"
#include "tools/replay/util.h"

enum REPLAY_FLAGS {
  REPLAY_FLAG_NONE = 0x0000,
  REPLAY_FLAG_DCAM = 0x0002,
  REPLAY_FLAG_ECAM = 0x0004,
  REPLAY_FLAG_NO_LOOP = 0x0010,
  REPLAY_FLAG_NO_FILE_CACHE = 0x0020,
  REPLAY_FLAG_QCAMERA = 0x0040,
  REPLAY_FLAG_NO_HW_DECODER = 0x0100,
  REPLAY_FLAG_NO_VIPC = 0x0400,
  REPLAY_FLAG_ALL_SERVICES = 0x0800,
  REPLAY_FLAG_BENCHMARK = 0x1000,
};

struct SegmentFile {
  std::string rlog;
  std::string qlog;
  std::string road_cam;
  std::string driver_cam;
  std::string wide_road_cam;
  std::string qcamera;
};

class Route {
public:
  Route() = default;

  void addSegment(int n, const std::string &rlog, const std::string &qlog,
                  const std::string &road_cam, const std::string &driver_cam,
                  const std::string &wide_road_cam, const std::string &qcamera) {
    segments_[n] = {rlog, qlog, road_cam, driver_cam, wide_road_cam, qcamera};
  }

  void setName(const std::string &name) { name_ = name; }
  void setDateTime(std::time_t dt) { date_time_ = dt; }

  const std::string &name() const { return name_; }
  std::time_t datetime() const { return date_time_; }
  const std::map<int, SegmentFile> &segments() const { return segments_; }
  const SegmentFile &at(int n) const { return segments_.at(n); }

private:
  std::string name_;
  std::map<int, SegmentFile> segments_;
  std::time_t date_time_ = 0;
};

class Segment {
public:
  enum class LoadState {Loading, Loaded, Failed};

  Segment(int n, const SegmentFile &files, uint32_t flags, const std::vector<bool> &filters,
          std::function<void(int, bool)> callback);
  ~Segment();
  LoadState getState();

  const int seg_num = 0;
  std::unique_ptr<LogReader> log;
  std::unique_ptr<FrameReader> frames[MAX_CAMERAS] = {};

protected:
  void loadFile(int id, const std::string file);

  std::atomic<bool> abort_ = false;
  std::atomic<int> loading_ = 0;
  std::mutex mutex_;
  std::vector<std::thread> threads_;
  std::function<void(int, bool)> on_load_finished_ = nullptr;
  uint32_t flags;
  std::vector<bool> filters_;
  LoadState load_state_  = LoadState::Loading;
};
