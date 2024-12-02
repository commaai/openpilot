#include "tools/replay/route.h"

#include <array>
#include <filesystem>
#include <regex>

#include "third_party/json11/json11.hpp"
#include "system/hardware/hw.h"
#include "tools/replay/api.h"
#include "tools/replay/replay.h"
#include "tools/replay/util.h"

Route::Route(const std::string &route, const std::string &data_dir) : data_dir_(data_dir) {
  route_ = parseRoute(route);
}

RouteIdentifier Route::parseRoute(const std::string &str) {
  RouteIdentifier identifier = {};
  static const std::regex pattern(R"(^(([a-z0-9]{16})[|_/])?(.{20})((--|/)((-?\d+(:(-?\d+)?)?)|(:-?\d+)))?$)");
  std::smatch match;

  if (std::regex_match(str, match, pattern)) {
    identifier.dongle_id = match[2].str();
    identifier.timestamp = match[3].str();
    identifier.str = identifier.dongle_id + "|" + identifier.timestamp;

    const auto separator = match[5].str();
    const auto range_str = match[6].str();
    if (!range_str.empty()) {
      if (separator == "/") {
        int pos = range_str.find(':');
        int begin_seg = std::stoi(range_str.substr(0, pos));
        identifier.begin_segment = identifier.end_segment = begin_seg;
        if (pos != std::string::npos) {
          auto end_seg_str = range_str.substr(pos + 1);
          identifier.end_segment = end_seg_str.empty() ? -1 : std::stoi(end_seg_str);
        }
      } else if (separator == "--") {
        identifier.begin_segment = std::atoi(range_str.c_str());
      }
    }
  }
  return identifier;
}

bool Route::load() {
  err_ = RouteLoadError::None;
  if (route_.str.empty() || (data_dir_.empty() && route_.dongle_id.empty())) {
    rInfo("invalid route format");
    return false;
  }

  struct tm tm_time = {0};
  strptime(route_.timestamp.c_str(), "%Y-%m-%d--%H-%M-%S", &tm_time);
  date_time_ = mktime(&tm_time);

  bool ret = data_dir_.empty() ? loadFromServer() : loadFromLocal();
  if (ret) {
    if (route_.begin_segment == -1) route_.begin_segment = segments_.rbegin()->first;
    if (route_.end_segment == -1) route_.end_segment = segments_.rbegin()->first;
    for (auto it = segments_.begin(); it != segments_.end(); /**/) {
      if (it->first < route_.begin_segment || it->first > route_.end_segment) {
        it = segments_.erase(it);
      } else {
        ++it;
      }
    }
  }
  return !segments_.empty();
}

bool Route::loadFromServer(int retries) {
  const std::string url = CommaApi2::BASE_URL + "/v1/route/" + route_.str + "/files";
  for (int i = 1; i <= retries; ++i) {
    long response_code = 0;
    std::string result = CommaApi2::httpGet(url, &response_code);
    if (response_code == 200) {
      return loadFromJson(result);
    }

    if (response_code == 401 || response_code == 403) {
      rWarning(">> Unauthorized. Authenticate with tools/lib/auth.py <<");
      err_ = RouteLoadError::Unauthorized;
      break;
    }
    if (response_code == 404) {
      rWarning("The specified route could not be found on the server.");
      err_ = RouteLoadError::FileNotFound;
      break;
    }

    err_ = RouteLoadError::NetworkError;
    rWarning("Retrying %d/%d", i, retries);
    util::sleep_for(3000);
  }

  return false;
}

bool Route::loadFromJson(const std::string &json) {
  const static std::regex rx(R"(\/(\d+)\/)");
  std::string err;
  auto jsonData = json11::Json::parse(json, err);
  if (!err.empty()) {
    rWarning("JSON parsing error: %s", err.c_str());
    return false;
  }
  for (const auto &value : jsonData.object_items()) {
    const auto &urlArray = value.second.array_items();
    for (const auto &url : urlArray) {
      std::string url_str = url.string_value();
      std::smatch match;
      if (std::regex_search(url_str, match, rx)) {
        addFileToSegment(std::stoi(match[1]), url_str);
      }
    }
  }
  return !segments_.empty();
}

bool Route::loadFromLocal() {
  std::string pattern = route_.timestamp + "--";
  for (const auto &entry : std::filesystem::directory_iterator(data_dir_)) {
    if (entry.is_directory() && entry.path().filename().string().find(pattern) == 0) {
      std::string segment = entry.path().string();
      int seg_num = std::atoi(segment.substr(segment.rfind("--") + 2).c_str());

      for (const auto &file : std::filesystem::directory_iterator(segment)) {
        if (file.is_regular_file()) {
          addFileToSegment(seg_num, file.path().string());
        }
      }
    }
  }
  return !segments_.empty();
}

void Route::addFileToSegment(int n, const std::string &file) {
  std::string name = extractFileName(file);
  auto pos = name.find_last_of("--");
  name = pos != std::string::npos ? name.substr(pos + 2) : name;

  if (name == "rlog.bz2" || name == "rlog.zst" || name == "rlog") {
    segments_[n].rlog = file;
  } else if (name == "qlog.bz2" || name == "qlog.zst" || name == "qlog") {
    segments_[n].qlog = file;
  } else if (name == "fcamera.hevc") {
    segments_[n].road_cam = file;
  } else if (name == "dcamera.hevc") {
    segments_[n].driver_cam = file;
  } else if (name == "ecamera.hevc") {
    segments_[n].wide_road_cam = file;
  } else if (name == "qcamera.ts") {
    segments_[n].qcamera = file;
  }
}

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
    success = frames[id]->load((CameraType)id, file, flags & REPLAY_FLAG_NO_HW_DECODER, &abort_, local_cache, 20 * 1024 * 1024, 3);
  } else {
    log = std::make_unique<LogReader>(filters_);
    success = log->load(file, &abort_, local_cache, 0, 3);
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
