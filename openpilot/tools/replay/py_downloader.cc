#include "tools/replay/py_downloader.h"

#include <mutex>
#include <vector>

#include "tools/replay/py_process.h"

namespace {

static std::mutex handler_mutex;
static DownloadProgressHandler progress_handler = nullptr;

// Run the file_downloader module and notify the progress handler on failure.
std::string runDownloader(const std::vector<std::string> &args, std::atomic<bool> *abort = nullptr) {
  std::string result = PyProcess::runModule("openpilot.tools.lib.file_downloader", args, abort);
  if (result.empty()) {
    std::lock_guard<std::mutex> lk(handler_mutex);
    if (progress_handler) progress_handler(0, 0, false);
  }
  return result;
}

}  // namespace

void installDownloadProgressHandler(DownloadProgressHandler handler) {
  std::lock_guard<std::mutex> lk(handler_mutex);
  progress_handler = handler;
}

namespace PyDownloader {

std::string download(const std::string &url, bool use_cache, std::atomic<bool> *abort) {
  std::vector<std::string> args = {"download", url};
  if (!use_cache) {
    args.push_back("--no-cache");
  }
  return runDownloader(args, abort);
}

std::string decompress(const std::string &path, std::atomic<bool> *abort) {
  return runDownloader({"decompress", path}, abort);
}

std::string getRouteFiles(const std::string &route) {
  return runDownloader({"route-files", route});
}

std::string getDevices() {
  return runDownloader({"devices"});
}

std::string getDeviceRoutes(const std::string &dongle_id, int64_t start_ms, int64_t end_ms, bool preserved) {
  std::vector<std::string> args = {"device-routes", dongle_id};
  if (preserved) {
    args.push_back("--preserved");
  } else {
    if (start_ms > 0) {
      args.push_back("--start");
      args.push_back(std::to_string(start_ms));
    }
    if (end_ms > 0) {
      args.push_back("--end");
      args.push_back(std::to_string(end_ms));
    }
  }
  return runDownloader(args);
}

}  // namespace PyDownloader
