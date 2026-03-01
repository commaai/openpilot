#include "tools/replay/py_downloader.h"

#include <mutex>

#include "tools/replay/util.h"

namespace {

static std::mutex handler_mutex;
static DownloadProgressHandler progress_handler = nullptr;
static DownloadHandlerFn download_handler = nullptr;

}  // namespace

void installDownloadProgressHandler(DownloadProgressHandler handler) {
  std::lock_guard<std::mutex> lk(handler_mutex);
  progress_handler = handler;
}

void installDownloadHandler(DownloadHandlerFn handler) {
  std::lock_guard<std::mutex> lk(handler_mutex);
  download_handler = handler;
}

namespace PyDownloader {

std::string download(const std::string &url, bool use_cache, std::atomic<bool> *abort) {
  DownloadHandlerFn handler_copy;
  {
    std::lock_guard<std::mutex> lk(handler_mutex);
    handler_copy = download_handler;
  }
  if (handler_copy) {
    std::string result;
    handler_copy(url, use_cache, &result);
    if (result.empty()) {
      DownloadProgressHandler progress_copy;
      {
        std::lock_guard<std::mutex> lk(handler_mutex);
        progress_copy = progress_handler;
      }
      if (progress_copy) {
        progress_copy(0, 0, false);
      }
    }
    return result;
  }
  rWarning("py_downloader: no download handler installed");
  return {};
}

}  // namespace PyDownloader
