#include "tools/replay/py_tools.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

#include "tools/replay/py_process.h"

namespace {

constexpr const char *DOWNLOADER_MODULE = "openpilot.tools.lib.file_downloader";

static std::mutex handler_mutex;
static DownloadProgressHandler progress_handler = nullptr;

void reportProgress(const char *line) {
  uint64_t cur = 0, total = 0;
  if (sscanf(line, "PROGRESS:%llu:%llu", (unsigned long long *)&cur, (unsigned long long *)&total) != 2) return;
  std::lock_guard<std::mutex> lk(handler_mutex);
  if (progress_handler && total > 0) progress_handler(cur, total, true);
}

// Run a Python module, report download progress from PROGRESS lines on stderr,
// and notify the progress handler on failure.
std::string runModuleWithProgress(const std::string &module, const std::vector<std::string> &args, std::atomic<bool> *abort = nullptr) {
  std::string result = PyProcess::runModule(module, args, abort, true, [](const char *line) {
    if (strncmp(line, "PROGRESS:", 9) == 0) {
      reportProgress(line);
    } else {
      fputs(line, stderr);
    }
  });
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

namespace PyTools {

std::string download(const std::string &url, bool use_cache, std::atomic<bool> *abort) {
  std::vector<std::string> args = {"download", url};
  if (!use_cache) {
    args.push_back("--no-cache");
  }
  return runModuleWithProgress(DOWNLOADER_MODULE, args, abort);
}

std::string decompress(const std::string &path, std::atomic<bool> *abort) {
  return runModuleWithProgress(DOWNLOADER_MODULE, {"decompress", path}, abort);
}

std::string getRouteFiles(const std::string &route) {
  return runModuleWithProgress(DOWNLOADER_MODULE, {"route-files", route});
}

}  // namespace PyTools
