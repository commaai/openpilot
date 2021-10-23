#include "selfdrive/ui/replay/filereader.h"

#include <sys/stat.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>

#include "selfdrive/common/util.h"
#include "selfdrive/ui/replay/util.h"

namespace {

std::string getCachePath() {
  std::string cache_path = util::getenv("COMMA_CACHE", "/tmp/comma_download_cache/");
  if (cache_path.rfind('/') != cache_path.length() - 1) {
    cache_path += '/';
  }
  return cache_path;
}

std::string cacheFilePath(const std::string &url) {
  static std::once_flag once_flag;
  std::call_once(once_flag, [=]() {
    std::string cache_path = getCachePath();
    if (!util::file_exists(cache_path)) {
      mkdir(cache_path.c_str(), 0775);
    }
  });

  return getCachePath() + sha256(getUrlWithoutQuery(url));
}

}  // namespace

FileReader::FileReader(bool local_cache, int max_retries) : cache_to_local_(local_cache), max_retries_(max_retries) {}

FileReader::~FileReader() {
  abort();
}

std::string FileReader::read(const std::string &file) {
  const bool is_remote = file.find("https://") == 0;
  const std::string local_file = is_remote ? cacheFilePath(file) : file;
  std::string file_content;

  if ((!is_remote || cache_to_local_) && util::file_exists(local_file)) {
    file_content = util::read_file(local_file);
  } else if (is_remote) {
    file_content = download(file);
    // write to local file cache
    if (cache_to_local_ && !file_content.empty()) {
      std::ofstream fs(local_file, fs.binary | fs.out);
      fs.write(file_content.data(), file_content.size());
    }
  }
  return file_content;
}

std::string FileReader::download(const std::string &url) {
  const int chunk_size = 20 * 1024 * 1024;  // 20MB
  size_t remote_file_size = 0;
  std::string content;

  for (int i = 1; i <= max_retries_; ++i) {
    if (remote_file_size <= 0) {
      remote_file_size = getRemoteFileSize(url);
    }
    if (remote_file_size > 0 && !abort_) {
      std::ostringstream oss;
      content.resize(remote_file_size);
      oss.rdbuf()->pubsetbuf(content.data(), content.size());
      int chunks = std::min(1, (int)std::nearbyint(remote_file_size / (float)chunk_size));
      bool ret = httpMultiPartDownload(url, oss, chunks, remote_file_size, &abort_);
      if (ret) {
        return content;
      }
    }
    if (abort_) break;

    std::cout << "download failed, retrying" << i << std::endl;
  }
  return {};
}
