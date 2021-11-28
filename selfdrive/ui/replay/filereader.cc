#include "selfdrive/ui/replay/filereader.h"

#include <sys/stat.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

#include "selfdrive/common/util.h"
#include "selfdrive/ui/replay/util.h"

std::string cacheFilePath(const std::string &url) {
  static std::string cache_path = [] {
    const std::string comma_cache = util::getenv("COMMA_CACHE", "/tmp/comma_download_cache/");
    util::create_directories(comma_cache, 0755);
    return comma_cache.back() == '/' ? comma_cache : comma_cache + "/";
  }();

  return cache_path + sha256(getUrlWithoutQuery(url));
}

std::string FileReader::read(const std::string &file, std::atomic<bool> *abort) {
  const bool is_remote = file.find("https://") == 0;
  const std::string local_file = is_remote ? cacheFilePath(file) : file;
  std::string result;

  if ((!is_remote || cache_to_local_) && util::file_exists(local_file)) {
    result = util::read_file(local_file);
  } else if (is_remote) {
    result = download(file, abort);
    if (cache_to_local_ && !result.empty()) {
      std::ofstream fs(local_file, std::ofstream::binary | std::ofstream::out);
      fs.write(result.data(), result.size());
    }
  }
  return result;
}

std::string FileReader::download(const std::string &url, std::atomic<bool> *abort) {
  std::string result;
  size_t remote_file_size = 0;
  for (int i = 0; i <= max_retries_ && !(abort && *abort); ++i) {
    if (i > 0) {
      std::cout << "download failed, retrying" << i << std::endl;
    }
    if (remote_file_size <= 0) {
      remote_file_size = getRemoteFileSize(url);
    }
    if (remote_file_size > 0 && !(abort && *abort)) {
      int chunks = chunk_size_ > 0 ? std::max(1, (int)std::nearbyint(remote_file_size / (float)chunk_size_)) : 1;
      result = httpGet(url, chunks, remote_file_size, abort);
      if (!result.empty()) {
        return result;
      }
    }
  }
  return {};
}
