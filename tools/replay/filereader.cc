#include "tools/replay/filereader.h"

#include <fstream>

#include "common/util.h"
#include "system/hardware/hw.h"
#include "tools/replay/util.h"

std::string cacheFilePath(const std::string &url) {
  static std::string cache_path = [] {
    const std::string comma_cache = Path::download_cache_root();
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
      std::ofstream fs(local_file, std::ios::binary | std::ios::out);
      fs.write(result.data(), result.size());
    }
  }
  return result;
}

std::string FileReader::download(const std::string &url, std::atomic<bool> *abort) {
  for (int i = 0; i <= max_retries_ && !(abort && *abort); ++i) {
    if (i > 0) rWarning("download failed, retrying %d", i);

    std::string result = httpGet(url, chunk_size_, abort);
    if (!result.empty()) {
      return result;
    }
  }
  return {};
}
