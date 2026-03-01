#include "tools/replay/filereader.h"

#include "common/util.h"
#include "tools/replay/py_downloader.h"

std::string FileReader::read(const std::string &file, std::atomic<bool> *abort) {
  if (file.find("https://") == 0 || file.find("http://") == 0) {
    std::string local_path = PyDownloader::download(file, cache_to_local_, abort);
    if (local_path.empty()) return {};
    return util::read_file(local_path);
  }
  return util::read_file(file);
}
