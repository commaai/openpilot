#include <assert.h>
#include <string>
#include "common/swaglog.h"
#include "logger.h"

int main(int argc, char** argv) {
  char filename[64] = {'\0'};

  time_t rawtime = time(NULL);
  struct tm timeinfo;

  localtime_r(&rawtime, &timeinfo);
  strftime(filename, sizeof(filename),
           "%Y-%m-%d--%H-%M-%S.bz2", &timeinfo);

  std::string path = LOG_ROOT + "/boot/" + std::string(filename);
  LOGW("bootlog to %s", path.c_str());

  // Open bootlog
  int r = logger_mkpath((char*)path.c_str());
  assert(r == 0);

  BZFile bz_file(path.c_str());

  // Write initdata
  bz_file.write(logger_build_init_data().asBytes());

  // Write bootlog
  bz_file.write(logger_build_boot().asBytes());

  return 0;
}
