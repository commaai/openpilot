#include <assert.h>
#include <string>
#include "common/swaglog.h"
#include "logger.h"

int main(int argc, char** argv) {

  const std::string path = LOG_ROOT + "/boot/" + logger_get_route_name();
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
