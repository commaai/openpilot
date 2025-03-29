#include <iostream>
#include "application.h"

int main(int argc, char *argv[]) {
#ifdef __APPLE__
  // With all sockets opened, we might hit the default limit of 256 on macOS
  util::set_file_descriptor_limit(1024);
#endif
  Application a(argc, argv);
  return a.exec();
}