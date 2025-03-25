#include <iostream>
#include "application.h"

int main(int argc, char *argv[]) {
  Application a(argc, argv);

  if (a.exec()) {
    std::cerr << "Failed to start app." << std::endl;
  }

  return 0;
}