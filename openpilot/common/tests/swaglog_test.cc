#include <chrono>
#include <thread>

#include "common/swaglog.h"

int main() {
  LOGD("python-e2e-cpp-log");
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  return 0;
}
