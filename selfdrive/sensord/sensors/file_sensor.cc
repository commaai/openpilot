#include "file_sensor.h"

#include <string>

FileSensor::FileSensor(std::string filename, uint64_t _init_delay) : file(filename) {
  init_delay = _init_delay;
}

int FileSensor::init() {
  return file.is_open() ? 0 : 1;
}

FileSensor::~FileSensor() {
  file.close();
}

bool FileSensor::has_interrupt_enabled() {
  return false;
}