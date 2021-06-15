#include "file_sensor.h"

#include <string>

FileSensor::FileSensor(std::string filename) : file(filename) {
}

int FileSensor::init() {
  return file.is_open() ? 0 : 1;
}

FileSensor::~FileSensor() {
  file.close();
}
