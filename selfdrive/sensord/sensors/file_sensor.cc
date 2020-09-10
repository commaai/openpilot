#include <iostream>
#include <string>

#include "file_sensor.hpp"

FileSensor::FileSensor(std::string filename) : file(filename) {
}

int FileSensor::init() {
  return file.is_open() ? 0 : 1;
}

FileSensor::~FileSensor(){
  file.close();
}
