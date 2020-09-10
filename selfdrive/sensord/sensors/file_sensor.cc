#include <iostream>
#include <string>

#include "file_sensor.hpp"

FileSensor::FileSensor(std::string filename) : file(filename) {
}

int FileSensor::init() {
  return file.is_open() ? 0 : 1;
}

void FileSensor::get_event(cereal::SensorEventData::Builder &event){
  file.clear();
  file.seekg(0);

  std::string line;
  std::getline(file, line);
  std::cout << line;
}


FileSensor::~FileSensor(){
  file.close();
}
