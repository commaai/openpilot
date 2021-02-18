#include "panda_flashing.hpp"
#include <iostream>

int main(int argc, char* argv[]){
  std::string serial(argv[0]);
  std::string dfu_serial(argv[1]);
  update_panda();
  std::cout<<serial<<" "<<dfu_serial<<std::endl;
}