#include "panda_flashing.hpp"
#include <iostream>

int main(int argc, char* argv[]){
  std::string serial(argv[1]);
  std::string dfu_serial(argv[2]);
  update_panda(serial="", dfu_serial="");
  std::cout<<"console parsing"<<std::endl;
  std::cout<<serial<<" "<<dfu_serial<<std::endl;
}