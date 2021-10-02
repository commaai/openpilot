#include "selfdrive/boardd/panda_flashing.h"
#include <iostream>
#include <algorithm>

int main(int argc, char* argv[]){
  std::string serial(argv[1]);
  update_panda(serial);
}