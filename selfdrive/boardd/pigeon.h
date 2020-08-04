#pragma once
#include <string>

#include "panda.h"

class Pigeon {
  Panda * panda;

 public:
  Pigeon(Panda * panda);
  void init();
  void set_baud(int baud);
  void send(std::string s);
  int receive(unsigned char * dat);
  void set_power(int power);
};
