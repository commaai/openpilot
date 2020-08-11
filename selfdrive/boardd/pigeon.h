#pragma once
#include <string>

#include "panda.h"

class Pigeon {
 public:
  static Pigeon* create(Panda * p);
  virtual ~Pigeon(){};

  void init();
  virtual void set_baud(int baud) = 0;
  virtual void send(std::string s) = 0;
  virtual int receive(unsigned char * dat) = 0;
  virtual void set_power(int power) = 0;
};

class PandaPigeon : public Pigeon {
  Panda * panda = NULL;
public:
  void set_panda(Panda * p);
  void set_baud(int baud);
  void send(std::string s);
  int receive(unsigned char * dat);
  void set_power(int power);
};
