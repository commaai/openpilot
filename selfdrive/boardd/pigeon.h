#pragma once
#include <string>
#include <termios.h>


#include "panda.h"

class Pigeon {
 public:
  static Pigeon* connect(Panda * p);
  static Pigeon* connect(const char * tty);
  virtual ~Pigeon(){};

  void init();
  virtual void set_baud(int baud) = 0;
  virtual void send(std::string s) = 0;
  virtual int receive(unsigned char * dat) = 0;
  virtual void set_power(bool power) = 0;
};

class PandaPigeon : public Pigeon {
  Panda * panda = NULL;
public:
  void connect(Panda * p);
  void set_baud(int baud);
  void send(std::string s);
  int receive(unsigned char * dat);
  void set_power(bool power);
};


class TTYPigeon : public Pigeon {
  int pigeon_tty_fd = -1;
  struct termios pigeon_tty;
public:
  void connect(const char* tty);
  void set_baud(int baud);
  void send(std::string s);
  int receive(unsigned char * dat);
  void set_power(bool power);
};
