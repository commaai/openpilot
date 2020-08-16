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
  virtual std::string receive() = 0;
  virtual void set_power(bool power) = 0;
};

class PandaPigeon : public Pigeon {
  Panda * panda = NULL;
public:
  ~PandaPigeon();
  void connect(Panda * p);
  void set_baud(int baud);
  void send(std::string s);
  std::string receive();
  void set_power(bool power);
};


class TTYPigeon : public Pigeon {
  int pigeon_tty_fd = -1;
  struct termios pigeon_tty;
public:
  ~TTYPigeon();
  void connect(const char* tty);
  void set_baud(int baud);
  void send(std::string s);
  std::string receive();
  void set_power(bool power);
};
