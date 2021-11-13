#pragma once

#include <termios.h>

#include <atomic>
#include <string>

#include "selfdrive/boardd/panda.h"

class Pigeon {
 public:
  static Pigeon* connect(Panda * p);
  static Pigeon* connect(const char * tty);
  virtual ~Pigeon(){};

  void init();
  void stop();
  bool wait_for_ack();
  bool wait_for_ack(const std::string &ack, const std::string &nack, int timeout_ms = 1000);
  bool send_with_ack(const std::string &cmd);
  virtual void set_baud(int baud) = 0;
  virtual void send(const std::string &s) = 0;
  virtual std::string receive() = 0;
  virtual void set_power(bool power) = 0;
};

class PandaPigeon : public Pigeon {
  Panda * panda = NULL;
public:
  ~PandaPigeon();
  void connect(Panda * p);
  void set_baud(int baud);
  void send(const std::string &s);
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
  void send(const std::string &s);
  std::string receive();
  void set_power(bool power);
};
