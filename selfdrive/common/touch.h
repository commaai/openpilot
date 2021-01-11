#pragma once

class TouchState {
public:
  TouchState();
  ~TouchState();
  bool poll(int *out_x, int *out_y, int timeout = 0);
  int fd;
  int last_x, last_y;
};

