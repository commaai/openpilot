#ifndef RUNMODEL_H
#define RUNMODEL_H

class RunModel {
public:
  virtual void addRecurrent(float *state, int state_size) {}
  virtual void addDesire(float *state, int state_size) {}
  virtual void execute(float *net_input_buf) {}
};

#endif

