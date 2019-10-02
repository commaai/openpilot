#include <algorithm>
#include <cmath>
#include <iostream>

#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include "cereal/gen/cpp/log.capnp.h"
#include "cereal/gen/cpp/car.capnp.h"
#include "params_learner.h"

// #define DEBUG

template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
  return std::max(lower, std::min(n, upper));
}

ParamsLearner::ParamsLearner(cereal::CarParams::Reader car_params,
                             double angle_offset,
                             double stiffness_factor,
                             double steer_ratio,
                             double learning_rate) :
  ao(angle_offset * DEGREES_TO_RADIANS),
  slow_ao(angle_offset * DEGREES_TO_RADIANS),
  x(stiffness_factor),
  sR(steer_ratio) {

  cF0 = car_params.getTireStiffnessFront();
  cR0 = car_params.getTireStiffnessRear();

  l = car_params.getWheelbase();
  m = car_params.getMass();

  aF = car_params.getCenterToFront();
  aR = l - aF;

  min_sr = MIN_SR * car_params.getSteerRatio();
  max_sr = MAX_SR * car_params.getSteerRatio();
  min_sr_th = MIN_SR_TH * car_params.getSteerRatio();
  max_sr_th = MAX_SR_TH * car_params.getSteerRatio();
  alpha1 = 0.01 * learning_rate;
  alpha2 = 0.0005 * learning_rate;
  alpha3 = 0.1 * learning_rate;
  alpha4 = 1.0 * learning_rate;
}

bool ParamsLearner::update(double psi, double u, double sa) {
  if (u > 10.0 && fabs(sa) < (DEGREES_TO_RADIANS * 90.)) {
    double ao_diff = 2.0*cF0*cR0*l*u*x*(1.0*cF0*cR0*l*u*x*(ao - sa) + psi*sR*(cF0*cR0*pow(l, 2)*x - m*pow(u, 2)*(aF*cF0 - aR*cR0)))/(pow(sR, 2)*pow(cF0*cR0*pow(l, 2)*x - m*pow(u, 2)*(aF*cF0 - aR*cR0), 2));
    double new_ao = ao - alpha1 * ao_diff;

    double slow_ao_diff = 2.0*cF0*cR0*l*u*x*(1.0*cF0*cR0*l*u*x*(slow_ao - sa) + psi*sR*(cF0*cR0*pow(l, 2)*x - m*pow(u, 2)*(aF*cF0 - aR*cR0)))/(pow(sR, 2)*pow(cF0*cR0*pow(l, 2)*x - m*pow(u, 2)*(aF*cF0 - aR*cR0), 2));
    double new_slow_ao = slow_ao - alpha2 * slow_ao_diff;

    double new_x = x - alpha3 * (-2.0*cF0*cR0*l*m*pow(u, 3)*(slow_ao - sa)*(aF*cF0 - aR*cR0)*(1.0*cF0*cR0*l*u*x*(slow_ao - sa) + psi*sR*(cF0*cR0*pow(l, 2)*x - m*pow(u, 2)*(aF*cF0 - aR*cR0)))/(pow(sR, 2)*pow(cF0*cR0*pow(l, 2)*x - m*pow(u, 2)*(aF*cF0 - aR*cR0), 3)));
    double new_sR = sR - alpha4 * (-2.0*cF0*cR0*l*u*x*(slow_ao - sa)*(1.0*cF0*cR0*l*u*x*(slow_ao - sa) + psi*sR*(cF0*cR0*pow(l, 2)*x - m*pow(u, 2)*(aF*cF0 - aR*cR0)))/(pow(sR, 3)*pow(cF0*cR0*pow(l, 2)*x - m*pow(u, 2)*(aF*cF0 - aR*cR0), 2)));

    ao = new_ao;
    slow_ao = new_slow_ao;
    x = new_x;
    sR = new_sR;
  }

#ifdef DEBUG
  std::cout << "Instant AO: " << (RADIANS_TO_DEGREES * ao) << "\tAverage AO: " << (RADIANS_TO_DEGREES * slow_ao);
  std::cout << "\tStiffness: " << x << "\t sR: " << sR << std::endl;
#endif

  ao = clip(ao, -MAX_ANGLE_OFFSET, MAX_ANGLE_OFFSET);
  slow_ao = clip(slow_ao, -MAX_ANGLE_OFFSET, MAX_ANGLE_OFFSET);
  x = clip(x, MIN_STIFFNESS, MAX_STIFFNESS);
  sR = clip(sR, min_sr, max_sr);

  bool valid = fabs(slow_ao) < MAX_ANGLE_OFFSET_TH;
  valid = valid && sR > min_sr_th;
  valid = valid && sR < max_sr_th;
  return valid;
}


extern "C" {
  void *params_learner_init(size_t len, char * params, double angle_offset, double stiffness_factor, double steer_ratio, double learning_rate) {

    auto amsg = kj::heapArray<capnp::word>((len / sizeof(capnp::word)) + 1);
    memcpy(amsg.begin(), params, len);

    capnp::FlatArrayMessageReader cmsg(amsg);
    cereal::CarParams::Reader car_params = cmsg.getRoot<cereal::CarParams>();

    ParamsLearner * p = new ParamsLearner(car_params, angle_offset, stiffness_factor, steer_ratio, learning_rate);
    return (void*)p;
  }

  bool params_learner_update(void * params_learner, double psi, double u, double sa) {
    ParamsLearner * p = (ParamsLearner*) params_learner;
    return p->update(psi, u, sa);
  }

  double params_learner_get_ao(void * params_learner){
    ParamsLearner * p = (ParamsLearner*) params_learner;
    return p->ao;
  }

  double params_learner_get_x(void * params_learner){
    ParamsLearner * p = (ParamsLearner*) params_learner;
    return p->x;
  }

  double params_learner_get_slow_ao(void * params_learner){
    ParamsLearner * p = (ParamsLearner*) params_learner;
    return p->slow_ao;
  }

  double params_learner_get_sR(void * params_learner){
    ParamsLearner * p = (ParamsLearner*) params_learner;
    return p->sR;
  }
}
