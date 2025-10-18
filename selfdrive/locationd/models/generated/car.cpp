#include "car.h"

namespace {
#define DIM 9
#define EDIM 9
#define MEDIM 9
typedef void (*Hfun)(double *, double *, double *);

double mass;

void set_mass(double x){ mass = x;}

double rotational_inertia;

void set_rotational_inertia(double x){ rotational_inertia = x;}

double center_to_front;

void set_center_to_front(double x){ center_to_front = x;}

double center_to_rear;

void set_center_to_rear(double x){ center_to_rear = x;}

double stiffness_front;

void set_stiffness_front(double x){ stiffness_front = x;}

double stiffness_rear;

void set_stiffness_rear(double x){ stiffness_rear = x;}
const static double MAHA_THRESH_25 = 3.8414588206941227;
const static double MAHA_THRESH_24 = 5.991464547107981;
const static double MAHA_THRESH_30 = 3.8414588206941227;
const static double MAHA_THRESH_26 = 3.8414588206941227;
const static double MAHA_THRESH_27 = 3.8414588206941227;
const static double MAHA_THRESH_29 = 3.8414588206941227;
const static double MAHA_THRESH_28 = 3.8414588206941227;
const static double MAHA_THRESH_31 = 3.8414588206941227;

/******************************************************************************
 *                      Code generated with SymPy 1.14.0                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_7584201253892190001) {
   out_7584201253892190001[0] = delta_x[0] + nom_x[0];
   out_7584201253892190001[1] = delta_x[1] + nom_x[1];
   out_7584201253892190001[2] = delta_x[2] + nom_x[2];
   out_7584201253892190001[3] = delta_x[3] + nom_x[3];
   out_7584201253892190001[4] = delta_x[4] + nom_x[4];
   out_7584201253892190001[5] = delta_x[5] + nom_x[5];
   out_7584201253892190001[6] = delta_x[6] + nom_x[6];
   out_7584201253892190001[7] = delta_x[7] + nom_x[7];
   out_7584201253892190001[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_4881621785150572117) {
   out_4881621785150572117[0] = -nom_x[0] + true_x[0];
   out_4881621785150572117[1] = -nom_x[1] + true_x[1];
   out_4881621785150572117[2] = -nom_x[2] + true_x[2];
   out_4881621785150572117[3] = -nom_x[3] + true_x[3];
   out_4881621785150572117[4] = -nom_x[4] + true_x[4];
   out_4881621785150572117[5] = -nom_x[5] + true_x[5];
   out_4881621785150572117[6] = -nom_x[6] + true_x[6];
   out_4881621785150572117[7] = -nom_x[7] + true_x[7];
   out_4881621785150572117[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_6380899889402302601) {
   out_6380899889402302601[0] = 1.0;
   out_6380899889402302601[1] = 0.0;
   out_6380899889402302601[2] = 0.0;
   out_6380899889402302601[3] = 0.0;
   out_6380899889402302601[4] = 0.0;
   out_6380899889402302601[5] = 0.0;
   out_6380899889402302601[6] = 0.0;
   out_6380899889402302601[7] = 0.0;
   out_6380899889402302601[8] = 0.0;
   out_6380899889402302601[9] = 0.0;
   out_6380899889402302601[10] = 1.0;
   out_6380899889402302601[11] = 0.0;
   out_6380899889402302601[12] = 0.0;
   out_6380899889402302601[13] = 0.0;
   out_6380899889402302601[14] = 0.0;
   out_6380899889402302601[15] = 0.0;
   out_6380899889402302601[16] = 0.0;
   out_6380899889402302601[17] = 0.0;
   out_6380899889402302601[18] = 0.0;
   out_6380899889402302601[19] = 0.0;
   out_6380899889402302601[20] = 1.0;
   out_6380899889402302601[21] = 0.0;
   out_6380899889402302601[22] = 0.0;
   out_6380899889402302601[23] = 0.0;
   out_6380899889402302601[24] = 0.0;
   out_6380899889402302601[25] = 0.0;
   out_6380899889402302601[26] = 0.0;
   out_6380899889402302601[27] = 0.0;
   out_6380899889402302601[28] = 0.0;
   out_6380899889402302601[29] = 0.0;
   out_6380899889402302601[30] = 1.0;
   out_6380899889402302601[31] = 0.0;
   out_6380899889402302601[32] = 0.0;
   out_6380899889402302601[33] = 0.0;
   out_6380899889402302601[34] = 0.0;
   out_6380899889402302601[35] = 0.0;
   out_6380899889402302601[36] = 0.0;
   out_6380899889402302601[37] = 0.0;
   out_6380899889402302601[38] = 0.0;
   out_6380899889402302601[39] = 0.0;
   out_6380899889402302601[40] = 1.0;
   out_6380899889402302601[41] = 0.0;
   out_6380899889402302601[42] = 0.0;
   out_6380899889402302601[43] = 0.0;
   out_6380899889402302601[44] = 0.0;
   out_6380899889402302601[45] = 0.0;
   out_6380899889402302601[46] = 0.0;
   out_6380899889402302601[47] = 0.0;
   out_6380899889402302601[48] = 0.0;
   out_6380899889402302601[49] = 0.0;
   out_6380899889402302601[50] = 1.0;
   out_6380899889402302601[51] = 0.0;
   out_6380899889402302601[52] = 0.0;
   out_6380899889402302601[53] = 0.0;
   out_6380899889402302601[54] = 0.0;
   out_6380899889402302601[55] = 0.0;
   out_6380899889402302601[56] = 0.0;
   out_6380899889402302601[57] = 0.0;
   out_6380899889402302601[58] = 0.0;
   out_6380899889402302601[59] = 0.0;
   out_6380899889402302601[60] = 1.0;
   out_6380899889402302601[61] = 0.0;
   out_6380899889402302601[62] = 0.0;
   out_6380899889402302601[63] = 0.0;
   out_6380899889402302601[64] = 0.0;
   out_6380899889402302601[65] = 0.0;
   out_6380899889402302601[66] = 0.0;
   out_6380899889402302601[67] = 0.0;
   out_6380899889402302601[68] = 0.0;
   out_6380899889402302601[69] = 0.0;
   out_6380899889402302601[70] = 1.0;
   out_6380899889402302601[71] = 0.0;
   out_6380899889402302601[72] = 0.0;
   out_6380899889402302601[73] = 0.0;
   out_6380899889402302601[74] = 0.0;
   out_6380899889402302601[75] = 0.0;
   out_6380899889402302601[76] = 0.0;
   out_6380899889402302601[77] = 0.0;
   out_6380899889402302601[78] = 0.0;
   out_6380899889402302601[79] = 0.0;
   out_6380899889402302601[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_6392402267122999460) {
   out_6392402267122999460[0] = state[0];
   out_6392402267122999460[1] = state[1];
   out_6392402267122999460[2] = state[2];
   out_6392402267122999460[3] = state[3];
   out_6392402267122999460[4] = state[4];
   out_6392402267122999460[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8100000000000005*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_6392402267122999460[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_6392402267122999460[7] = state[7];
   out_6392402267122999460[8] = state[8];
}
void F_fun(double *state, double dt, double *out_2526673064601828008) {
   out_2526673064601828008[0] = 1;
   out_2526673064601828008[1] = 0;
   out_2526673064601828008[2] = 0;
   out_2526673064601828008[3] = 0;
   out_2526673064601828008[4] = 0;
   out_2526673064601828008[5] = 0;
   out_2526673064601828008[6] = 0;
   out_2526673064601828008[7] = 0;
   out_2526673064601828008[8] = 0;
   out_2526673064601828008[9] = 0;
   out_2526673064601828008[10] = 1;
   out_2526673064601828008[11] = 0;
   out_2526673064601828008[12] = 0;
   out_2526673064601828008[13] = 0;
   out_2526673064601828008[14] = 0;
   out_2526673064601828008[15] = 0;
   out_2526673064601828008[16] = 0;
   out_2526673064601828008[17] = 0;
   out_2526673064601828008[18] = 0;
   out_2526673064601828008[19] = 0;
   out_2526673064601828008[20] = 1;
   out_2526673064601828008[21] = 0;
   out_2526673064601828008[22] = 0;
   out_2526673064601828008[23] = 0;
   out_2526673064601828008[24] = 0;
   out_2526673064601828008[25] = 0;
   out_2526673064601828008[26] = 0;
   out_2526673064601828008[27] = 0;
   out_2526673064601828008[28] = 0;
   out_2526673064601828008[29] = 0;
   out_2526673064601828008[30] = 1;
   out_2526673064601828008[31] = 0;
   out_2526673064601828008[32] = 0;
   out_2526673064601828008[33] = 0;
   out_2526673064601828008[34] = 0;
   out_2526673064601828008[35] = 0;
   out_2526673064601828008[36] = 0;
   out_2526673064601828008[37] = 0;
   out_2526673064601828008[38] = 0;
   out_2526673064601828008[39] = 0;
   out_2526673064601828008[40] = 1;
   out_2526673064601828008[41] = 0;
   out_2526673064601828008[42] = 0;
   out_2526673064601828008[43] = 0;
   out_2526673064601828008[44] = 0;
   out_2526673064601828008[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_2526673064601828008[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_2526673064601828008[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_2526673064601828008[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_2526673064601828008[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_2526673064601828008[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_2526673064601828008[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_2526673064601828008[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_2526673064601828008[53] = -9.8100000000000005*dt;
   out_2526673064601828008[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_2526673064601828008[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_2526673064601828008[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2526673064601828008[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2526673064601828008[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_2526673064601828008[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_2526673064601828008[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_2526673064601828008[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2526673064601828008[62] = 0;
   out_2526673064601828008[63] = 0;
   out_2526673064601828008[64] = 0;
   out_2526673064601828008[65] = 0;
   out_2526673064601828008[66] = 0;
   out_2526673064601828008[67] = 0;
   out_2526673064601828008[68] = 0;
   out_2526673064601828008[69] = 0;
   out_2526673064601828008[70] = 1;
   out_2526673064601828008[71] = 0;
   out_2526673064601828008[72] = 0;
   out_2526673064601828008[73] = 0;
   out_2526673064601828008[74] = 0;
   out_2526673064601828008[75] = 0;
   out_2526673064601828008[76] = 0;
   out_2526673064601828008[77] = 0;
   out_2526673064601828008[78] = 0;
   out_2526673064601828008[79] = 0;
   out_2526673064601828008[80] = 1;
}
void h_25(double *state, double *unused, double *out_4624397152353190203) {
   out_4624397152353190203[0] = state[6];
}
void H_25(double *state, double *unused, double *out_7162152359122584175) {
   out_7162152359122584175[0] = 0;
   out_7162152359122584175[1] = 0;
   out_7162152359122584175[2] = 0;
   out_7162152359122584175[3] = 0;
   out_7162152359122584175[4] = 0;
   out_7162152359122584175[5] = 0;
   out_7162152359122584175[6] = 1;
   out_7162152359122584175[7] = 0;
   out_7162152359122584175[8] = 0;
}
void h_24(double *state, double *unused, double *out_5737550318670326813) {
   out_5737550318670326813[0] = state[4];
   out_5737550318670326813[1] = state[5];
}
void H_24(double *state, double *unused, double *out_4984937935515434202) {
   out_4984937935515434202[0] = 0;
   out_4984937935515434202[1] = 0;
   out_4984937935515434202[2] = 0;
   out_4984937935515434202[3] = 0;
   out_4984937935515434202[4] = 1;
   out_4984937935515434202[5] = 0;
   out_4984937935515434202[6] = 0;
   out_4984937935515434202[7] = 0;
   out_4984937935515434202[8] = 0;
   out_4984937935515434202[9] = 0;
   out_4984937935515434202[10] = 0;
   out_4984937935515434202[11] = 0;
   out_4984937935515434202[12] = 0;
   out_4984937935515434202[13] = 0;
   out_4984937935515434202[14] = 1;
   out_4984937935515434202[15] = 0;
   out_4984937935515434202[16] = 0;
   out_4984937935515434202[17] = 0;
}
void h_30(double *state, double *unused, double *out_901660284197832910) {
   out_901660284197832910[0] = state[4];
}
void H_30(double *state, double *unused, double *out_4643819400615335548) {
   out_4643819400615335548[0] = 0;
   out_4643819400615335548[1] = 0;
   out_4643819400615335548[2] = 0;
   out_4643819400615335548[3] = 0;
   out_4643819400615335548[4] = 1;
   out_4643819400615335548[5] = 0;
   out_4643819400615335548[6] = 0;
   out_4643819400615335548[7] = 0;
   out_4643819400615335548[8] = 0;
}
void h_26(double *state, double *unused, double *out_4060719713483460322) {
   out_4060719713483460322[0] = state[7];
}
void H_26(double *state, double *unused, double *out_7543088395712911217) {
   out_7543088395712911217[0] = 0;
   out_7543088395712911217[1] = 0;
   out_7543088395712911217[2] = 0;
   out_7543088395712911217[3] = 0;
   out_7543088395712911217[4] = 0;
   out_7543088395712911217[5] = 0;
   out_7543088395712911217[6] = 0;
   out_7543088395712911217[7] = 1;
   out_7543088395712911217[8] = 0;
}
void h_27(double *state, double *unused, double *out_3557025719116557148) {
   out_3557025719116557148[0] = state[3];
}
void H_27(double *state, double *unused, double *out_2420225329431392331) {
   out_2420225329431392331[0] = 0;
   out_2420225329431392331[1] = 0;
   out_2420225329431392331[2] = 0;
   out_2420225329431392331[3] = 1;
   out_2420225329431392331[4] = 0;
   out_2420225329431392331[5] = 0;
   out_2420225329431392331[6] = 0;
   out_2420225329431392331[7] = 0;
   out_2420225329431392331[8] = 0;
}
void h_29(double *state, double *unused, double *out_4958990177267014116) {
   out_4958990177267014116[0] = state[1];
}
void H_29(double *state, double *unused, double *out_4133588056300943364) {
   out_4133588056300943364[0] = 0;
   out_4133588056300943364[1] = 1;
   out_4133588056300943364[2] = 0;
   out_4133588056300943364[3] = 0;
   out_4133588056300943364[4] = 0;
   out_4133588056300943364[5] = 0;
   out_4133588056300943364[6] = 0;
   out_4133588056300943364[7] = 0;
   out_4133588056300943364[8] = 0;
}
void h_28(double *state, double *unused, double *out_2953709834440079631) {
   out_2953709834440079631[0] = state[0];
}
void H_28(double *state, double *unused, double *out_9215987073370473938) {
   out_9215987073370473938[0] = 1;
   out_9215987073370473938[1] = 0;
   out_9215987073370473938[2] = 0;
   out_9215987073370473938[3] = 0;
   out_9215987073370473938[4] = 0;
   out_9215987073370473938[5] = 0;
   out_9215987073370473938[6] = 0;
   out_9215987073370473938[7] = 0;
   out_9215987073370473938[8] = 0;
}
void h_31(double *state, double *unused, double *out_6026361610503647171) {
   out_6026361610503647171[0] = state[8];
}
void H_31(double *state, double *unused, double *out_6916880293479559741) {
   out_6916880293479559741[0] = 0;
   out_6916880293479559741[1] = 0;
   out_6916880293479559741[2] = 0;
   out_6916880293479559741[3] = 0;
   out_6916880293479559741[4] = 0;
   out_6916880293479559741[5] = 0;
   out_6916880293479559741[6] = 0;
   out_6916880293479559741[7] = 0;
   out_6916880293479559741[8] = 1;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_25, H_25, NULL, in_z, in_R, in_ea, MAHA_THRESH_25);
}
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<2, 3, 0>(in_x, in_P, h_24, H_24, NULL, in_z, in_R, in_ea, MAHA_THRESH_24);
}
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_30, H_30, NULL, in_z, in_R, in_ea, MAHA_THRESH_30);
}
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_26, H_26, NULL, in_z, in_R, in_ea, MAHA_THRESH_26);
}
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_27, H_27, NULL, in_z, in_R, in_ea, MAHA_THRESH_27);
}
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_29, H_29, NULL, in_z, in_R, in_ea, MAHA_THRESH_29);
}
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_28, H_28, NULL, in_z, in_R, in_ea, MAHA_THRESH_28);
}
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_31, H_31, NULL, in_z, in_R, in_ea, MAHA_THRESH_31);
}
void car_err_fun(double *nom_x, double *delta_x, double *out_7584201253892190001) {
  err_fun(nom_x, delta_x, out_7584201253892190001);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_4881621785150572117) {
  inv_err_fun(nom_x, true_x, out_4881621785150572117);
}
void car_H_mod_fun(double *state, double *out_6380899889402302601) {
  H_mod_fun(state, out_6380899889402302601);
}
void car_f_fun(double *state, double dt, double *out_6392402267122999460) {
  f_fun(state,  dt, out_6392402267122999460);
}
void car_F_fun(double *state, double dt, double *out_2526673064601828008) {
  F_fun(state,  dt, out_2526673064601828008);
}
void car_h_25(double *state, double *unused, double *out_4624397152353190203) {
  h_25(state, unused, out_4624397152353190203);
}
void car_H_25(double *state, double *unused, double *out_7162152359122584175) {
  H_25(state, unused, out_7162152359122584175);
}
void car_h_24(double *state, double *unused, double *out_5737550318670326813) {
  h_24(state, unused, out_5737550318670326813);
}
void car_H_24(double *state, double *unused, double *out_4984937935515434202) {
  H_24(state, unused, out_4984937935515434202);
}
void car_h_30(double *state, double *unused, double *out_901660284197832910) {
  h_30(state, unused, out_901660284197832910);
}
void car_H_30(double *state, double *unused, double *out_4643819400615335548) {
  H_30(state, unused, out_4643819400615335548);
}
void car_h_26(double *state, double *unused, double *out_4060719713483460322) {
  h_26(state, unused, out_4060719713483460322);
}
void car_H_26(double *state, double *unused, double *out_7543088395712911217) {
  H_26(state, unused, out_7543088395712911217);
}
void car_h_27(double *state, double *unused, double *out_3557025719116557148) {
  h_27(state, unused, out_3557025719116557148);
}
void car_H_27(double *state, double *unused, double *out_2420225329431392331) {
  H_27(state, unused, out_2420225329431392331);
}
void car_h_29(double *state, double *unused, double *out_4958990177267014116) {
  h_29(state, unused, out_4958990177267014116);
}
void car_H_29(double *state, double *unused, double *out_4133588056300943364) {
  H_29(state, unused, out_4133588056300943364);
}
void car_h_28(double *state, double *unused, double *out_2953709834440079631) {
  h_28(state, unused, out_2953709834440079631);
}
void car_H_28(double *state, double *unused, double *out_9215987073370473938) {
  H_28(state, unused, out_9215987073370473938);
}
void car_h_31(double *state, double *unused, double *out_6026361610503647171) {
  h_31(state, unused, out_6026361610503647171);
}
void car_H_31(double *state, double *unused, double *out_6916880293479559741) {
  H_31(state, unused, out_6916880293479559741);
}
void car_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
void car_set_mass(double x) {
  set_mass(x);
}
void car_set_rotational_inertia(double x) {
  set_rotational_inertia(x);
}
void car_set_center_to_front(double x) {
  set_center_to_front(x);
}
void car_set_center_to_rear(double x) {
  set_center_to_rear(x);
}
void car_set_stiffness_front(double x) {
  set_stiffness_front(x);
}
void car_set_stiffness_rear(double x) {
  set_stiffness_rear(x);
}
}

const EKF car = {
  .name = "car",
  .kinds = { 25, 24, 30, 26, 27, 29, 28, 31 },
  .feature_kinds = {  },
  .f_fun = car_f_fun,
  .F_fun = car_F_fun,
  .err_fun = car_err_fun,
  .inv_err_fun = car_inv_err_fun,
  .H_mod_fun = car_H_mod_fun,
  .predict = car_predict,
  .hs = {
    { 25, car_h_25 },
    { 24, car_h_24 },
    { 30, car_h_30 },
    { 26, car_h_26 },
    { 27, car_h_27 },
    { 29, car_h_29 },
    { 28, car_h_28 },
    { 31, car_h_31 },
  },
  .Hs = {
    { 25, car_H_25 },
    { 24, car_H_24 },
    { 30, car_H_30 },
    { 26, car_H_26 },
    { 27, car_H_27 },
    { 29, car_H_29 },
    { 28, car_H_28 },
    { 31, car_H_31 },
  },
  .updates = {
    { 25, car_update_25 },
    { 24, car_update_24 },
    { 30, car_update_30 },
    { 26, car_update_26 },
    { 27, car_update_27 },
    { 29, car_update_29 },
    { 28, car_update_28 },
    { 31, car_update_31 },
  },
  .Hes = {
  },
  .sets = {
    { "mass", car_set_mass },
    { "rotational_inertia", car_set_rotational_inertia },
    { "center_to_front", car_set_center_to_front },
    { "center_to_rear", car_set_center_to_rear },
    { "stiffness_front", car_set_stiffness_front },
    { "stiffness_rear", car_set_stiffness_rear },
  },
  .extra_routines = {
  },
};

ekf_lib_init(car)
