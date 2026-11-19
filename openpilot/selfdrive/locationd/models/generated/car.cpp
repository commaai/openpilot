#include "car.h"

namespace {
#define DIM 9
#define EDIM 9
#define MEDIM 9
typedef void (*Hfun)(double *, double *, double *);

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
void err_fun(double *nom_x, double *delta_x, double *out_790466696376099935) {
   out_790466696376099935[0] = delta_x[0] + nom_x[0];
   out_790466696376099935[1] = delta_x[1] + nom_x[1];
   out_790466696376099935[2] = delta_x[2] + nom_x[2];
   out_790466696376099935[3] = delta_x[3] + nom_x[3];
   out_790466696376099935[4] = delta_x[4] + nom_x[4];
   out_790466696376099935[5] = delta_x[5] + nom_x[5];
   out_790466696376099935[6] = delta_x[6] + nom_x[6];
   out_790466696376099935[7] = delta_x[7] + nom_x[7];
   out_790466696376099935[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_529756539852428822) {
   out_529756539852428822[0] = -nom_x[0] + true_x[0];
   out_529756539852428822[1] = -nom_x[1] + true_x[1];
   out_529756539852428822[2] = -nom_x[2] + true_x[2];
   out_529756539852428822[3] = -nom_x[3] + true_x[3];
   out_529756539852428822[4] = -nom_x[4] + true_x[4];
   out_529756539852428822[5] = -nom_x[5] + true_x[5];
   out_529756539852428822[6] = -nom_x[6] + true_x[6];
   out_529756539852428822[7] = -nom_x[7] + true_x[7];
   out_529756539852428822[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_9219675536978344786) {
   out_9219675536978344786[0] = 1.0;
   out_9219675536978344786[1] = 0.0;
   out_9219675536978344786[2] = 0.0;
   out_9219675536978344786[3] = 0.0;
   out_9219675536978344786[4] = 0.0;
   out_9219675536978344786[5] = 0.0;
   out_9219675536978344786[6] = 0.0;
   out_9219675536978344786[7] = 0.0;
   out_9219675536978344786[8] = 0.0;
   out_9219675536978344786[9] = 0.0;
   out_9219675536978344786[10] = 1.0;
   out_9219675536978344786[11] = 0.0;
   out_9219675536978344786[12] = 0.0;
   out_9219675536978344786[13] = 0.0;
   out_9219675536978344786[14] = 0.0;
   out_9219675536978344786[15] = 0.0;
   out_9219675536978344786[16] = 0.0;
   out_9219675536978344786[17] = 0.0;
   out_9219675536978344786[18] = 0.0;
   out_9219675536978344786[19] = 0.0;
   out_9219675536978344786[20] = 1.0;
   out_9219675536978344786[21] = 0.0;
   out_9219675536978344786[22] = 0.0;
   out_9219675536978344786[23] = 0.0;
   out_9219675536978344786[24] = 0.0;
   out_9219675536978344786[25] = 0.0;
   out_9219675536978344786[26] = 0.0;
   out_9219675536978344786[27] = 0.0;
   out_9219675536978344786[28] = 0.0;
   out_9219675536978344786[29] = 0.0;
   out_9219675536978344786[30] = 1.0;
   out_9219675536978344786[31] = 0.0;
   out_9219675536978344786[32] = 0.0;
   out_9219675536978344786[33] = 0.0;
   out_9219675536978344786[34] = 0.0;
   out_9219675536978344786[35] = 0.0;
   out_9219675536978344786[36] = 0.0;
   out_9219675536978344786[37] = 0.0;
   out_9219675536978344786[38] = 0.0;
   out_9219675536978344786[39] = 0.0;
   out_9219675536978344786[40] = 1.0;
   out_9219675536978344786[41] = 0.0;
   out_9219675536978344786[42] = 0.0;
   out_9219675536978344786[43] = 0.0;
   out_9219675536978344786[44] = 0.0;
   out_9219675536978344786[45] = 0.0;
   out_9219675536978344786[46] = 0.0;
   out_9219675536978344786[47] = 0.0;
   out_9219675536978344786[48] = 0.0;
   out_9219675536978344786[49] = 0.0;
   out_9219675536978344786[50] = 1.0;
   out_9219675536978344786[51] = 0.0;
   out_9219675536978344786[52] = 0.0;
   out_9219675536978344786[53] = 0.0;
   out_9219675536978344786[54] = 0.0;
   out_9219675536978344786[55] = 0.0;
   out_9219675536978344786[56] = 0.0;
   out_9219675536978344786[57] = 0.0;
   out_9219675536978344786[58] = 0.0;
   out_9219675536978344786[59] = 0.0;
   out_9219675536978344786[60] = 1.0;
   out_9219675536978344786[61] = 0.0;
   out_9219675536978344786[62] = 0.0;
   out_9219675536978344786[63] = 0.0;
   out_9219675536978344786[64] = 0.0;
   out_9219675536978344786[65] = 0.0;
   out_9219675536978344786[66] = 0.0;
   out_9219675536978344786[67] = 0.0;
   out_9219675536978344786[68] = 0.0;
   out_9219675536978344786[69] = 0.0;
   out_9219675536978344786[70] = 1.0;
   out_9219675536978344786[71] = 0.0;
   out_9219675536978344786[72] = 0.0;
   out_9219675536978344786[73] = 0.0;
   out_9219675536978344786[74] = 0.0;
   out_9219675536978344786[75] = 0.0;
   out_9219675536978344786[76] = 0.0;
   out_9219675536978344786[77] = 0.0;
   out_9219675536978344786[78] = 0.0;
   out_9219675536978344786[79] = 0.0;
   out_9219675536978344786[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_6134589376329078738) {
   out_6134589376329078738[0] = state[0];
   out_6134589376329078738[1] = state[1];
   out_6134589376329078738[2] = state[2];
   out_6134589376329078738[3] = state[3];
   out_6134589376329078738[4] = state[4];
   out_6134589376329078738[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_6134589376329078738[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_6134589376329078738[7] = state[7];
   out_6134589376329078738[8] = state[8];
}
void F_fun(double *state, double dt, double *out_4746845148006465944) {
   out_4746845148006465944[0] = 1;
   out_4746845148006465944[1] = 0;
   out_4746845148006465944[2] = 0;
   out_4746845148006465944[3] = 0;
   out_4746845148006465944[4] = 0;
   out_4746845148006465944[5] = 0;
   out_4746845148006465944[6] = 0;
   out_4746845148006465944[7] = 0;
   out_4746845148006465944[8] = 0;
   out_4746845148006465944[9] = 0;
   out_4746845148006465944[10] = 1;
   out_4746845148006465944[11] = 0;
   out_4746845148006465944[12] = 0;
   out_4746845148006465944[13] = 0;
   out_4746845148006465944[14] = 0;
   out_4746845148006465944[15] = 0;
   out_4746845148006465944[16] = 0;
   out_4746845148006465944[17] = 0;
   out_4746845148006465944[18] = 0;
   out_4746845148006465944[19] = 0;
   out_4746845148006465944[20] = 1;
   out_4746845148006465944[21] = 0;
   out_4746845148006465944[22] = 0;
   out_4746845148006465944[23] = 0;
   out_4746845148006465944[24] = 0;
   out_4746845148006465944[25] = 0;
   out_4746845148006465944[26] = 0;
   out_4746845148006465944[27] = 0;
   out_4746845148006465944[28] = 0;
   out_4746845148006465944[29] = 0;
   out_4746845148006465944[30] = 1;
   out_4746845148006465944[31] = 0;
   out_4746845148006465944[32] = 0;
   out_4746845148006465944[33] = 0;
   out_4746845148006465944[34] = 0;
   out_4746845148006465944[35] = 0;
   out_4746845148006465944[36] = 0;
   out_4746845148006465944[37] = 0;
   out_4746845148006465944[38] = 0;
   out_4746845148006465944[39] = 0;
   out_4746845148006465944[40] = 1;
   out_4746845148006465944[41] = 0;
   out_4746845148006465944[42] = 0;
   out_4746845148006465944[43] = 0;
   out_4746845148006465944[44] = 0;
   out_4746845148006465944[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_4746845148006465944[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_4746845148006465944[47] = -dt*stiffness_front*state[0]/state[1];
   out_4746845148006465944[48] = -dt*stiffness_front*state[0]/state[1];
   out_4746845148006465944[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_4746845148006465944[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_4746845148006465944[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_4746845148006465944[52] = dt*stiffness_front*state[0]/state[1];
   out_4746845148006465944[53] = -9.8100000000000005*dt;
   out_4746845148006465944[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_4746845148006465944[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_4746845148006465944[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4746845148006465944[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4746845148006465944[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_4746845148006465944[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_4746845148006465944[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_4746845148006465944[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4746845148006465944[62] = 0;
   out_4746845148006465944[63] = 0;
   out_4746845148006465944[64] = 0;
   out_4746845148006465944[65] = 0;
   out_4746845148006465944[66] = 0;
   out_4746845148006465944[67] = 0;
   out_4746845148006465944[68] = 0;
   out_4746845148006465944[69] = 0;
   out_4746845148006465944[70] = 1;
   out_4746845148006465944[71] = 0;
   out_4746845148006465944[72] = 0;
   out_4746845148006465944[73] = 0;
   out_4746845148006465944[74] = 0;
   out_4746845148006465944[75] = 0;
   out_4746845148006465944[76] = 0;
   out_4746845148006465944[77] = 0;
   out_4746845148006465944[78] = 0;
   out_4746845148006465944[79] = 0;
   out_4746845148006465944[80] = 1;
}
void h_25(double *state, double *unused, double *out_8027414408540613172) {
   out_8027414408540613172[0] = state[6];
}
void H_25(double *state, double *unused, double *out_33880310818969593) {
   out_33880310818969593[0] = 0;
   out_33880310818969593[1] = 0;
   out_33880310818969593[2] = 0;
   out_33880310818969593[3] = 0;
   out_33880310818969593[4] = 0;
   out_33880310818969593[5] = 0;
   out_33880310818969593[6] = 1;
   out_33880310818969593[7] = 0;
   out_33880310818969593[8] = 0;
}
void h_24(double *state, double *unused, double *out_113766384020535879) {
   out_113766384020535879[0] = state[4];
   out_113766384020535879[1] = state[5];
}
void H_24(double *state, double *unused, double *out_8741012096993201301) {
   out_8741012096993201301[0] = 0;
   out_8741012096993201301[1] = 0;
   out_8741012096993201301[2] = 0;
   out_8741012096993201301[3] = 0;
   out_8741012096993201301[4] = 1;
   out_8741012096993201301[5] = 0;
   out_8741012096993201301[6] = 0;
   out_8741012096993201301[7] = 0;
   out_8741012096993201301[8] = 0;
   out_8741012096993201301[9] = 0;
   out_8741012096993201301[10] = 0;
   out_8741012096993201301[11] = 0;
   out_8741012096993201301[12] = 0;
   out_8741012096993201301[13] = 0;
   out_8741012096993201301[14] = 1;
   out_8741012096993201301[15] = 0;
   out_8741012096993201301[16] = 0;
   out_8741012096993201301[17] = 0;
}
void h_30(double *state, double *unused, double *out_8968698272540055277) {
   out_8968698272540055277[0] = state[4];
}
void H_30(double *state, double *unused, double *out_95458636324270477) {
   out_95458636324270477[0] = 0;
   out_95458636324270477[1] = 0;
   out_95458636324270477[2] = 0;
   out_95458636324270477[3] = 0;
   out_95458636324270477[4] = 1;
   out_95458636324270477[5] = 0;
   out_95458636324270477[6] = 0;
   out_95458636324270477[7] = 0;
   out_95458636324270477[8] = 0;
}
void h_26(double *state, double *unused, double *out_6441259749725163654) {
   out_6441259749725163654[0] = state[7];
}
void H_26(double *state, double *unused, double *out_3707623008055086631) {
   out_3707623008055086631[0] = 0;
   out_3707623008055086631[1] = 0;
   out_3707623008055086631[2] = 0;
   out_3707623008055086631[3] = 0;
   out_3707623008055086631[4] = 0;
   out_3707623008055086631[5] = 0;
   out_3707623008055086631[6] = 0;
   out_3707623008055086631[7] = 1;
   out_3707623008055086631[8] = 0;
}
void h_27(double *state, double *unused, double *out_3416566961375872665) {
   out_3416566961375872665[0] = state[3];
}
void H_27(double *state, double *unused, double *out_2270221948124695388) {
   out_2270221948124695388[0] = 0;
   out_2270221948124695388[1] = 0;
   out_2270221948124695388[2] = 0;
   out_2270221948124695388[3] = 1;
   out_2270221948124695388[4] = 0;
   out_2270221948124695388[5] = 0;
   out_2270221948124695388[6] = 0;
   out_2270221948124695388[7] = 0;
   out_2270221948124695388[8] = 0;
}
void h_29(double *state, double *unused, double *out_4476372533508650694) {
   out_4476372533508650694[0] = state[1];
}
void H_29(double *state, double *unused, double *out_414772707990121707) {
   out_414772707990121707[0] = 0;
   out_414772707990121707[1] = 1;
   out_414772707990121707[2] = 0;
   out_414772707990121707[3] = 0;
   out_414772707990121707[4] = 0;
   out_414772707990121707[5] = 0;
   out_414772707990121707[6] = 0;
   out_414772707990121707[7] = 0;
   out_414772707990121707[8] = 0;
}
void h_28(double *state, double *unused, double *out_5565402210630968100) {
   out_5565402210630968100[0] = state[0];
}
void H_28(double *state, double *unused, double *out_2019954403428920170) {
   out_2019954403428920170[0] = 1;
   out_2019954403428920170[1] = 0;
   out_2019954403428920170[2] = 0;
   out_2019954403428920170[3] = 0;
   out_2019954403428920170[4] = 0;
   out_2019954403428920170[5] = 0;
   out_2019954403428920170[6] = 0;
   out_2019954403428920170[7] = 0;
   out_2019954403428920170[8] = 0;
}
void h_31(double *state, double *unused, double *out_1860300368569478869) {
   out_1860300368569478869[0] = state[8];
}
void H_31(double *state, double *unused, double *out_64526272695930021) {
   out_64526272695930021[0] = 0;
   out_64526272695930021[1] = 0;
   out_64526272695930021[2] = 0;
   out_64526272695930021[3] = 0;
   out_64526272695930021[4] = 0;
   out_64526272695930021[5] = 0;
   out_64526272695930021[6] = 0;
   out_64526272695930021[7] = 0;
   out_64526272695930021[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_790466696376099935) {
  err_fun(nom_x, delta_x, out_790466696376099935);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_529756539852428822) {
  inv_err_fun(nom_x, true_x, out_529756539852428822);
}
void car_H_mod_fun(double *state, double *out_9219675536978344786) {
  H_mod_fun(state, out_9219675536978344786);
}
void car_f_fun(double *state, double dt, double *out_6134589376329078738) {
  f_fun(state,  dt, out_6134589376329078738);
}
void car_F_fun(double *state, double dt, double *out_4746845148006465944) {
  F_fun(state,  dt, out_4746845148006465944);
}
void car_h_25(double *state, double *unused, double *out_8027414408540613172) {
  h_25(state, unused, out_8027414408540613172);
}
void car_H_25(double *state, double *unused, double *out_33880310818969593) {
  H_25(state, unused, out_33880310818969593);
}
void car_h_24(double *state, double *unused, double *out_113766384020535879) {
  h_24(state, unused, out_113766384020535879);
}
void car_H_24(double *state, double *unused, double *out_8741012096993201301) {
  H_24(state, unused, out_8741012096993201301);
}
void car_h_30(double *state, double *unused, double *out_8968698272540055277) {
  h_30(state, unused, out_8968698272540055277);
}
void car_H_30(double *state, double *unused, double *out_95458636324270477) {
  H_30(state, unused, out_95458636324270477);
}
void car_h_26(double *state, double *unused, double *out_6441259749725163654) {
  h_26(state, unused, out_6441259749725163654);
}
void car_H_26(double *state, double *unused, double *out_3707623008055086631) {
  H_26(state, unused, out_3707623008055086631);
}
void car_h_27(double *state, double *unused, double *out_3416566961375872665) {
  h_27(state, unused, out_3416566961375872665);
}
void car_H_27(double *state, double *unused, double *out_2270221948124695388) {
  H_27(state, unused, out_2270221948124695388);
}
void car_h_29(double *state, double *unused, double *out_4476372533508650694) {
  h_29(state, unused, out_4476372533508650694);
}
void car_H_29(double *state, double *unused, double *out_414772707990121707) {
  H_29(state, unused, out_414772707990121707);
}
void car_h_28(double *state, double *unused, double *out_5565402210630968100) {
  h_28(state, unused, out_5565402210630968100);
}
void car_H_28(double *state, double *unused, double *out_2019954403428920170) {
  H_28(state, unused, out_2019954403428920170);
}
void car_h_31(double *state, double *unused, double *out_1860300368569478869) {
  h_31(state, unused, out_1860300368569478869);
}
void car_H_31(double *state, double *unused, double *out_64526272695930021) {
  H_31(state, unused, out_64526272695930021);
}
void car_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
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
