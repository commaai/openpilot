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
void err_fun(double *nom_x, double *delta_x, double *out_6912122597548111650) {
   out_6912122597548111650[0] = delta_x[0] + nom_x[0];
   out_6912122597548111650[1] = delta_x[1] + nom_x[1];
   out_6912122597548111650[2] = delta_x[2] + nom_x[2];
   out_6912122597548111650[3] = delta_x[3] + nom_x[3];
   out_6912122597548111650[4] = delta_x[4] + nom_x[4];
   out_6912122597548111650[5] = delta_x[5] + nom_x[5];
   out_6912122597548111650[6] = delta_x[6] + nom_x[6];
   out_6912122597548111650[7] = delta_x[7] + nom_x[7];
   out_6912122597548111650[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_5839740732604401630) {
   out_5839740732604401630[0] = -nom_x[0] + true_x[0];
   out_5839740732604401630[1] = -nom_x[1] + true_x[1];
   out_5839740732604401630[2] = -nom_x[2] + true_x[2];
   out_5839740732604401630[3] = -nom_x[3] + true_x[3];
   out_5839740732604401630[4] = -nom_x[4] + true_x[4];
   out_5839740732604401630[5] = -nom_x[5] + true_x[5];
   out_5839740732604401630[6] = -nom_x[6] + true_x[6];
   out_5839740732604401630[7] = -nom_x[7] + true_x[7];
   out_5839740732604401630[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_5459200475037534350) {
   out_5459200475037534350[0] = 1.0;
   out_5459200475037534350[1] = 0.0;
   out_5459200475037534350[2] = 0.0;
   out_5459200475037534350[3] = 0.0;
   out_5459200475037534350[4] = 0.0;
   out_5459200475037534350[5] = 0.0;
   out_5459200475037534350[6] = 0.0;
   out_5459200475037534350[7] = 0.0;
   out_5459200475037534350[8] = 0.0;
   out_5459200475037534350[9] = 0.0;
   out_5459200475037534350[10] = 1.0;
   out_5459200475037534350[11] = 0.0;
   out_5459200475037534350[12] = 0.0;
   out_5459200475037534350[13] = 0.0;
   out_5459200475037534350[14] = 0.0;
   out_5459200475037534350[15] = 0.0;
   out_5459200475037534350[16] = 0.0;
   out_5459200475037534350[17] = 0.0;
   out_5459200475037534350[18] = 0.0;
   out_5459200475037534350[19] = 0.0;
   out_5459200475037534350[20] = 1.0;
   out_5459200475037534350[21] = 0.0;
   out_5459200475037534350[22] = 0.0;
   out_5459200475037534350[23] = 0.0;
   out_5459200475037534350[24] = 0.0;
   out_5459200475037534350[25] = 0.0;
   out_5459200475037534350[26] = 0.0;
   out_5459200475037534350[27] = 0.0;
   out_5459200475037534350[28] = 0.0;
   out_5459200475037534350[29] = 0.0;
   out_5459200475037534350[30] = 1.0;
   out_5459200475037534350[31] = 0.0;
   out_5459200475037534350[32] = 0.0;
   out_5459200475037534350[33] = 0.0;
   out_5459200475037534350[34] = 0.0;
   out_5459200475037534350[35] = 0.0;
   out_5459200475037534350[36] = 0.0;
   out_5459200475037534350[37] = 0.0;
   out_5459200475037534350[38] = 0.0;
   out_5459200475037534350[39] = 0.0;
   out_5459200475037534350[40] = 1.0;
   out_5459200475037534350[41] = 0.0;
   out_5459200475037534350[42] = 0.0;
   out_5459200475037534350[43] = 0.0;
   out_5459200475037534350[44] = 0.0;
   out_5459200475037534350[45] = 0.0;
   out_5459200475037534350[46] = 0.0;
   out_5459200475037534350[47] = 0.0;
   out_5459200475037534350[48] = 0.0;
   out_5459200475037534350[49] = 0.0;
   out_5459200475037534350[50] = 1.0;
   out_5459200475037534350[51] = 0.0;
   out_5459200475037534350[52] = 0.0;
   out_5459200475037534350[53] = 0.0;
   out_5459200475037534350[54] = 0.0;
   out_5459200475037534350[55] = 0.0;
   out_5459200475037534350[56] = 0.0;
   out_5459200475037534350[57] = 0.0;
   out_5459200475037534350[58] = 0.0;
   out_5459200475037534350[59] = 0.0;
   out_5459200475037534350[60] = 1.0;
   out_5459200475037534350[61] = 0.0;
   out_5459200475037534350[62] = 0.0;
   out_5459200475037534350[63] = 0.0;
   out_5459200475037534350[64] = 0.0;
   out_5459200475037534350[65] = 0.0;
   out_5459200475037534350[66] = 0.0;
   out_5459200475037534350[67] = 0.0;
   out_5459200475037534350[68] = 0.0;
   out_5459200475037534350[69] = 0.0;
   out_5459200475037534350[70] = 1.0;
   out_5459200475037534350[71] = 0.0;
   out_5459200475037534350[72] = 0.0;
   out_5459200475037534350[73] = 0.0;
   out_5459200475037534350[74] = 0.0;
   out_5459200475037534350[75] = 0.0;
   out_5459200475037534350[76] = 0.0;
   out_5459200475037534350[77] = 0.0;
   out_5459200475037534350[78] = 0.0;
   out_5459200475037534350[79] = 0.0;
   out_5459200475037534350[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_7919248626531568362) {
   out_7919248626531568362[0] = state[0];
   out_7919248626531568362[1] = state[1];
   out_7919248626531568362[2] = state[2];
   out_7919248626531568362[3] = state[3];
   out_7919248626531568362[4] = state[4];
   out_7919248626531568362[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_7919248626531568362[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_7919248626531568362[7] = state[7];
   out_7919248626531568362[8] = state[8];
}
void F_fun(double *state, double dt, double *out_8728212283248373813) {
   out_8728212283248373813[0] = 1;
   out_8728212283248373813[1] = 0;
   out_8728212283248373813[2] = 0;
   out_8728212283248373813[3] = 0;
   out_8728212283248373813[4] = 0;
   out_8728212283248373813[5] = 0;
   out_8728212283248373813[6] = 0;
   out_8728212283248373813[7] = 0;
   out_8728212283248373813[8] = 0;
   out_8728212283248373813[9] = 0;
   out_8728212283248373813[10] = 1;
   out_8728212283248373813[11] = 0;
   out_8728212283248373813[12] = 0;
   out_8728212283248373813[13] = 0;
   out_8728212283248373813[14] = 0;
   out_8728212283248373813[15] = 0;
   out_8728212283248373813[16] = 0;
   out_8728212283248373813[17] = 0;
   out_8728212283248373813[18] = 0;
   out_8728212283248373813[19] = 0;
   out_8728212283248373813[20] = 1;
   out_8728212283248373813[21] = 0;
   out_8728212283248373813[22] = 0;
   out_8728212283248373813[23] = 0;
   out_8728212283248373813[24] = 0;
   out_8728212283248373813[25] = 0;
   out_8728212283248373813[26] = 0;
   out_8728212283248373813[27] = 0;
   out_8728212283248373813[28] = 0;
   out_8728212283248373813[29] = 0;
   out_8728212283248373813[30] = 1;
   out_8728212283248373813[31] = 0;
   out_8728212283248373813[32] = 0;
   out_8728212283248373813[33] = 0;
   out_8728212283248373813[34] = 0;
   out_8728212283248373813[35] = 0;
   out_8728212283248373813[36] = 0;
   out_8728212283248373813[37] = 0;
   out_8728212283248373813[38] = 0;
   out_8728212283248373813[39] = 0;
   out_8728212283248373813[40] = 1;
   out_8728212283248373813[41] = 0;
   out_8728212283248373813[42] = 0;
   out_8728212283248373813[43] = 0;
   out_8728212283248373813[44] = 0;
   out_8728212283248373813[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_8728212283248373813[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_8728212283248373813[47] = -dt*stiffness_front*state[0]/state[1];
   out_8728212283248373813[48] = -dt*stiffness_front*state[0]/state[1];
   out_8728212283248373813[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_8728212283248373813[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_8728212283248373813[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_8728212283248373813[52] = dt*stiffness_front*state[0]/state[1];
   out_8728212283248373813[53] = -9.8100000000000005*dt;
   out_8728212283248373813[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_8728212283248373813[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_8728212283248373813[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8728212283248373813[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8728212283248373813[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_8728212283248373813[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_8728212283248373813[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_8728212283248373813[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8728212283248373813[62] = 0;
   out_8728212283248373813[63] = 0;
   out_8728212283248373813[64] = 0;
   out_8728212283248373813[65] = 0;
   out_8728212283248373813[66] = 0;
   out_8728212283248373813[67] = 0;
   out_8728212283248373813[68] = 0;
   out_8728212283248373813[69] = 0;
   out_8728212283248373813[70] = 1;
   out_8728212283248373813[71] = 0;
   out_8728212283248373813[72] = 0;
   out_8728212283248373813[73] = 0;
   out_8728212283248373813[74] = 0;
   out_8728212283248373813[75] = 0;
   out_8728212283248373813[76] = 0;
   out_8728212283248373813[77] = 0;
   out_8728212283248373813[78] = 0;
   out_8728212283248373813[79] = 0;
   out_8728212283248373813[80] = 1;
}
void h_25(double *state, double *unused, double *out_7012538561194516599) {
   out_7012538561194516599[0] = state[6];
}
void H_25(double *state, double *unused, double *out_4510079553009052141) {
   out_4510079553009052141[0] = 0;
   out_4510079553009052141[1] = 0;
   out_4510079553009052141[2] = 0;
   out_4510079553009052141[3] = 0;
   out_4510079553009052141[4] = 0;
   out_4510079553009052141[5] = 0;
   out_4510079553009052141[6] = 1;
   out_4510079553009052141[7] = 0;
   out_4510079553009052141[8] = 0;
}
void h_24(double *state, double *unused, double *out_3347341305780913336) {
   out_3347341305780913336[0] = state[4];
   out_3347341305780913336[1] = state[5];
}
void H_24(double *state, double *unused, double *out_523191577009633836) {
   out_523191577009633836[0] = 0;
   out_523191577009633836[1] = 0;
   out_523191577009633836[2] = 0;
   out_523191577009633836[3] = 0;
   out_523191577009633836[4] = 1;
   out_523191577009633836[5] = 0;
   out_523191577009633836[6] = 0;
   out_523191577009633836[7] = 0;
   out_523191577009633836[8] = 0;
   out_523191577009633836[9] = 0;
   out_523191577009633836[10] = 0;
   out_523191577009633836[11] = 0;
   out_523191577009633836[12] = 0;
   out_523191577009633836[13] = 0;
   out_523191577009633836[14] = 1;
   out_523191577009633836[15] = 0;
   out_523191577009633836[16] = 0;
   out_523191577009633836[17] = 0;
}
void h_30(double *state, double *unused, double *out_4663370286164684081) {
   out_4663370286164684081[0] = state[4];
}
void H_30(double *state, double *unused, double *out_9037775883136660339) {
   out_9037775883136660339[0] = 0;
   out_9037775883136660339[1] = 0;
   out_9037775883136660339[2] = 0;
   out_9037775883136660339[3] = 0;
   out_9037775883136660339[4] = 1;
   out_9037775883136660339[5] = 0;
   out_9037775883136660339[6] = 0;
   out_9037775883136660339[7] = 0;
   out_9037775883136660339[8] = 0;
}
void h_26(double *state, double *unused, double *out_6353818120900365041) {
   out_6353818120900365041[0] = state[7];
}
void H_26(double *state, double *unused, double *out_8251582871883108365) {
   out_8251582871883108365[0] = 0;
   out_8251582871883108365[1] = 0;
   out_8251582871883108365[2] = 0;
   out_8251582871883108365[3] = 0;
   out_8251582871883108365[4] = 0;
   out_8251582871883108365[5] = 0;
   out_8251582871883108365[6] = 0;
   out_8251582871883108365[7] = 1;
   out_8251582871883108365[8] = 0;
}
void h_27(double *state, double *unused, double *out_2010865293533670954) {
   out_2010865293533670954[0] = state[3];
}
void H_27(double *state, double *unused, double *out_7234204878772466366) {
   out_7234204878772466366[0] = 0;
   out_7234204878772466366[1] = 0;
   out_7234204878772466366[2] = 0;
   out_7234204878772466366[3] = 1;
   out_7234204878772466366[4] = 0;
   out_7234204878772466366[5] = 0;
   out_7234204878772466366[6] = 0;
   out_7234204878772466366[7] = 0;
   out_7234204878772466366[8] = 0;
}
void h_29(double *state, double *unused, double *out_3303742755155314253) {
   out_3303742755155314253[0] = state[1];
}
void H_29(double *state, double *unused, double *out_8527544538822268155) {
   out_8527544538822268155[0] = 0;
   out_8527544538822268155[1] = 1;
   out_8527544538822268155[2] = 0;
   out_8527544538822268155[3] = 0;
   out_8527544538822268155[4] = 0;
   out_8527544538822268155[5] = 0;
   out_8527544538822268155[6] = 0;
   out_8527544538822268155[7] = 0;
   out_8527544538822268155[8] = 0;
}
void h_28(double *state, double *unused, double *out_2954375146435333389) {
   out_2954375146435333389[0] = state[0];
}
void H_28(double *state, double *unused, double *out_4836800517817752887) {
   out_4836800517817752887[0] = 1;
   out_4836800517817752887[1] = 0;
   out_4836800517817752887[2] = 0;
   out_4836800517817752887[3] = 0;
   out_4836800517817752887[4] = 0;
   out_4836800517817752887[5] = 0;
   out_4836800517817752887[6] = 0;
   out_4836800517817752887[7] = 0;
   out_4836800517817752887[8] = 0;
}
void h_31(double *state, double *unused, double *out_5561907168565633432) {
   out_5561907168565633432[0] = state[8];
}
void H_31(double *state, double *unused, double *out_8877790974116459841) {
   out_8877790974116459841[0] = 0;
   out_8877790974116459841[1] = 0;
   out_8877790974116459841[2] = 0;
   out_8877790974116459841[3] = 0;
   out_8877790974116459841[4] = 0;
   out_8877790974116459841[5] = 0;
   out_8877790974116459841[6] = 0;
   out_8877790974116459841[7] = 0;
   out_8877790974116459841[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_6912122597548111650) {
  err_fun(nom_x, delta_x, out_6912122597548111650);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5839740732604401630) {
  inv_err_fun(nom_x, true_x, out_5839740732604401630);
}
void car_H_mod_fun(double *state, double *out_5459200475037534350) {
  H_mod_fun(state, out_5459200475037534350);
}
void car_f_fun(double *state, double dt, double *out_7919248626531568362) {
  f_fun(state,  dt, out_7919248626531568362);
}
void car_F_fun(double *state, double dt, double *out_8728212283248373813) {
  F_fun(state,  dt, out_8728212283248373813);
}
void car_h_25(double *state, double *unused, double *out_7012538561194516599) {
  h_25(state, unused, out_7012538561194516599);
}
void car_H_25(double *state, double *unused, double *out_4510079553009052141) {
  H_25(state, unused, out_4510079553009052141);
}
void car_h_24(double *state, double *unused, double *out_3347341305780913336) {
  h_24(state, unused, out_3347341305780913336);
}
void car_H_24(double *state, double *unused, double *out_523191577009633836) {
  H_24(state, unused, out_523191577009633836);
}
void car_h_30(double *state, double *unused, double *out_4663370286164684081) {
  h_30(state, unused, out_4663370286164684081);
}
void car_H_30(double *state, double *unused, double *out_9037775883136660339) {
  H_30(state, unused, out_9037775883136660339);
}
void car_h_26(double *state, double *unused, double *out_6353818120900365041) {
  h_26(state, unused, out_6353818120900365041);
}
void car_H_26(double *state, double *unused, double *out_8251582871883108365) {
  H_26(state, unused, out_8251582871883108365);
}
void car_h_27(double *state, double *unused, double *out_2010865293533670954) {
  h_27(state, unused, out_2010865293533670954);
}
void car_H_27(double *state, double *unused, double *out_7234204878772466366) {
  H_27(state, unused, out_7234204878772466366);
}
void car_h_29(double *state, double *unused, double *out_3303742755155314253) {
  h_29(state, unused, out_3303742755155314253);
}
void car_H_29(double *state, double *unused, double *out_8527544538822268155) {
  H_29(state, unused, out_8527544538822268155);
}
void car_h_28(double *state, double *unused, double *out_2954375146435333389) {
  h_28(state, unused, out_2954375146435333389);
}
void car_H_28(double *state, double *unused, double *out_4836800517817752887) {
  H_28(state, unused, out_4836800517817752887);
}
void car_h_31(double *state, double *unused, double *out_5561907168565633432) {
  h_31(state, unused, out_5561907168565633432);
}
void car_H_31(double *state, double *unused, double *out_8877790974116459841) {
  H_31(state, unused, out_8877790974116459841);
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
