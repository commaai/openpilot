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
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_1345357640820912998) {
   out_1345357640820912998[0] = delta_x[0] + nom_x[0];
   out_1345357640820912998[1] = delta_x[1] + nom_x[1];
   out_1345357640820912998[2] = delta_x[2] + nom_x[2];
   out_1345357640820912998[3] = delta_x[3] + nom_x[3];
   out_1345357640820912998[4] = delta_x[4] + nom_x[4];
   out_1345357640820912998[5] = delta_x[5] + nom_x[5];
   out_1345357640820912998[6] = delta_x[6] + nom_x[6];
   out_1345357640820912998[7] = delta_x[7] + nom_x[7];
   out_1345357640820912998[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_565955163399411295) {
   out_565955163399411295[0] = -nom_x[0] + true_x[0];
   out_565955163399411295[1] = -nom_x[1] + true_x[1];
   out_565955163399411295[2] = -nom_x[2] + true_x[2];
   out_565955163399411295[3] = -nom_x[3] + true_x[3];
   out_565955163399411295[4] = -nom_x[4] + true_x[4];
   out_565955163399411295[5] = -nom_x[5] + true_x[5];
   out_565955163399411295[6] = -nom_x[6] + true_x[6];
   out_565955163399411295[7] = -nom_x[7] + true_x[7];
   out_565955163399411295[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_84820155383353969) {
   out_84820155383353969[0] = 1.0;
   out_84820155383353969[1] = 0;
   out_84820155383353969[2] = 0;
   out_84820155383353969[3] = 0;
   out_84820155383353969[4] = 0;
   out_84820155383353969[5] = 0;
   out_84820155383353969[6] = 0;
   out_84820155383353969[7] = 0;
   out_84820155383353969[8] = 0;
   out_84820155383353969[9] = 0;
   out_84820155383353969[10] = 1.0;
   out_84820155383353969[11] = 0;
   out_84820155383353969[12] = 0;
   out_84820155383353969[13] = 0;
   out_84820155383353969[14] = 0;
   out_84820155383353969[15] = 0;
   out_84820155383353969[16] = 0;
   out_84820155383353969[17] = 0;
   out_84820155383353969[18] = 0;
   out_84820155383353969[19] = 0;
   out_84820155383353969[20] = 1.0;
   out_84820155383353969[21] = 0;
   out_84820155383353969[22] = 0;
   out_84820155383353969[23] = 0;
   out_84820155383353969[24] = 0;
   out_84820155383353969[25] = 0;
   out_84820155383353969[26] = 0;
   out_84820155383353969[27] = 0;
   out_84820155383353969[28] = 0;
   out_84820155383353969[29] = 0;
   out_84820155383353969[30] = 1.0;
   out_84820155383353969[31] = 0;
   out_84820155383353969[32] = 0;
   out_84820155383353969[33] = 0;
   out_84820155383353969[34] = 0;
   out_84820155383353969[35] = 0;
   out_84820155383353969[36] = 0;
   out_84820155383353969[37] = 0;
   out_84820155383353969[38] = 0;
   out_84820155383353969[39] = 0;
   out_84820155383353969[40] = 1.0;
   out_84820155383353969[41] = 0;
   out_84820155383353969[42] = 0;
   out_84820155383353969[43] = 0;
   out_84820155383353969[44] = 0;
   out_84820155383353969[45] = 0;
   out_84820155383353969[46] = 0;
   out_84820155383353969[47] = 0;
   out_84820155383353969[48] = 0;
   out_84820155383353969[49] = 0;
   out_84820155383353969[50] = 1.0;
   out_84820155383353969[51] = 0;
   out_84820155383353969[52] = 0;
   out_84820155383353969[53] = 0;
   out_84820155383353969[54] = 0;
   out_84820155383353969[55] = 0;
   out_84820155383353969[56] = 0;
   out_84820155383353969[57] = 0;
   out_84820155383353969[58] = 0;
   out_84820155383353969[59] = 0;
   out_84820155383353969[60] = 1.0;
   out_84820155383353969[61] = 0;
   out_84820155383353969[62] = 0;
   out_84820155383353969[63] = 0;
   out_84820155383353969[64] = 0;
   out_84820155383353969[65] = 0;
   out_84820155383353969[66] = 0;
   out_84820155383353969[67] = 0;
   out_84820155383353969[68] = 0;
   out_84820155383353969[69] = 0;
   out_84820155383353969[70] = 1.0;
   out_84820155383353969[71] = 0;
   out_84820155383353969[72] = 0;
   out_84820155383353969[73] = 0;
   out_84820155383353969[74] = 0;
   out_84820155383353969[75] = 0;
   out_84820155383353969[76] = 0;
   out_84820155383353969[77] = 0;
   out_84820155383353969[78] = 0;
   out_84820155383353969[79] = 0;
   out_84820155383353969[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_1203605472893796358) {
   out_1203605472893796358[0] = state[0];
   out_1203605472893796358[1] = state[1];
   out_1203605472893796358[2] = state[2];
   out_1203605472893796358[3] = state[3];
   out_1203605472893796358[4] = state[4];
   out_1203605472893796358[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_1203605472893796358[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_1203605472893796358[7] = state[7];
   out_1203605472893796358[8] = state[8];
}
void F_fun(double *state, double dt, double *out_7074967459219019176) {
   out_7074967459219019176[0] = 1;
   out_7074967459219019176[1] = 0;
   out_7074967459219019176[2] = 0;
   out_7074967459219019176[3] = 0;
   out_7074967459219019176[4] = 0;
   out_7074967459219019176[5] = 0;
   out_7074967459219019176[6] = 0;
   out_7074967459219019176[7] = 0;
   out_7074967459219019176[8] = 0;
   out_7074967459219019176[9] = 0;
   out_7074967459219019176[10] = 1;
   out_7074967459219019176[11] = 0;
   out_7074967459219019176[12] = 0;
   out_7074967459219019176[13] = 0;
   out_7074967459219019176[14] = 0;
   out_7074967459219019176[15] = 0;
   out_7074967459219019176[16] = 0;
   out_7074967459219019176[17] = 0;
   out_7074967459219019176[18] = 0;
   out_7074967459219019176[19] = 0;
   out_7074967459219019176[20] = 1;
   out_7074967459219019176[21] = 0;
   out_7074967459219019176[22] = 0;
   out_7074967459219019176[23] = 0;
   out_7074967459219019176[24] = 0;
   out_7074967459219019176[25] = 0;
   out_7074967459219019176[26] = 0;
   out_7074967459219019176[27] = 0;
   out_7074967459219019176[28] = 0;
   out_7074967459219019176[29] = 0;
   out_7074967459219019176[30] = 1;
   out_7074967459219019176[31] = 0;
   out_7074967459219019176[32] = 0;
   out_7074967459219019176[33] = 0;
   out_7074967459219019176[34] = 0;
   out_7074967459219019176[35] = 0;
   out_7074967459219019176[36] = 0;
   out_7074967459219019176[37] = 0;
   out_7074967459219019176[38] = 0;
   out_7074967459219019176[39] = 0;
   out_7074967459219019176[40] = 1;
   out_7074967459219019176[41] = 0;
   out_7074967459219019176[42] = 0;
   out_7074967459219019176[43] = 0;
   out_7074967459219019176[44] = 0;
   out_7074967459219019176[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_7074967459219019176[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_7074967459219019176[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_7074967459219019176[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_7074967459219019176[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_7074967459219019176[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_7074967459219019176[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_7074967459219019176[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_7074967459219019176[53] = -9.8000000000000007*dt;
   out_7074967459219019176[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_7074967459219019176[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_7074967459219019176[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7074967459219019176[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7074967459219019176[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_7074967459219019176[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_7074967459219019176[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_7074967459219019176[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7074967459219019176[62] = 0;
   out_7074967459219019176[63] = 0;
   out_7074967459219019176[64] = 0;
   out_7074967459219019176[65] = 0;
   out_7074967459219019176[66] = 0;
   out_7074967459219019176[67] = 0;
   out_7074967459219019176[68] = 0;
   out_7074967459219019176[69] = 0;
   out_7074967459219019176[70] = 1;
   out_7074967459219019176[71] = 0;
   out_7074967459219019176[72] = 0;
   out_7074967459219019176[73] = 0;
   out_7074967459219019176[74] = 0;
   out_7074967459219019176[75] = 0;
   out_7074967459219019176[76] = 0;
   out_7074967459219019176[77] = 0;
   out_7074967459219019176[78] = 0;
   out_7074967459219019176[79] = 0;
   out_7074967459219019176[80] = 1;
}
void h_25(double *state, double *unused, double *out_2135589314781233871) {
   out_2135589314781233871[0] = state[6];
}
void H_25(double *state, double *unused, double *out_5897105509821386881) {
   out_5897105509821386881[0] = 0;
   out_5897105509821386881[1] = 0;
   out_5897105509821386881[2] = 0;
   out_5897105509821386881[3] = 0;
   out_5897105509821386881[4] = 0;
   out_5897105509821386881[5] = 0;
   out_5897105509821386881[6] = 1;
   out_5897105509821386881[7] = 0;
   out_5897105509821386881[8] = 0;
}
void h_24(double *state, double *unused, double *out_746443049065001390) {
   out_746443049065001390[0] = state[4];
   out_746443049065001390[1] = state[5];
}
void H_24(double *state, double *unused, double *out_2488139849162399552) {
   out_2488139849162399552[0] = 0;
   out_2488139849162399552[1] = 0;
   out_2488139849162399552[2] = 0;
   out_2488139849162399552[3] = 0;
   out_2488139849162399552[4] = 1;
   out_2488139849162399552[5] = 0;
   out_2488139849162399552[6] = 0;
   out_2488139849162399552[7] = 0;
   out_2488139849162399552[8] = 0;
   out_2488139849162399552[9] = 0;
   out_2488139849162399552[10] = 0;
   out_2488139849162399552[11] = 0;
   out_2488139849162399552[12] = 0;
   out_2488139849162399552[13] = 0;
   out_2488139849162399552[14] = 1;
   out_2488139849162399552[15] = 0;
   out_2488139849162399552[16] = 0;
   out_2488139849162399552[17] = 0;
}
void h_30(double *state, double *unused, double *out_8723417241257529034) {
   out_8723417241257529034[0] = state[4];
}
void H_30(double *state, double *unused, double *out_3378772551314138254) {
   out_3378772551314138254[0] = 0;
   out_3378772551314138254[1] = 0;
   out_3378772551314138254[2] = 0;
   out_3378772551314138254[3] = 0;
   out_3378772551314138254[4] = 1;
   out_3378772551314138254[5] = 0;
   out_3378772551314138254[6] = 0;
   out_3378772551314138254[7] = 0;
   out_3378772551314138254[8] = 0;
}
void h_26(double *state, double *unused, double *out_3324285315038173436) {
   out_3324285315038173436[0] = state[7];
}
void H_26(double *state, double *unused, double *out_8808135245014108511) {
   out_8808135245014108511[0] = 0;
   out_8808135245014108511[1] = 0;
   out_8808135245014108511[2] = 0;
   out_8808135245014108511[3] = 0;
   out_8808135245014108511[4] = 0;
   out_8808135245014108511[5] = 0;
   out_8808135245014108511[6] = 0;
   out_8808135245014108511[7] = 1;
   out_8808135245014108511[8] = 0;
}
void h_27(double *state, double *unused, double *out_6533541396062862883) {
   out_6533541396062862883[0] = state[3];
}
void H_27(double *state, double *unused, double *out_1155178480130195037) {
   out_1155178480130195037[0] = 0;
   out_1155178480130195037[1] = 0;
   out_1155178480130195037[2] = 0;
   out_1155178480130195037[3] = 1;
   out_1155178480130195037[4] = 0;
   out_1155178480130195037[5] = 0;
   out_1155178480130195037[6] = 0;
   out_1155178480130195037[7] = 0;
   out_1155178480130195037[8] = 0;
}
void h_29(double *state, double *unused, double *out_4721653564791176937) {
   out_4721653564791176937[0] = state[1];
}
void H_29(double *state, double *unused, double *out_2868541206999746070) {
   out_2868541206999746070[0] = 0;
   out_2868541206999746070[1] = 1;
   out_2868541206999746070[2] = 0;
   out_2868541206999746070[3] = 0;
   out_2868541206999746070[4] = 0;
   out_2868541206999746070[5] = 0;
   out_2868541206999746070[6] = 0;
   out_2868541206999746070[7] = 0;
   out_2868541206999746070[8] = 0;
}
void h_28(double *state, double *unused, double *out_5590031543161200448) {
   out_5590031543161200448[0] = state[0];
}
void H_28(double *state, double *unused, double *out_7950940224069276644) {
   out_7950940224069276644[0] = 1;
   out_7950940224069276644[1] = 0;
   out_7950940224069276644[2] = 0;
   out_7950940224069276644[3] = 0;
   out_7950940224069276644[4] = 0;
   out_7950940224069276644[5] = 0;
   out_7950940224069276644[6] = 0;
   out_7950940224069276644[7] = 0;
   out_7950940224069276644[8] = 0;
}
void h_31(double *state, double *unused, double *out_2292775983132363016) {
   out_2292775983132363016[0] = state[8];
}
void H_31(double *state, double *unused, double *out_5866459547944426453) {
   out_5866459547944426453[0] = 0;
   out_5866459547944426453[1] = 0;
   out_5866459547944426453[2] = 0;
   out_5866459547944426453[3] = 0;
   out_5866459547944426453[4] = 0;
   out_5866459547944426453[5] = 0;
   out_5866459547944426453[6] = 0;
   out_5866459547944426453[7] = 0;
   out_5866459547944426453[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_1345357640820912998) {
  err_fun(nom_x, delta_x, out_1345357640820912998);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_565955163399411295) {
  inv_err_fun(nom_x, true_x, out_565955163399411295);
}
void car_H_mod_fun(double *state, double *out_84820155383353969) {
  H_mod_fun(state, out_84820155383353969);
}
void car_f_fun(double *state, double dt, double *out_1203605472893796358) {
  f_fun(state,  dt, out_1203605472893796358);
}
void car_F_fun(double *state, double dt, double *out_7074967459219019176) {
  F_fun(state,  dt, out_7074967459219019176);
}
void car_h_25(double *state, double *unused, double *out_2135589314781233871) {
  h_25(state, unused, out_2135589314781233871);
}
void car_H_25(double *state, double *unused, double *out_5897105509821386881) {
  H_25(state, unused, out_5897105509821386881);
}
void car_h_24(double *state, double *unused, double *out_746443049065001390) {
  h_24(state, unused, out_746443049065001390);
}
void car_H_24(double *state, double *unused, double *out_2488139849162399552) {
  H_24(state, unused, out_2488139849162399552);
}
void car_h_30(double *state, double *unused, double *out_8723417241257529034) {
  h_30(state, unused, out_8723417241257529034);
}
void car_H_30(double *state, double *unused, double *out_3378772551314138254) {
  H_30(state, unused, out_3378772551314138254);
}
void car_h_26(double *state, double *unused, double *out_3324285315038173436) {
  h_26(state, unused, out_3324285315038173436);
}
void car_H_26(double *state, double *unused, double *out_8808135245014108511) {
  H_26(state, unused, out_8808135245014108511);
}
void car_h_27(double *state, double *unused, double *out_6533541396062862883) {
  h_27(state, unused, out_6533541396062862883);
}
void car_H_27(double *state, double *unused, double *out_1155178480130195037) {
  H_27(state, unused, out_1155178480130195037);
}
void car_h_29(double *state, double *unused, double *out_4721653564791176937) {
  h_29(state, unused, out_4721653564791176937);
}
void car_H_29(double *state, double *unused, double *out_2868541206999746070) {
  H_29(state, unused, out_2868541206999746070);
}
void car_h_28(double *state, double *unused, double *out_5590031543161200448) {
  h_28(state, unused, out_5590031543161200448);
}
void car_H_28(double *state, double *unused, double *out_7950940224069276644) {
  H_28(state, unused, out_7950940224069276644);
}
void car_h_31(double *state, double *unused, double *out_2292775983132363016) {
  h_31(state, unused, out_2292775983132363016);
}
void car_H_31(double *state, double *unused, double *out_5866459547944426453) {
  H_31(state, unused, out_5866459547944426453);
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
