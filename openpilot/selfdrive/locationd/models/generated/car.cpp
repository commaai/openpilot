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
void err_fun(double *nom_x, double *delta_x, double *out_3664446416432096692) {
   out_3664446416432096692[0] = delta_x[0] + nom_x[0];
   out_3664446416432096692[1] = delta_x[1] + nom_x[1];
   out_3664446416432096692[2] = delta_x[2] + nom_x[2];
   out_3664446416432096692[3] = delta_x[3] + nom_x[3];
   out_3664446416432096692[4] = delta_x[4] + nom_x[4];
   out_3664446416432096692[5] = delta_x[5] + nom_x[5];
   out_3664446416432096692[6] = delta_x[6] + nom_x[6];
   out_3664446416432096692[7] = delta_x[7] + nom_x[7];
   out_3664446416432096692[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2220382408259350967) {
   out_2220382408259350967[0] = -nom_x[0] + true_x[0];
   out_2220382408259350967[1] = -nom_x[1] + true_x[1];
   out_2220382408259350967[2] = -nom_x[2] + true_x[2];
   out_2220382408259350967[3] = -nom_x[3] + true_x[3];
   out_2220382408259350967[4] = -nom_x[4] + true_x[4];
   out_2220382408259350967[5] = -nom_x[5] + true_x[5];
   out_2220382408259350967[6] = -nom_x[6] + true_x[6];
   out_2220382408259350967[7] = -nom_x[7] + true_x[7];
   out_2220382408259350967[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_5101481675894590140) {
   out_5101481675894590140[0] = 1.0;
   out_5101481675894590140[1] = 0.0;
   out_5101481675894590140[2] = 0.0;
   out_5101481675894590140[3] = 0.0;
   out_5101481675894590140[4] = 0.0;
   out_5101481675894590140[5] = 0.0;
   out_5101481675894590140[6] = 0.0;
   out_5101481675894590140[7] = 0.0;
   out_5101481675894590140[8] = 0.0;
   out_5101481675894590140[9] = 0.0;
   out_5101481675894590140[10] = 1.0;
   out_5101481675894590140[11] = 0.0;
   out_5101481675894590140[12] = 0.0;
   out_5101481675894590140[13] = 0.0;
   out_5101481675894590140[14] = 0.0;
   out_5101481675894590140[15] = 0.0;
   out_5101481675894590140[16] = 0.0;
   out_5101481675894590140[17] = 0.0;
   out_5101481675894590140[18] = 0.0;
   out_5101481675894590140[19] = 0.0;
   out_5101481675894590140[20] = 1.0;
   out_5101481675894590140[21] = 0.0;
   out_5101481675894590140[22] = 0.0;
   out_5101481675894590140[23] = 0.0;
   out_5101481675894590140[24] = 0.0;
   out_5101481675894590140[25] = 0.0;
   out_5101481675894590140[26] = 0.0;
   out_5101481675894590140[27] = 0.0;
   out_5101481675894590140[28] = 0.0;
   out_5101481675894590140[29] = 0.0;
   out_5101481675894590140[30] = 1.0;
   out_5101481675894590140[31] = 0.0;
   out_5101481675894590140[32] = 0.0;
   out_5101481675894590140[33] = 0.0;
   out_5101481675894590140[34] = 0.0;
   out_5101481675894590140[35] = 0.0;
   out_5101481675894590140[36] = 0.0;
   out_5101481675894590140[37] = 0.0;
   out_5101481675894590140[38] = 0.0;
   out_5101481675894590140[39] = 0.0;
   out_5101481675894590140[40] = 1.0;
   out_5101481675894590140[41] = 0.0;
   out_5101481675894590140[42] = 0.0;
   out_5101481675894590140[43] = 0.0;
   out_5101481675894590140[44] = 0.0;
   out_5101481675894590140[45] = 0.0;
   out_5101481675894590140[46] = 0.0;
   out_5101481675894590140[47] = 0.0;
   out_5101481675894590140[48] = 0.0;
   out_5101481675894590140[49] = 0.0;
   out_5101481675894590140[50] = 1.0;
   out_5101481675894590140[51] = 0.0;
   out_5101481675894590140[52] = 0.0;
   out_5101481675894590140[53] = 0.0;
   out_5101481675894590140[54] = 0.0;
   out_5101481675894590140[55] = 0.0;
   out_5101481675894590140[56] = 0.0;
   out_5101481675894590140[57] = 0.0;
   out_5101481675894590140[58] = 0.0;
   out_5101481675894590140[59] = 0.0;
   out_5101481675894590140[60] = 1.0;
   out_5101481675894590140[61] = 0.0;
   out_5101481675894590140[62] = 0.0;
   out_5101481675894590140[63] = 0.0;
   out_5101481675894590140[64] = 0.0;
   out_5101481675894590140[65] = 0.0;
   out_5101481675894590140[66] = 0.0;
   out_5101481675894590140[67] = 0.0;
   out_5101481675894590140[68] = 0.0;
   out_5101481675894590140[69] = 0.0;
   out_5101481675894590140[70] = 1.0;
   out_5101481675894590140[71] = 0.0;
   out_5101481675894590140[72] = 0.0;
   out_5101481675894590140[73] = 0.0;
   out_5101481675894590140[74] = 0.0;
   out_5101481675894590140[75] = 0.0;
   out_5101481675894590140[76] = 0.0;
   out_5101481675894590140[77] = 0.0;
   out_5101481675894590140[78] = 0.0;
   out_5101481675894590140[79] = 0.0;
   out_5101481675894590140[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_1374429393672774254) {
   out_1374429393672774254[0] = state[0];
   out_1374429393672774254[1] = state[1];
   out_1374429393672774254[2] = state[2];
   out_1374429393672774254[3] = state[3];
   out_1374429393672774254[4] = state[4];
   out_1374429393672774254[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_1374429393672774254[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_1374429393672774254[7] = state[7];
   out_1374429393672774254[8] = state[8];
}
void F_fun(double *state, double dt, double *out_2233291611378471363) {
   out_2233291611378471363[0] = 1;
   out_2233291611378471363[1] = 0;
   out_2233291611378471363[2] = 0;
   out_2233291611378471363[3] = 0;
   out_2233291611378471363[4] = 0;
   out_2233291611378471363[5] = 0;
   out_2233291611378471363[6] = 0;
   out_2233291611378471363[7] = 0;
   out_2233291611378471363[8] = 0;
   out_2233291611378471363[9] = 0;
   out_2233291611378471363[10] = 1;
   out_2233291611378471363[11] = 0;
   out_2233291611378471363[12] = 0;
   out_2233291611378471363[13] = 0;
   out_2233291611378471363[14] = 0;
   out_2233291611378471363[15] = 0;
   out_2233291611378471363[16] = 0;
   out_2233291611378471363[17] = 0;
   out_2233291611378471363[18] = 0;
   out_2233291611378471363[19] = 0;
   out_2233291611378471363[20] = 1;
   out_2233291611378471363[21] = 0;
   out_2233291611378471363[22] = 0;
   out_2233291611378471363[23] = 0;
   out_2233291611378471363[24] = 0;
   out_2233291611378471363[25] = 0;
   out_2233291611378471363[26] = 0;
   out_2233291611378471363[27] = 0;
   out_2233291611378471363[28] = 0;
   out_2233291611378471363[29] = 0;
   out_2233291611378471363[30] = 1;
   out_2233291611378471363[31] = 0;
   out_2233291611378471363[32] = 0;
   out_2233291611378471363[33] = 0;
   out_2233291611378471363[34] = 0;
   out_2233291611378471363[35] = 0;
   out_2233291611378471363[36] = 0;
   out_2233291611378471363[37] = 0;
   out_2233291611378471363[38] = 0;
   out_2233291611378471363[39] = 0;
   out_2233291611378471363[40] = 1;
   out_2233291611378471363[41] = 0;
   out_2233291611378471363[42] = 0;
   out_2233291611378471363[43] = 0;
   out_2233291611378471363[44] = 0;
   out_2233291611378471363[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_2233291611378471363[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_2233291611378471363[47] = -dt*stiffness_front*state[0]/state[1];
   out_2233291611378471363[48] = -dt*stiffness_front*state[0]/state[1];
   out_2233291611378471363[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_2233291611378471363[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_2233291611378471363[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_2233291611378471363[52] = dt*stiffness_front*state[0]/state[1];
   out_2233291611378471363[53] = -9.8100000000000005*dt;
   out_2233291611378471363[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_2233291611378471363[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_2233291611378471363[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2233291611378471363[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2233291611378471363[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_2233291611378471363[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_2233291611378471363[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_2233291611378471363[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2233291611378471363[62] = 0;
   out_2233291611378471363[63] = 0;
   out_2233291611378471363[64] = 0;
   out_2233291611378471363[65] = 0;
   out_2233291611378471363[66] = 0;
   out_2233291611378471363[67] = 0;
   out_2233291611378471363[68] = 0;
   out_2233291611378471363[69] = 0;
   out_2233291611378471363[70] = 1;
   out_2233291611378471363[71] = 0;
   out_2233291611378471363[72] = 0;
   out_2233291611378471363[73] = 0;
   out_2233291611378471363[74] = 0;
   out_2233291611378471363[75] = 0;
   out_2233291611378471363[76] = 0;
   out_2233291611378471363[77] = 0;
   out_2233291611378471363[78] = 0;
   out_2233291611378471363[79] = 0;
   out_2233291611378471363[80] = 1;
}
void h_25(double *state, double *unused, double *out_3490619282190537336) {
   out_3490619282190537336[0] = state[6];
}
void H_25(double *state, double *unused, double *out_1578501025905098463) {
   out_1578501025905098463[0] = 0;
   out_1578501025905098463[1] = 0;
   out_1578501025905098463[2] = 0;
   out_1578501025905098463[3] = 0;
   out_1578501025905098463[4] = 0;
   out_1578501025905098463[5] = 0;
   out_1578501025905098463[6] = 1;
   out_1578501025905098463[7] = 0;
   out_1578501025905098463[8] = 0;
}
void h_24(double *state, double *unused, double *out_1487557793108610185) {
   out_1487557793108610185[0] = state[4];
   out_1487557793108610185[1] = state[5];
}
void H_24(double *state, double *unused, double *out_3751150624910598029) {
   out_3751150624910598029[0] = 0;
   out_3751150624910598029[1] = 0;
   out_3751150624910598029[2] = 0;
   out_3751150624910598029[3] = 0;
   out_3751150624910598029[4] = 1;
   out_3751150624910598029[5] = 0;
   out_3751150624910598029[6] = 0;
   out_3751150624910598029[7] = 0;
   out_3751150624910598029[8] = 0;
   out_3751150624910598029[9] = 0;
   out_3751150624910598029[10] = 0;
   out_3751150624910598029[11] = 0;
   out_3751150624910598029[12] = 0;
   out_3751150624910598029[13] = 0;
   out_3751150624910598029[14] = 1;
   out_3751150624910598029[15] = 0;
   out_3751150624910598029[16] = 0;
   out_3751150624910598029[17] = 0;
}
void h_30(double *state, double *unused, double *out_8164170727459411353) {
   out_8164170727459411353[0] = state[4];
}
void H_30(double *state, double *unused, double *out_5338189315586518292) {
   out_5338189315586518292[0] = 0;
   out_5338189315586518292[1] = 0;
   out_5338189315586518292[2] = 0;
   out_5338189315586518292[3] = 0;
   out_5338189315586518292[4] = 1;
   out_5338189315586518292[5] = 0;
   out_5338189315586518292[6] = 0;
   out_5338189315586518292[7] = 0;
   out_5338189315586518292[8] = 0;
}
void h_26(double *state, double *unused, double *out_3162497459798565708) {
   out_3162497459798565708[0] = state[7];
}
void H_26(double *state, double *unused, double *out_5320004344779154687) {
   out_5320004344779154687[0] = 0;
   out_5320004344779154687[1] = 0;
   out_5320004344779154687[2] = 0;
   out_5320004344779154687[3] = 0;
   out_5320004344779154687[4] = 0;
   out_5320004344779154687[5] = 0;
   out_5320004344779154687[6] = 0;
   out_5320004344779154687[7] = 1;
   out_5320004344779154687[8] = 0;
}
void h_27(double *state, double *unused, double *out_886213961381839685) {
   out_886213961381839685[0] = state[3];
}
void H_27(double *state, double *unused, double *out_3163426003786093381) {
   out_3163426003786093381[0] = 0;
   out_3163426003786093381[1] = 0;
   out_3163426003786093381[2] = 0;
   out_3163426003786093381[3] = 1;
   out_3163426003786093381[4] = 0;
   out_3163426003786093381[5] = 0;
   out_3163426003786093381[6] = 0;
   out_3163426003786093381[7] = 0;
   out_3163426003786093381[8] = 0;
}
void h_29(double *state, double *unused, double *out_729027293030710540) {
   out_729027293030710540[0] = state[1];
}
void H_29(double *state, double *unused, double *out_1450063276916542348) {
   out_1450063276916542348[0] = 0;
   out_1450063276916542348[1] = 1;
   out_1450063276916542348[2] = 0;
   out_1450063276916542348[3] = 0;
   out_1450063276916542348[4] = 0;
   out_1450063276916542348[5] = 0;
   out_1450063276916542348[6] = 0;
   out_1450063276916542348[7] = 0;
   out_1450063276916542348[8] = 0;
}
void h_28(double *state, double *unused, double *out_6369929755711612024) {
   out_6369929755711612024[0] = state[0];
}
void H_28(double *state, double *unused, double *out_3632335740152988226) {
   out_3632335740152988226[0] = 1;
   out_3632335740152988226[1] = 0;
   out_3632335740152988226[2] = 0;
   out_3632335740152988226[3] = 0;
   out_3632335740152988226[4] = 0;
   out_3632335740152988226[5] = 0;
   out_3632335740152988226[6] = 0;
   out_3632335740152988226[7] = 0;
   out_3632335740152988226[8] = 0;
}
void h_31(double *state, double *unused, double *out_2664827913650122793) {
   out_2664827913650122793[0] = state[8];
}
void H_31(double *state, double *unused, double *out_1547855064028138035) {
   out_1547855064028138035[0] = 0;
   out_1547855064028138035[1] = 0;
   out_1547855064028138035[2] = 0;
   out_1547855064028138035[3] = 0;
   out_1547855064028138035[4] = 0;
   out_1547855064028138035[5] = 0;
   out_1547855064028138035[6] = 0;
   out_1547855064028138035[7] = 0;
   out_1547855064028138035[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_3664446416432096692) {
  err_fun(nom_x, delta_x, out_3664446416432096692);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2220382408259350967) {
  inv_err_fun(nom_x, true_x, out_2220382408259350967);
}
void car_H_mod_fun(double *state, double *out_5101481675894590140) {
  H_mod_fun(state, out_5101481675894590140);
}
void car_f_fun(double *state, double dt, double *out_1374429393672774254) {
  f_fun(state,  dt, out_1374429393672774254);
}
void car_F_fun(double *state, double dt, double *out_2233291611378471363) {
  F_fun(state,  dt, out_2233291611378471363);
}
void car_h_25(double *state, double *unused, double *out_3490619282190537336) {
  h_25(state, unused, out_3490619282190537336);
}
void car_H_25(double *state, double *unused, double *out_1578501025905098463) {
  H_25(state, unused, out_1578501025905098463);
}
void car_h_24(double *state, double *unused, double *out_1487557793108610185) {
  h_24(state, unused, out_1487557793108610185);
}
void car_H_24(double *state, double *unused, double *out_3751150624910598029) {
  H_24(state, unused, out_3751150624910598029);
}
void car_h_30(double *state, double *unused, double *out_8164170727459411353) {
  h_30(state, unused, out_8164170727459411353);
}
void car_H_30(double *state, double *unused, double *out_5338189315586518292) {
  H_30(state, unused, out_5338189315586518292);
}
void car_h_26(double *state, double *unused, double *out_3162497459798565708) {
  h_26(state, unused, out_3162497459798565708);
}
void car_H_26(double *state, double *unused, double *out_5320004344779154687) {
  H_26(state, unused, out_5320004344779154687);
}
void car_h_27(double *state, double *unused, double *out_886213961381839685) {
  h_27(state, unused, out_886213961381839685);
}
void car_H_27(double *state, double *unused, double *out_3163426003786093381) {
  H_27(state, unused, out_3163426003786093381);
}
void car_h_29(double *state, double *unused, double *out_729027293030710540) {
  h_29(state, unused, out_729027293030710540);
}
void car_H_29(double *state, double *unused, double *out_1450063276916542348) {
  H_29(state, unused, out_1450063276916542348);
}
void car_h_28(double *state, double *unused, double *out_6369929755711612024) {
  h_28(state, unused, out_6369929755711612024);
}
void car_H_28(double *state, double *unused, double *out_3632335740152988226) {
  H_28(state, unused, out_3632335740152988226);
}
void car_h_31(double *state, double *unused, double *out_2664827913650122793) {
  h_31(state, unused, out_2664827913650122793);
}
void car_H_31(double *state, double *unused, double *out_1547855064028138035) {
  H_31(state, unused, out_1547855064028138035);
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
