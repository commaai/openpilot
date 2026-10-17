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
void err_fun(double *nom_x, double *delta_x, double *out_1193858814788666420) {
   out_1193858814788666420[0] = delta_x[0] + nom_x[0];
   out_1193858814788666420[1] = delta_x[1] + nom_x[1];
   out_1193858814788666420[2] = delta_x[2] + nom_x[2];
   out_1193858814788666420[3] = delta_x[3] + nom_x[3];
   out_1193858814788666420[4] = delta_x[4] + nom_x[4];
   out_1193858814788666420[5] = delta_x[5] + nom_x[5];
   out_1193858814788666420[6] = delta_x[6] + nom_x[6];
   out_1193858814788666420[7] = delta_x[7] + nom_x[7];
   out_1193858814788666420[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_8463344000198758499) {
   out_8463344000198758499[0] = -nom_x[0] + true_x[0];
   out_8463344000198758499[1] = -nom_x[1] + true_x[1];
   out_8463344000198758499[2] = -nom_x[2] + true_x[2];
   out_8463344000198758499[3] = -nom_x[3] + true_x[3];
   out_8463344000198758499[4] = -nom_x[4] + true_x[4];
   out_8463344000198758499[5] = -nom_x[5] + true_x[5];
   out_8463344000198758499[6] = -nom_x[6] + true_x[6];
   out_8463344000198758499[7] = -nom_x[7] + true_x[7];
   out_8463344000198758499[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_5310208349491607054) {
   out_5310208349491607054[0] = 1.0;
   out_5310208349491607054[1] = 0.0;
   out_5310208349491607054[2] = 0.0;
   out_5310208349491607054[3] = 0.0;
   out_5310208349491607054[4] = 0.0;
   out_5310208349491607054[5] = 0.0;
   out_5310208349491607054[6] = 0.0;
   out_5310208349491607054[7] = 0.0;
   out_5310208349491607054[8] = 0.0;
   out_5310208349491607054[9] = 0.0;
   out_5310208349491607054[10] = 1.0;
   out_5310208349491607054[11] = 0.0;
   out_5310208349491607054[12] = 0.0;
   out_5310208349491607054[13] = 0.0;
   out_5310208349491607054[14] = 0.0;
   out_5310208349491607054[15] = 0.0;
   out_5310208349491607054[16] = 0.0;
   out_5310208349491607054[17] = 0.0;
   out_5310208349491607054[18] = 0.0;
   out_5310208349491607054[19] = 0.0;
   out_5310208349491607054[20] = 1.0;
   out_5310208349491607054[21] = 0.0;
   out_5310208349491607054[22] = 0.0;
   out_5310208349491607054[23] = 0.0;
   out_5310208349491607054[24] = 0.0;
   out_5310208349491607054[25] = 0.0;
   out_5310208349491607054[26] = 0.0;
   out_5310208349491607054[27] = 0.0;
   out_5310208349491607054[28] = 0.0;
   out_5310208349491607054[29] = 0.0;
   out_5310208349491607054[30] = 1.0;
   out_5310208349491607054[31] = 0.0;
   out_5310208349491607054[32] = 0.0;
   out_5310208349491607054[33] = 0.0;
   out_5310208349491607054[34] = 0.0;
   out_5310208349491607054[35] = 0.0;
   out_5310208349491607054[36] = 0.0;
   out_5310208349491607054[37] = 0.0;
   out_5310208349491607054[38] = 0.0;
   out_5310208349491607054[39] = 0.0;
   out_5310208349491607054[40] = 1.0;
   out_5310208349491607054[41] = 0.0;
   out_5310208349491607054[42] = 0.0;
   out_5310208349491607054[43] = 0.0;
   out_5310208349491607054[44] = 0.0;
   out_5310208349491607054[45] = 0.0;
   out_5310208349491607054[46] = 0.0;
   out_5310208349491607054[47] = 0.0;
   out_5310208349491607054[48] = 0.0;
   out_5310208349491607054[49] = 0.0;
   out_5310208349491607054[50] = 1.0;
   out_5310208349491607054[51] = 0.0;
   out_5310208349491607054[52] = 0.0;
   out_5310208349491607054[53] = 0.0;
   out_5310208349491607054[54] = 0.0;
   out_5310208349491607054[55] = 0.0;
   out_5310208349491607054[56] = 0.0;
   out_5310208349491607054[57] = 0.0;
   out_5310208349491607054[58] = 0.0;
   out_5310208349491607054[59] = 0.0;
   out_5310208349491607054[60] = 1.0;
   out_5310208349491607054[61] = 0.0;
   out_5310208349491607054[62] = 0.0;
   out_5310208349491607054[63] = 0.0;
   out_5310208349491607054[64] = 0.0;
   out_5310208349491607054[65] = 0.0;
   out_5310208349491607054[66] = 0.0;
   out_5310208349491607054[67] = 0.0;
   out_5310208349491607054[68] = 0.0;
   out_5310208349491607054[69] = 0.0;
   out_5310208349491607054[70] = 1.0;
   out_5310208349491607054[71] = 0.0;
   out_5310208349491607054[72] = 0.0;
   out_5310208349491607054[73] = 0.0;
   out_5310208349491607054[74] = 0.0;
   out_5310208349491607054[75] = 0.0;
   out_5310208349491607054[76] = 0.0;
   out_5310208349491607054[77] = 0.0;
   out_5310208349491607054[78] = 0.0;
   out_5310208349491607054[79] = 0.0;
   out_5310208349491607054[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_3672197714364501904) {
   out_3672197714364501904[0] = state[0];
   out_3672197714364501904[1] = state[1];
   out_3672197714364501904[2] = state[2];
   out_3672197714364501904[3] = state[3];
   out_3672197714364501904[4] = state[4];
   out_3672197714364501904[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_3672197714364501904[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_3672197714364501904[7] = state[7];
   out_3672197714364501904[8] = state[8];
}
void F_fun(double *state, double dt, double *out_3045272393170548372) {
   out_3045272393170548372[0] = 1;
   out_3045272393170548372[1] = 0;
   out_3045272393170548372[2] = 0;
   out_3045272393170548372[3] = 0;
   out_3045272393170548372[4] = 0;
   out_3045272393170548372[5] = 0;
   out_3045272393170548372[6] = 0;
   out_3045272393170548372[7] = 0;
   out_3045272393170548372[8] = 0;
   out_3045272393170548372[9] = 0;
   out_3045272393170548372[10] = 1;
   out_3045272393170548372[11] = 0;
   out_3045272393170548372[12] = 0;
   out_3045272393170548372[13] = 0;
   out_3045272393170548372[14] = 0;
   out_3045272393170548372[15] = 0;
   out_3045272393170548372[16] = 0;
   out_3045272393170548372[17] = 0;
   out_3045272393170548372[18] = 0;
   out_3045272393170548372[19] = 0;
   out_3045272393170548372[20] = 1;
   out_3045272393170548372[21] = 0;
   out_3045272393170548372[22] = 0;
   out_3045272393170548372[23] = 0;
   out_3045272393170548372[24] = 0;
   out_3045272393170548372[25] = 0;
   out_3045272393170548372[26] = 0;
   out_3045272393170548372[27] = 0;
   out_3045272393170548372[28] = 0;
   out_3045272393170548372[29] = 0;
   out_3045272393170548372[30] = 1;
   out_3045272393170548372[31] = 0;
   out_3045272393170548372[32] = 0;
   out_3045272393170548372[33] = 0;
   out_3045272393170548372[34] = 0;
   out_3045272393170548372[35] = 0;
   out_3045272393170548372[36] = 0;
   out_3045272393170548372[37] = 0;
   out_3045272393170548372[38] = 0;
   out_3045272393170548372[39] = 0;
   out_3045272393170548372[40] = 1;
   out_3045272393170548372[41] = 0;
   out_3045272393170548372[42] = 0;
   out_3045272393170548372[43] = 0;
   out_3045272393170548372[44] = 0;
   out_3045272393170548372[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_3045272393170548372[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_3045272393170548372[47] = -dt*stiffness_front*state[0]/state[1];
   out_3045272393170548372[48] = -dt*stiffness_front*state[0]/state[1];
   out_3045272393170548372[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_3045272393170548372[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_3045272393170548372[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_3045272393170548372[52] = dt*stiffness_front*state[0]/state[1];
   out_3045272393170548372[53] = -9.8100000000000005*dt;
   out_3045272393170548372[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_3045272393170548372[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_3045272393170548372[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3045272393170548372[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3045272393170548372[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_3045272393170548372[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_3045272393170548372[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_3045272393170548372[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3045272393170548372[62] = 0;
   out_3045272393170548372[63] = 0;
   out_3045272393170548372[64] = 0;
   out_3045272393170548372[65] = 0;
   out_3045272393170548372[66] = 0;
   out_3045272393170548372[67] = 0;
   out_3045272393170548372[68] = 0;
   out_3045272393170548372[69] = 0;
   out_3045272393170548372[70] = 1;
   out_3045272393170548372[71] = 0;
   out_3045272393170548372[72] = 0;
   out_3045272393170548372[73] = 0;
   out_3045272393170548372[74] = 0;
   out_3045272393170548372[75] = 0;
   out_3045272393170548372[76] = 0;
   out_3045272393170548372[77] = 0;
   out_3045272393170548372[78] = 0;
   out_3045272393170548372[79] = 0;
   out_3045272393170548372[80] = 1;
}
void h_25(double *state, double *unused, double *out_770246915850585039) {
   out_770246915850585039[0] = state[6];
}
void H_25(double *state, double *unused, double *out_406516524306942214) {
   out_406516524306942214[0] = 0;
   out_406516524306942214[1] = 0;
   out_406516524306942214[2] = 0;
   out_406516524306942214[3] = 0;
   out_406516524306942214[4] = 0;
   out_406516524306942214[5] = 0;
   out_406516524306942214[6] = 1;
   out_406516524306942214[7] = 0;
   out_406516524306942214[8] = 0;
}
void h_24(double *state, double *unused, double *out_3584346622172585855) {
   out_3584346622172585855[0] = state[4];
   out_3584346622172585855[1] = state[5];
}
void H_24(double *state, double *unused, double *out_6169055282284575887) {
   out_6169055282284575887[0] = 0;
   out_6169055282284575887[1] = 0;
   out_6169055282284575887[2] = 0;
   out_6169055282284575887[3] = 0;
   out_6169055282284575887[4] = 1;
   out_6169055282284575887[5] = 0;
   out_6169055282284575887[6] = 0;
   out_6169055282284575887[7] = 0;
   out_6169055282284575887[8] = 0;
   out_6169055282284575887[9] = 0;
   out_6169055282284575887[10] = 0;
   out_6169055282284575887[11] = 0;
   out_6169055282284575887[12] = 0;
   out_6169055282284575887[13] = 0;
   out_6169055282284575887[14] = 1;
   out_6169055282284575887[15] = 0;
   out_6169055282284575887[16] = 0;
   out_6169055282284575887[17] = 0;
}
void h_30(double *state, double *unused, double *out_9036619681140424693) {
   out_9036619681140424693[0] = state[4];
}
void H_30(double *state, double *unused, double *out_6510173817184674541) {
   out_6510173817184674541[0] = 0;
   out_6510173817184674541[1] = 0;
   out_6510173817184674541[2] = 0;
   out_6510173817184674541[3] = 0;
   out_6510173817184674541[4] = 1;
   out_6510173817184674541[5] = 0;
   out_6510173817184674541[6] = 0;
   out_6510173817184674541[7] = 0;
   out_6510173817184674541[8] = 0;
}
void h_26(double *state, double *unused, double *out_8323397744212662596) {
   out_8323397744212662596[0] = state[7];
}
void H_26(double *state, double *unused, double *out_4148019843180998438) {
   out_4148019843180998438[0] = 0;
   out_4148019843180998438[1] = 0;
   out_4148019843180998438[2] = 0;
   out_4148019843180998438[3] = 0;
   out_4148019843180998438[4] = 0;
   out_4148019843180998438[5] = 0;
   out_4148019843180998438[6] = 0;
   out_4148019843180998438[7] = 1;
   out_4148019843180998438[8] = 0;
}
void h_27(double *state, double *unused, double *out_2505600896896871550) {
   out_2505600896896871550[0] = state[3];
}
void H_27(double *state, double *unused, double *out_2710618783250607195) {
   out_2710618783250607195[0] = 0;
   out_2710618783250607195[1] = 0;
   out_2710618783250607195[2] = 0;
   out_2710618783250607195[3] = 1;
   out_2710618783250607195[4] = 0;
   out_2710618783250607195[5] = 0;
   out_2710618783250607195[6] = 0;
   out_2710618783250607195[7] = 0;
   out_2710618783250607195[8] = 0;
}
void h_29(double *state, double *unused, double *out_4265234329453479386) {
   out_4265234329453479386[0] = state[1];
}
void H_29(double *state, double *unused, double *out_2622047778514698597) {
   out_2622047778514698597[0] = 0;
   out_2622047778514698597[1] = 1;
   out_2622047778514698597[2] = 0;
   out_2622047778514698597[3] = 0;
   out_2622047778514698597[4] = 0;
   out_2622047778514698597[5] = 0;
   out_2622047778514698597[6] = 0;
   out_2622047778514698597[7] = 0;
   out_2622047778514698597[8] = 0;
}
void h_28(double *state, double *unused, double *out_7507274164557717195) {
   out_7507274164557717195[0] = state[0];
}
void H_28(double *state, double *unused, double *out_2460351238554831977) {
   out_2460351238554831977[0] = 1;
   out_2460351238554831977[1] = 0;
   out_2460351238554831977[2] = 0;
   out_2460351238554831977[3] = 0;
   out_2460351238554831977[4] = 0;
   out_2460351238554831977[5] = 0;
   out_2460351238554831977[6] = 0;
   out_2460351238554831977[7] = 0;
   out_2460351238554831977[8] = 0;
}
void h_31(double *state, double *unused, double *out_1045440978135090928) {
   out_1045440978135090928[0] = state[8];
}
void H_31(double *state, double *unused, double *out_375870562429981786) {
   out_375870562429981786[0] = 0;
   out_375870562429981786[1] = 0;
   out_375870562429981786[2] = 0;
   out_375870562429981786[3] = 0;
   out_375870562429981786[4] = 0;
   out_375870562429981786[5] = 0;
   out_375870562429981786[6] = 0;
   out_375870562429981786[7] = 0;
   out_375870562429981786[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_1193858814788666420) {
  err_fun(nom_x, delta_x, out_1193858814788666420);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_8463344000198758499) {
  inv_err_fun(nom_x, true_x, out_8463344000198758499);
}
void car_H_mod_fun(double *state, double *out_5310208349491607054) {
  H_mod_fun(state, out_5310208349491607054);
}
void car_f_fun(double *state, double dt, double *out_3672197714364501904) {
  f_fun(state,  dt, out_3672197714364501904);
}
void car_F_fun(double *state, double dt, double *out_3045272393170548372) {
  F_fun(state,  dt, out_3045272393170548372);
}
void car_h_25(double *state, double *unused, double *out_770246915850585039) {
  h_25(state, unused, out_770246915850585039);
}
void car_H_25(double *state, double *unused, double *out_406516524306942214) {
  H_25(state, unused, out_406516524306942214);
}
void car_h_24(double *state, double *unused, double *out_3584346622172585855) {
  h_24(state, unused, out_3584346622172585855);
}
void car_H_24(double *state, double *unused, double *out_6169055282284575887) {
  H_24(state, unused, out_6169055282284575887);
}
void car_h_30(double *state, double *unused, double *out_9036619681140424693) {
  h_30(state, unused, out_9036619681140424693);
}
void car_H_30(double *state, double *unused, double *out_6510173817184674541) {
  H_30(state, unused, out_6510173817184674541);
}
void car_h_26(double *state, double *unused, double *out_8323397744212662596) {
  h_26(state, unused, out_8323397744212662596);
}
void car_H_26(double *state, double *unused, double *out_4148019843180998438) {
  H_26(state, unused, out_4148019843180998438);
}
void car_h_27(double *state, double *unused, double *out_2505600896896871550) {
  h_27(state, unused, out_2505600896896871550);
}
void car_H_27(double *state, double *unused, double *out_2710618783250607195) {
  H_27(state, unused, out_2710618783250607195);
}
void car_h_29(double *state, double *unused, double *out_4265234329453479386) {
  h_29(state, unused, out_4265234329453479386);
}
void car_H_29(double *state, double *unused, double *out_2622047778514698597) {
  H_29(state, unused, out_2622047778514698597);
}
void car_h_28(double *state, double *unused, double *out_7507274164557717195) {
  h_28(state, unused, out_7507274164557717195);
}
void car_H_28(double *state, double *unused, double *out_2460351238554831977) {
  H_28(state, unused, out_2460351238554831977);
}
void car_h_31(double *state, double *unused, double *out_1045440978135090928) {
  h_31(state, unused, out_1045440978135090928);
}
void car_H_31(double *state, double *unused, double *out_375870562429981786) {
  H_31(state, unused, out_375870562429981786);
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
