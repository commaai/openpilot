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
void err_fun(double *nom_x, double *delta_x, double *out_2927168366743118455) {
   out_2927168366743118455[0] = delta_x[0] + nom_x[0];
   out_2927168366743118455[1] = delta_x[1] + nom_x[1];
   out_2927168366743118455[2] = delta_x[2] + nom_x[2];
   out_2927168366743118455[3] = delta_x[3] + nom_x[3];
   out_2927168366743118455[4] = delta_x[4] + nom_x[4];
   out_2927168366743118455[5] = delta_x[5] + nom_x[5];
   out_2927168366743118455[6] = delta_x[6] + nom_x[6];
   out_2927168366743118455[7] = delta_x[7] + nom_x[7];
   out_2927168366743118455[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_633536280694991254) {
   out_633536280694991254[0] = -nom_x[0] + true_x[0];
   out_633536280694991254[1] = -nom_x[1] + true_x[1];
   out_633536280694991254[2] = -nom_x[2] + true_x[2];
   out_633536280694991254[3] = -nom_x[3] + true_x[3];
   out_633536280694991254[4] = -nom_x[4] + true_x[4];
   out_633536280694991254[5] = -nom_x[5] + true_x[5];
   out_633536280694991254[6] = -nom_x[6] + true_x[6];
   out_633536280694991254[7] = -nom_x[7] + true_x[7];
   out_633536280694991254[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_3756852632545933054) {
   out_3756852632545933054[0] = 1.0;
   out_3756852632545933054[1] = 0.0;
   out_3756852632545933054[2] = 0.0;
   out_3756852632545933054[3] = 0.0;
   out_3756852632545933054[4] = 0.0;
   out_3756852632545933054[5] = 0.0;
   out_3756852632545933054[6] = 0.0;
   out_3756852632545933054[7] = 0.0;
   out_3756852632545933054[8] = 0.0;
   out_3756852632545933054[9] = 0.0;
   out_3756852632545933054[10] = 1.0;
   out_3756852632545933054[11] = 0.0;
   out_3756852632545933054[12] = 0.0;
   out_3756852632545933054[13] = 0.0;
   out_3756852632545933054[14] = 0.0;
   out_3756852632545933054[15] = 0.0;
   out_3756852632545933054[16] = 0.0;
   out_3756852632545933054[17] = 0.0;
   out_3756852632545933054[18] = 0.0;
   out_3756852632545933054[19] = 0.0;
   out_3756852632545933054[20] = 1.0;
   out_3756852632545933054[21] = 0.0;
   out_3756852632545933054[22] = 0.0;
   out_3756852632545933054[23] = 0.0;
   out_3756852632545933054[24] = 0.0;
   out_3756852632545933054[25] = 0.0;
   out_3756852632545933054[26] = 0.0;
   out_3756852632545933054[27] = 0.0;
   out_3756852632545933054[28] = 0.0;
   out_3756852632545933054[29] = 0.0;
   out_3756852632545933054[30] = 1.0;
   out_3756852632545933054[31] = 0.0;
   out_3756852632545933054[32] = 0.0;
   out_3756852632545933054[33] = 0.0;
   out_3756852632545933054[34] = 0.0;
   out_3756852632545933054[35] = 0.0;
   out_3756852632545933054[36] = 0.0;
   out_3756852632545933054[37] = 0.0;
   out_3756852632545933054[38] = 0.0;
   out_3756852632545933054[39] = 0.0;
   out_3756852632545933054[40] = 1.0;
   out_3756852632545933054[41] = 0.0;
   out_3756852632545933054[42] = 0.0;
   out_3756852632545933054[43] = 0.0;
   out_3756852632545933054[44] = 0.0;
   out_3756852632545933054[45] = 0.0;
   out_3756852632545933054[46] = 0.0;
   out_3756852632545933054[47] = 0.0;
   out_3756852632545933054[48] = 0.0;
   out_3756852632545933054[49] = 0.0;
   out_3756852632545933054[50] = 1.0;
   out_3756852632545933054[51] = 0.0;
   out_3756852632545933054[52] = 0.0;
   out_3756852632545933054[53] = 0.0;
   out_3756852632545933054[54] = 0.0;
   out_3756852632545933054[55] = 0.0;
   out_3756852632545933054[56] = 0.0;
   out_3756852632545933054[57] = 0.0;
   out_3756852632545933054[58] = 0.0;
   out_3756852632545933054[59] = 0.0;
   out_3756852632545933054[60] = 1.0;
   out_3756852632545933054[61] = 0.0;
   out_3756852632545933054[62] = 0.0;
   out_3756852632545933054[63] = 0.0;
   out_3756852632545933054[64] = 0.0;
   out_3756852632545933054[65] = 0.0;
   out_3756852632545933054[66] = 0.0;
   out_3756852632545933054[67] = 0.0;
   out_3756852632545933054[68] = 0.0;
   out_3756852632545933054[69] = 0.0;
   out_3756852632545933054[70] = 1.0;
   out_3756852632545933054[71] = 0.0;
   out_3756852632545933054[72] = 0.0;
   out_3756852632545933054[73] = 0.0;
   out_3756852632545933054[74] = 0.0;
   out_3756852632545933054[75] = 0.0;
   out_3756852632545933054[76] = 0.0;
   out_3756852632545933054[77] = 0.0;
   out_3756852632545933054[78] = 0.0;
   out_3756852632545933054[79] = 0.0;
   out_3756852632545933054[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_1622189069467381624) {
   out_1622189069467381624[0] = state[0];
   out_1622189069467381624[1] = state[1];
   out_1622189069467381624[2] = state[2];
   out_1622189069467381624[3] = state[3];
   out_1622189069467381624[4] = state[4];
   out_1622189069467381624[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_1622189069467381624[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_1622189069467381624[7] = state[7];
   out_1622189069467381624[8] = state[8];
}
void F_fun(double *state, double dt, double *out_1503549483011366244) {
   out_1503549483011366244[0] = 1;
   out_1503549483011366244[1] = 0;
   out_1503549483011366244[2] = 0;
   out_1503549483011366244[3] = 0;
   out_1503549483011366244[4] = 0;
   out_1503549483011366244[5] = 0;
   out_1503549483011366244[6] = 0;
   out_1503549483011366244[7] = 0;
   out_1503549483011366244[8] = 0;
   out_1503549483011366244[9] = 0;
   out_1503549483011366244[10] = 1;
   out_1503549483011366244[11] = 0;
   out_1503549483011366244[12] = 0;
   out_1503549483011366244[13] = 0;
   out_1503549483011366244[14] = 0;
   out_1503549483011366244[15] = 0;
   out_1503549483011366244[16] = 0;
   out_1503549483011366244[17] = 0;
   out_1503549483011366244[18] = 0;
   out_1503549483011366244[19] = 0;
   out_1503549483011366244[20] = 1;
   out_1503549483011366244[21] = 0;
   out_1503549483011366244[22] = 0;
   out_1503549483011366244[23] = 0;
   out_1503549483011366244[24] = 0;
   out_1503549483011366244[25] = 0;
   out_1503549483011366244[26] = 0;
   out_1503549483011366244[27] = 0;
   out_1503549483011366244[28] = 0;
   out_1503549483011366244[29] = 0;
   out_1503549483011366244[30] = 1;
   out_1503549483011366244[31] = 0;
   out_1503549483011366244[32] = 0;
   out_1503549483011366244[33] = 0;
   out_1503549483011366244[34] = 0;
   out_1503549483011366244[35] = 0;
   out_1503549483011366244[36] = 0;
   out_1503549483011366244[37] = 0;
   out_1503549483011366244[38] = 0;
   out_1503549483011366244[39] = 0;
   out_1503549483011366244[40] = 1;
   out_1503549483011366244[41] = 0;
   out_1503549483011366244[42] = 0;
   out_1503549483011366244[43] = 0;
   out_1503549483011366244[44] = 0;
   out_1503549483011366244[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_1503549483011366244[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_1503549483011366244[47] = -dt*stiffness_front*state[0]/state[1];
   out_1503549483011366244[48] = -dt*stiffness_front*state[0]/state[1];
   out_1503549483011366244[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_1503549483011366244[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_1503549483011366244[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_1503549483011366244[52] = dt*stiffness_front*state[0]/state[1];
   out_1503549483011366244[53] = -9.8100000000000005*dt;
   out_1503549483011366244[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_1503549483011366244[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_1503549483011366244[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1503549483011366244[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1503549483011366244[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_1503549483011366244[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_1503549483011366244[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_1503549483011366244[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1503549483011366244[62] = 0;
   out_1503549483011366244[63] = 0;
   out_1503549483011366244[64] = 0;
   out_1503549483011366244[65] = 0;
   out_1503549483011366244[66] = 0;
   out_1503549483011366244[67] = 0;
   out_1503549483011366244[68] = 0;
   out_1503549483011366244[69] = 0;
   out_1503549483011366244[70] = 1;
   out_1503549483011366244[71] = 0;
   out_1503549483011366244[72] = 0;
   out_1503549483011366244[73] = 0;
   out_1503549483011366244[74] = 0;
   out_1503549483011366244[75] = 0;
   out_1503549483011366244[76] = 0;
   out_1503549483011366244[77] = 0;
   out_1503549483011366244[78] = 0;
   out_1503549483011366244[79] = 0;
   out_1503549483011366244[80] = 1;
}
void h_25(double *state, double *unused, double *out_921512697149304703) {
   out_921512697149304703[0] = state[6];
}
void H_25(double *state, double *unused, double *out_5496703215251381325) {
   out_5496703215251381325[0] = 0;
   out_5496703215251381325[1] = 0;
   out_5496703215251381325[2] = 0;
   out_5496703215251381325[3] = 0;
   out_5496703215251381325[4] = 0;
   out_5496703215251381325[5] = 0;
   out_5496703215251381325[6] = 1;
   out_5496703215251381325[7] = 0;
   out_5496703215251381325[8] = 0;
}
void h_24(double *state, double *unused, double *out_2538808957834133388) {
   out_2538808957834133388[0] = state[4];
   out_2538808957834133388[1] = state[5];
}
void H_24(double *state, double *unused, double *out_3324053616245881759) {
   out_3324053616245881759[0] = 0;
   out_3324053616245881759[1] = 0;
   out_3324053616245881759[2] = 0;
   out_3324053616245881759[3] = 0;
   out_3324053616245881759[4] = 1;
   out_3324053616245881759[5] = 0;
   out_3324053616245881759[6] = 0;
   out_3324053616245881759[7] = 0;
   out_3324053616245881759[8] = 0;
   out_3324053616245881759[9] = 0;
   out_3324053616245881759[10] = 0;
   out_3324053616245881759[11] = 0;
   out_3324053616245881759[12] = 0;
   out_3324053616245881759[13] = 0;
   out_3324053616245881759[14] = 1;
   out_3324053616245881759[15] = 0;
   out_3324053616245881759[16] = 0;
   out_3324053616245881759[17] = 0;
}
void h_30(double *state, double *unused, double *out_2715936831104717647) {
   out_2715936831104717647[0] = state[4];
}
void H_30(double *state, double *unused, double *out_8015036173758629952) {
   out_8015036173758629952[0] = 0;
   out_8015036173758629952[1] = 0;
   out_8015036173758629952[2] = 0;
   out_8015036173758629952[3] = 0;
   out_8015036173758629952[4] = 1;
   out_8015036173758629952[5] = 0;
   out_8015036173758629952[6] = 0;
   out_8015036173758629952[7] = 0;
   out_8015036173758629952[8] = 0;
}
void h_26(double *state, double *unused, double *out_4838765396476033641) {
   out_4838765396476033641[0] = state[7];
}
void H_26(double *state, double *unused, double *out_8801229185012181926) {
   out_8801229185012181926[0] = 0;
   out_8801229185012181926[1] = 0;
   out_8801229185012181926[2] = 0;
   out_8801229185012181926[3] = 0;
   out_8801229185012181926[4] = 0;
   out_8801229185012181926[5] = 0;
   out_8801229185012181926[6] = 0;
   out_8801229185012181926[7] = 1;
   out_8801229185012181926[8] = 0;
}
void h_27(double *state, double *unused, double *out_2089256498225446440) {
   out_2089256498225446440[0] = state[3];
}
void H_27(double *state, double *unused, double *out_8208113828766978447) {
   out_8208113828766978447[0] = 0;
   out_8208113828766978447[1] = 0;
   out_8208113828766978447[2] = 0;
   out_8208113828766978447[3] = 1;
   out_8208113828766978447[4] = 0;
   out_8208113828766978447[5] = 0;
   out_8208113828766978447[6] = 0;
   out_8208113828766978447[7] = 0;
   out_8208113828766978447[8] = 0;
}
void h_29(double *state, double *unused, double *out_7717610098765563292) {
   out_7717610098765563292[0] = state[1];
}
void H_29(double *state, double *unused, double *out_8525267518073022136) {
   out_8525267518073022136[0] = 0;
   out_8525267518073022136[1] = 1;
   out_8525267518073022136[2] = 0;
   out_8525267518073022136[3] = 0;
   out_8525267518073022136[4] = 0;
   out_8525267518073022136[5] = 0;
   out_8525267518073022136[6] = 0;
   out_8525267518073022136[7] = 0;
   out_8525267518073022136[8] = 0;
}
void h_28(double *state, double *unused, double *out_4901808601495299823) {
   out_4901808601495299823[0] = state[0];
}
void H_28(double *state, double *unused, double *out_3442868501003491562) {
   out_3442868501003491562[0] = 1;
   out_3442868501003491562[1] = 0;
   out_3442868501003491562[2] = 0;
   out_3442868501003491562[3] = 0;
   out_3442868501003491562[4] = 0;
   out_3442868501003491562[5] = 0;
   out_3442868501003491562[6] = 0;
   out_3442868501003491562[7] = 0;
   out_3442868501003491562[8] = 0;
}
void h_31(double *state, double *unused, double *out_320094539535941149) {
   out_320094539535941149[0] = state[8];
}
void H_31(double *state, double *unused, double *out_5873365607946353038) {
   out_5873365607946353038[0] = 0;
   out_5873365607946353038[1] = 0;
   out_5873365607946353038[2] = 0;
   out_5873365607946353038[3] = 0;
   out_5873365607946353038[4] = 0;
   out_5873365607946353038[5] = 0;
   out_5873365607946353038[6] = 0;
   out_5873365607946353038[7] = 0;
   out_5873365607946353038[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_2927168366743118455) {
  err_fun(nom_x, delta_x, out_2927168366743118455);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_633536280694991254) {
  inv_err_fun(nom_x, true_x, out_633536280694991254);
}
void car_H_mod_fun(double *state, double *out_3756852632545933054) {
  H_mod_fun(state, out_3756852632545933054);
}
void car_f_fun(double *state, double dt, double *out_1622189069467381624) {
  f_fun(state,  dt, out_1622189069467381624);
}
void car_F_fun(double *state, double dt, double *out_1503549483011366244) {
  F_fun(state,  dt, out_1503549483011366244);
}
void car_h_25(double *state, double *unused, double *out_921512697149304703) {
  h_25(state, unused, out_921512697149304703);
}
void car_H_25(double *state, double *unused, double *out_5496703215251381325) {
  H_25(state, unused, out_5496703215251381325);
}
void car_h_24(double *state, double *unused, double *out_2538808957834133388) {
  h_24(state, unused, out_2538808957834133388);
}
void car_H_24(double *state, double *unused, double *out_3324053616245881759) {
  H_24(state, unused, out_3324053616245881759);
}
void car_h_30(double *state, double *unused, double *out_2715936831104717647) {
  h_30(state, unused, out_2715936831104717647);
}
void car_H_30(double *state, double *unused, double *out_8015036173758629952) {
  H_30(state, unused, out_8015036173758629952);
}
void car_h_26(double *state, double *unused, double *out_4838765396476033641) {
  h_26(state, unused, out_4838765396476033641);
}
void car_H_26(double *state, double *unused, double *out_8801229185012181926) {
  H_26(state, unused, out_8801229185012181926);
}
void car_h_27(double *state, double *unused, double *out_2089256498225446440) {
  h_27(state, unused, out_2089256498225446440);
}
void car_H_27(double *state, double *unused, double *out_8208113828766978447) {
  H_27(state, unused, out_8208113828766978447);
}
void car_h_29(double *state, double *unused, double *out_7717610098765563292) {
  h_29(state, unused, out_7717610098765563292);
}
void car_H_29(double *state, double *unused, double *out_8525267518073022136) {
  H_29(state, unused, out_8525267518073022136);
}
void car_h_28(double *state, double *unused, double *out_4901808601495299823) {
  h_28(state, unused, out_4901808601495299823);
}
void car_H_28(double *state, double *unused, double *out_3442868501003491562) {
  H_28(state, unused, out_3442868501003491562);
}
void car_h_31(double *state, double *unused, double *out_320094539535941149) {
  h_31(state, unused, out_320094539535941149);
}
void car_H_31(double *state, double *unused, double *out_5873365607946353038) {
  H_31(state, unused, out_5873365607946353038);
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
