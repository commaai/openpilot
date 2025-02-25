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
void err_fun(double *nom_x, double *delta_x, double *out_7082197508459355294) {
   out_7082197508459355294[0] = delta_x[0] + nom_x[0];
   out_7082197508459355294[1] = delta_x[1] + nom_x[1];
   out_7082197508459355294[2] = delta_x[2] + nom_x[2];
   out_7082197508459355294[3] = delta_x[3] + nom_x[3];
   out_7082197508459355294[4] = delta_x[4] + nom_x[4];
   out_7082197508459355294[5] = delta_x[5] + nom_x[5];
   out_7082197508459355294[6] = delta_x[6] + nom_x[6];
   out_7082197508459355294[7] = delta_x[7] + nom_x[7];
   out_7082197508459355294[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_7553533010452763308) {
   out_7553533010452763308[0] = -nom_x[0] + true_x[0];
   out_7553533010452763308[1] = -nom_x[1] + true_x[1];
   out_7553533010452763308[2] = -nom_x[2] + true_x[2];
   out_7553533010452763308[3] = -nom_x[3] + true_x[3];
   out_7553533010452763308[4] = -nom_x[4] + true_x[4];
   out_7553533010452763308[5] = -nom_x[5] + true_x[5];
   out_7553533010452763308[6] = -nom_x[6] + true_x[6];
   out_7553533010452763308[7] = -nom_x[7] + true_x[7];
   out_7553533010452763308[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_2748508627650133070) {
   out_2748508627650133070[0] = 1.0;
   out_2748508627650133070[1] = 0;
   out_2748508627650133070[2] = 0;
   out_2748508627650133070[3] = 0;
   out_2748508627650133070[4] = 0;
   out_2748508627650133070[5] = 0;
   out_2748508627650133070[6] = 0;
   out_2748508627650133070[7] = 0;
   out_2748508627650133070[8] = 0;
   out_2748508627650133070[9] = 0;
   out_2748508627650133070[10] = 1.0;
   out_2748508627650133070[11] = 0;
   out_2748508627650133070[12] = 0;
   out_2748508627650133070[13] = 0;
   out_2748508627650133070[14] = 0;
   out_2748508627650133070[15] = 0;
   out_2748508627650133070[16] = 0;
   out_2748508627650133070[17] = 0;
   out_2748508627650133070[18] = 0;
   out_2748508627650133070[19] = 0;
   out_2748508627650133070[20] = 1.0;
   out_2748508627650133070[21] = 0;
   out_2748508627650133070[22] = 0;
   out_2748508627650133070[23] = 0;
   out_2748508627650133070[24] = 0;
   out_2748508627650133070[25] = 0;
   out_2748508627650133070[26] = 0;
   out_2748508627650133070[27] = 0;
   out_2748508627650133070[28] = 0;
   out_2748508627650133070[29] = 0;
   out_2748508627650133070[30] = 1.0;
   out_2748508627650133070[31] = 0;
   out_2748508627650133070[32] = 0;
   out_2748508627650133070[33] = 0;
   out_2748508627650133070[34] = 0;
   out_2748508627650133070[35] = 0;
   out_2748508627650133070[36] = 0;
   out_2748508627650133070[37] = 0;
   out_2748508627650133070[38] = 0;
   out_2748508627650133070[39] = 0;
   out_2748508627650133070[40] = 1.0;
   out_2748508627650133070[41] = 0;
   out_2748508627650133070[42] = 0;
   out_2748508627650133070[43] = 0;
   out_2748508627650133070[44] = 0;
   out_2748508627650133070[45] = 0;
   out_2748508627650133070[46] = 0;
   out_2748508627650133070[47] = 0;
   out_2748508627650133070[48] = 0;
   out_2748508627650133070[49] = 0;
   out_2748508627650133070[50] = 1.0;
   out_2748508627650133070[51] = 0;
   out_2748508627650133070[52] = 0;
   out_2748508627650133070[53] = 0;
   out_2748508627650133070[54] = 0;
   out_2748508627650133070[55] = 0;
   out_2748508627650133070[56] = 0;
   out_2748508627650133070[57] = 0;
   out_2748508627650133070[58] = 0;
   out_2748508627650133070[59] = 0;
   out_2748508627650133070[60] = 1.0;
   out_2748508627650133070[61] = 0;
   out_2748508627650133070[62] = 0;
   out_2748508627650133070[63] = 0;
   out_2748508627650133070[64] = 0;
   out_2748508627650133070[65] = 0;
   out_2748508627650133070[66] = 0;
   out_2748508627650133070[67] = 0;
   out_2748508627650133070[68] = 0;
   out_2748508627650133070[69] = 0;
   out_2748508627650133070[70] = 1.0;
   out_2748508627650133070[71] = 0;
   out_2748508627650133070[72] = 0;
   out_2748508627650133070[73] = 0;
   out_2748508627650133070[74] = 0;
   out_2748508627650133070[75] = 0;
   out_2748508627650133070[76] = 0;
   out_2748508627650133070[77] = 0;
   out_2748508627650133070[78] = 0;
   out_2748508627650133070[79] = 0;
   out_2748508627650133070[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_1858329490310860980) {
   out_1858329490310860980[0] = state[0];
   out_1858329490310860980[1] = state[1];
   out_1858329490310860980[2] = state[2];
   out_1858329490310860980[3] = state[3];
   out_1858329490310860980[4] = state[4];
   out_1858329490310860980[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_1858329490310860980[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_1858329490310860980[7] = state[7];
   out_1858329490310860980[8] = state[8];
}
void F_fun(double *state, double dt, double *out_8266273866488598058) {
   out_8266273866488598058[0] = 1;
   out_8266273866488598058[1] = 0;
   out_8266273866488598058[2] = 0;
   out_8266273866488598058[3] = 0;
   out_8266273866488598058[4] = 0;
   out_8266273866488598058[5] = 0;
   out_8266273866488598058[6] = 0;
   out_8266273866488598058[7] = 0;
   out_8266273866488598058[8] = 0;
   out_8266273866488598058[9] = 0;
   out_8266273866488598058[10] = 1;
   out_8266273866488598058[11] = 0;
   out_8266273866488598058[12] = 0;
   out_8266273866488598058[13] = 0;
   out_8266273866488598058[14] = 0;
   out_8266273866488598058[15] = 0;
   out_8266273866488598058[16] = 0;
   out_8266273866488598058[17] = 0;
   out_8266273866488598058[18] = 0;
   out_8266273866488598058[19] = 0;
   out_8266273866488598058[20] = 1;
   out_8266273866488598058[21] = 0;
   out_8266273866488598058[22] = 0;
   out_8266273866488598058[23] = 0;
   out_8266273866488598058[24] = 0;
   out_8266273866488598058[25] = 0;
   out_8266273866488598058[26] = 0;
   out_8266273866488598058[27] = 0;
   out_8266273866488598058[28] = 0;
   out_8266273866488598058[29] = 0;
   out_8266273866488598058[30] = 1;
   out_8266273866488598058[31] = 0;
   out_8266273866488598058[32] = 0;
   out_8266273866488598058[33] = 0;
   out_8266273866488598058[34] = 0;
   out_8266273866488598058[35] = 0;
   out_8266273866488598058[36] = 0;
   out_8266273866488598058[37] = 0;
   out_8266273866488598058[38] = 0;
   out_8266273866488598058[39] = 0;
   out_8266273866488598058[40] = 1;
   out_8266273866488598058[41] = 0;
   out_8266273866488598058[42] = 0;
   out_8266273866488598058[43] = 0;
   out_8266273866488598058[44] = 0;
   out_8266273866488598058[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_8266273866488598058[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_8266273866488598058[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8266273866488598058[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8266273866488598058[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_8266273866488598058[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_8266273866488598058[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_8266273866488598058[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_8266273866488598058[53] = -9.8000000000000007*dt;
   out_8266273866488598058[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_8266273866488598058[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_8266273866488598058[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8266273866488598058[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8266273866488598058[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_8266273866488598058[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_8266273866488598058[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_8266273866488598058[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8266273866488598058[62] = 0;
   out_8266273866488598058[63] = 0;
   out_8266273866488598058[64] = 0;
   out_8266273866488598058[65] = 0;
   out_8266273866488598058[66] = 0;
   out_8266273866488598058[67] = 0;
   out_8266273866488598058[68] = 0;
   out_8266273866488598058[69] = 0;
   out_8266273866488598058[70] = 1;
   out_8266273866488598058[71] = 0;
   out_8266273866488598058[72] = 0;
   out_8266273866488598058[73] = 0;
   out_8266273866488598058[74] = 0;
   out_8266273866488598058[75] = 0;
   out_8266273866488598058[76] = 0;
   out_8266273866488598058[77] = 0;
   out_8266273866488598058[78] = 0;
   out_8266273866488598058[79] = 0;
   out_8266273866488598058[80] = 1;
}
void h_25(double *state, double *unused, double *out_8442152877268712832) {
   out_8442152877268712832[0] = state[6];
}
void H_25(double *state, double *unused, double *out_6916230363893539613) {
   out_6916230363893539613[0] = 0;
   out_6916230363893539613[1] = 0;
   out_6916230363893539613[2] = 0;
   out_6916230363893539613[3] = 0;
   out_6916230363893539613[4] = 0;
   out_6916230363893539613[5] = 0;
   out_6916230363893539613[6] = 1;
   out_6916230363893539613[7] = 0;
   out_6916230363893539613[8] = 0;
}
void h_24(double *state, double *unused, double *out_450177087820241572) {
   out_450177087820241572[0] = state[4];
   out_450177087820241572[1] = state[5];
}
void H_24(double *state, double *unused, double *out_6784004382849502969) {
   out_6784004382849502969[0] = 0;
   out_6784004382849502969[1] = 0;
   out_6784004382849502969[2] = 0;
   out_6784004382849502969[3] = 0;
   out_6784004382849502969[4] = 1;
   out_6784004382849502969[5] = 0;
   out_6784004382849502969[6] = 0;
   out_6784004382849502969[7] = 0;
   out_6784004382849502969[8] = 0;
   out_6784004382849502969[9] = 0;
   out_6784004382849502969[10] = 0;
   out_6784004382849502969[11] = 0;
   out_6784004382849502969[12] = 0;
   out_6784004382849502969[13] = 0;
   out_6784004382849502969[14] = 1;
   out_6784004382849502969[15] = 0;
   out_6784004382849502969[16] = 0;
   out_6784004382849502969[17] = 0;
}
void h_30(double *state, double *unused, double *out_9043571034882076386) {
   out_9043571034882076386[0] = state[4];
}
void H_30(double *state, double *unused, double *out_4397897405386290986) {
   out_4397897405386290986[0] = 0;
   out_4397897405386290986[1] = 0;
   out_4397897405386290986[2] = 0;
   out_4397897405386290986[3] = 0;
   out_4397897405386290986[4] = 1;
   out_4397897405386290986[5] = 0;
   out_4397897405386290986[6] = 0;
   out_4397897405386290986[7] = 0;
   out_4397897405386290986[8] = 0;
}
void h_26(double *state, double *unused, double *out_4401499771166629585) {
   out_4401499771166629585[0] = state[7];
}
void H_26(double *state, double *unused, double *out_7789010390941955779) {
   out_7789010390941955779[0] = 0;
   out_7789010390941955779[1] = 0;
   out_7789010390941955779[2] = 0;
   out_7789010390941955779[3] = 0;
   out_7789010390941955779[4] = 0;
   out_7789010390941955779[5] = 0;
   out_7789010390941955779[6] = 0;
   out_7789010390941955779[7] = 1;
   out_7789010390941955779[8] = 0;
}
void h_27(double *state, double *unused, double *out_3819966847812604518) {
   out_3819966847812604518[0] = state[3];
}
void H_27(double *state, double *unused, double *out_6572660717186715897) {
   out_6572660717186715897[0] = 0;
   out_6572660717186715897[1] = 0;
   out_6572660717186715897[2] = 0;
   out_6572660717186715897[3] = 1;
   out_6572660717186715897[4] = 0;
   out_6572660717186715897[5] = 0;
   out_6572660717186715897[6] = 0;
   out_6572660717186715897[7] = 0;
   out_6572660717186715897[8] = 0;
}
void h_29(double *state, double *unused, double *out_3147348316868427976) {
   out_3147348316868427976[0] = state[1];
}
void H_29(double *state, double *unused, double *out_3887666061071898802) {
   out_3887666061071898802[0] = 0;
   out_3887666061071898802[1] = 1;
   out_3887666061071898802[2] = 0;
   out_3887666061071898802[3] = 0;
   out_3887666061071898802[4] = 0;
   out_3887666061071898802[5] = 0;
   out_3887666061071898802[6] = 0;
   out_3887666061071898802[7] = 0;
   out_3887666061071898802[8] = 0;
}
void h_28(double *state, double *unused, double *out_8227735708483097952) {
   out_8227735708483097952[0] = state[0];
}
void H_28(double *state, double *unused, double *out_8970065078141429376) {
   out_8970065078141429376[0] = 1;
   out_8970065078141429376[1] = 0;
   out_8970065078141429376[2] = 0;
   out_8970065078141429376[3] = 0;
   out_8970065078141429376[4] = 0;
   out_8970065078141429376[5] = 0;
   out_8970065078141429376[6] = 0;
   out_8970065078141429376[7] = 0;
   out_8970065078141429376[8] = 0;
}
void h_31(double *state, double *unused, double *out_6367141668186816434) {
   out_6367141668186816434[0] = state[8];
}
void H_31(double *state, double *unused, double *out_7162802288708604303) {
   out_7162802288708604303[0] = 0;
   out_7162802288708604303[1] = 0;
   out_7162802288708604303[2] = 0;
   out_7162802288708604303[3] = 0;
   out_7162802288708604303[4] = 0;
   out_7162802288708604303[5] = 0;
   out_7162802288708604303[6] = 0;
   out_7162802288708604303[7] = 0;
   out_7162802288708604303[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_7082197508459355294) {
  err_fun(nom_x, delta_x, out_7082197508459355294);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7553533010452763308) {
  inv_err_fun(nom_x, true_x, out_7553533010452763308);
}
void car_H_mod_fun(double *state, double *out_2748508627650133070) {
  H_mod_fun(state, out_2748508627650133070);
}
void car_f_fun(double *state, double dt, double *out_1858329490310860980) {
  f_fun(state,  dt, out_1858329490310860980);
}
void car_F_fun(double *state, double dt, double *out_8266273866488598058) {
  F_fun(state,  dt, out_8266273866488598058);
}
void car_h_25(double *state, double *unused, double *out_8442152877268712832) {
  h_25(state, unused, out_8442152877268712832);
}
void car_H_25(double *state, double *unused, double *out_6916230363893539613) {
  H_25(state, unused, out_6916230363893539613);
}
void car_h_24(double *state, double *unused, double *out_450177087820241572) {
  h_24(state, unused, out_450177087820241572);
}
void car_H_24(double *state, double *unused, double *out_6784004382849502969) {
  H_24(state, unused, out_6784004382849502969);
}
void car_h_30(double *state, double *unused, double *out_9043571034882076386) {
  h_30(state, unused, out_9043571034882076386);
}
void car_H_30(double *state, double *unused, double *out_4397897405386290986) {
  H_30(state, unused, out_4397897405386290986);
}
void car_h_26(double *state, double *unused, double *out_4401499771166629585) {
  h_26(state, unused, out_4401499771166629585);
}
void car_H_26(double *state, double *unused, double *out_7789010390941955779) {
  H_26(state, unused, out_7789010390941955779);
}
void car_h_27(double *state, double *unused, double *out_3819966847812604518) {
  h_27(state, unused, out_3819966847812604518);
}
void car_H_27(double *state, double *unused, double *out_6572660717186715897) {
  H_27(state, unused, out_6572660717186715897);
}
void car_h_29(double *state, double *unused, double *out_3147348316868427976) {
  h_29(state, unused, out_3147348316868427976);
}
void car_H_29(double *state, double *unused, double *out_3887666061071898802) {
  H_29(state, unused, out_3887666061071898802);
}
void car_h_28(double *state, double *unused, double *out_8227735708483097952) {
  h_28(state, unused, out_8227735708483097952);
}
void car_H_28(double *state, double *unused, double *out_8970065078141429376) {
  H_28(state, unused, out_8970065078141429376);
}
void car_h_31(double *state, double *unused, double *out_6367141668186816434) {
  h_31(state, unused, out_6367141668186816434);
}
void car_H_31(double *state, double *unused, double *out_7162802288708604303) {
  H_31(state, unused, out_7162802288708604303);
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
