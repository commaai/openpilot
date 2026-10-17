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
void err_fun(double *nom_x, double *delta_x, double *out_8879023178729925980) {
   out_8879023178729925980[0] = delta_x[0] + nom_x[0];
   out_8879023178729925980[1] = delta_x[1] + nom_x[1];
   out_8879023178729925980[2] = delta_x[2] + nom_x[2];
   out_8879023178729925980[3] = delta_x[3] + nom_x[3];
   out_8879023178729925980[4] = delta_x[4] + nom_x[4];
   out_8879023178729925980[5] = delta_x[5] + nom_x[5];
   out_8879023178729925980[6] = delta_x[6] + nom_x[6];
   out_8879023178729925980[7] = delta_x[7] + nom_x[7];
   out_8879023178729925980[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_917364609278737637) {
   out_917364609278737637[0] = -nom_x[0] + true_x[0];
   out_917364609278737637[1] = -nom_x[1] + true_x[1];
   out_917364609278737637[2] = -nom_x[2] + true_x[2];
   out_917364609278737637[3] = -nom_x[3] + true_x[3];
   out_917364609278737637[4] = -nom_x[4] + true_x[4];
   out_917364609278737637[5] = -nom_x[5] + true_x[5];
   out_917364609278737637[6] = -nom_x[6] + true_x[6];
   out_917364609278737637[7] = -nom_x[7] + true_x[7];
   out_917364609278737637[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_6731364818185773625) {
   out_6731364818185773625[0] = 1.0;
   out_6731364818185773625[1] = 0.0;
   out_6731364818185773625[2] = 0.0;
   out_6731364818185773625[3] = 0.0;
   out_6731364818185773625[4] = 0.0;
   out_6731364818185773625[5] = 0.0;
   out_6731364818185773625[6] = 0.0;
   out_6731364818185773625[7] = 0.0;
   out_6731364818185773625[8] = 0.0;
   out_6731364818185773625[9] = 0.0;
   out_6731364818185773625[10] = 1.0;
   out_6731364818185773625[11] = 0.0;
   out_6731364818185773625[12] = 0.0;
   out_6731364818185773625[13] = 0.0;
   out_6731364818185773625[14] = 0.0;
   out_6731364818185773625[15] = 0.0;
   out_6731364818185773625[16] = 0.0;
   out_6731364818185773625[17] = 0.0;
   out_6731364818185773625[18] = 0.0;
   out_6731364818185773625[19] = 0.0;
   out_6731364818185773625[20] = 1.0;
   out_6731364818185773625[21] = 0.0;
   out_6731364818185773625[22] = 0.0;
   out_6731364818185773625[23] = 0.0;
   out_6731364818185773625[24] = 0.0;
   out_6731364818185773625[25] = 0.0;
   out_6731364818185773625[26] = 0.0;
   out_6731364818185773625[27] = 0.0;
   out_6731364818185773625[28] = 0.0;
   out_6731364818185773625[29] = 0.0;
   out_6731364818185773625[30] = 1.0;
   out_6731364818185773625[31] = 0.0;
   out_6731364818185773625[32] = 0.0;
   out_6731364818185773625[33] = 0.0;
   out_6731364818185773625[34] = 0.0;
   out_6731364818185773625[35] = 0.0;
   out_6731364818185773625[36] = 0.0;
   out_6731364818185773625[37] = 0.0;
   out_6731364818185773625[38] = 0.0;
   out_6731364818185773625[39] = 0.0;
   out_6731364818185773625[40] = 1.0;
   out_6731364818185773625[41] = 0.0;
   out_6731364818185773625[42] = 0.0;
   out_6731364818185773625[43] = 0.0;
   out_6731364818185773625[44] = 0.0;
   out_6731364818185773625[45] = 0.0;
   out_6731364818185773625[46] = 0.0;
   out_6731364818185773625[47] = 0.0;
   out_6731364818185773625[48] = 0.0;
   out_6731364818185773625[49] = 0.0;
   out_6731364818185773625[50] = 1.0;
   out_6731364818185773625[51] = 0.0;
   out_6731364818185773625[52] = 0.0;
   out_6731364818185773625[53] = 0.0;
   out_6731364818185773625[54] = 0.0;
   out_6731364818185773625[55] = 0.0;
   out_6731364818185773625[56] = 0.0;
   out_6731364818185773625[57] = 0.0;
   out_6731364818185773625[58] = 0.0;
   out_6731364818185773625[59] = 0.0;
   out_6731364818185773625[60] = 1.0;
   out_6731364818185773625[61] = 0.0;
   out_6731364818185773625[62] = 0.0;
   out_6731364818185773625[63] = 0.0;
   out_6731364818185773625[64] = 0.0;
   out_6731364818185773625[65] = 0.0;
   out_6731364818185773625[66] = 0.0;
   out_6731364818185773625[67] = 0.0;
   out_6731364818185773625[68] = 0.0;
   out_6731364818185773625[69] = 0.0;
   out_6731364818185773625[70] = 1.0;
   out_6731364818185773625[71] = 0.0;
   out_6731364818185773625[72] = 0.0;
   out_6731364818185773625[73] = 0.0;
   out_6731364818185773625[74] = 0.0;
   out_6731364818185773625[75] = 0.0;
   out_6731364818185773625[76] = 0.0;
   out_6731364818185773625[77] = 0.0;
   out_6731364818185773625[78] = 0.0;
   out_6731364818185773625[79] = 0.0;
   out_6731364818185773625[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_4909689618608652368) {
   out_4909689618608652368[0] = state[0];
   out_4909689618608652368[1] = state[1];
   out_4909689618608652368[2] = state[2];
   out_4909689618608652368[3] = state[3];
   out_4909689618608652368[4] = state[4];
   out_4909689618608652368[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_4909689618608652368[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_4909689618608652368[7] = state[7];
   out_4909689618608652368[8] = state[8];
}
void F_fun(double *state, double dt, double *out_415125029152466339) {
   out_415125029152466339[0] = 1;
   out_415125029152466339[1] = 0;
   out_415125029152466339[2] = 0;
   out_415125029152466339[3] = 0;
   out_415125029152466339[4] = 0;
   out_415125029152466339[5] = 0;
   out_415125029152466339[6] = 0;
   out_415125029152466339[7] = 0;
   out_415125029152466339[8] = 0;
   out_415125029152466339[9] = 0;
   out_415125029152466339[10] = 1;
   out_415125029152466339[11] = 0;
   out_415125029152466339[12] = 0;
   out_415125029152466339[13] = 0;
   out_415125029152466339[14] = 0;
   out_415125029152466339[15] = 0;
   out_415125029152466339[16] = 0;
   out_415125029152466339[17] = 0;
   out_415125029152466339[18] = 0;
   out_415125029152466339[19] = 0;
   out_415125029152466339[20] = 1;
   out_415125029152466339[21] = 0;
   out_415125029152466339[22] = 0;
   out_415125029152466339[23] = 0;
   out_415125029152466339[24] = 0;
   out_415125029152466339[25] = 0;
   out_415125029152466339[26] = 0;
   out_415125029152466339[27] = 0;
   out_415125029152466339[28] = 0;
   out_415125029152466339[29] = 0;
   out_415125029152466339[30] = 1;
   out_415125029152466339[31] = 0;
   out_415125029152466339[32] = 0;
   out_415125029152466339[33] = 0;
   out_415125029152466339[34] = 0;
   out_415125029152466339[35] = 0;
   out_415125029152466339[36] = 0;
   out_415125029152466339[37] = 0;
   out_415125029152466339[38] = 0;
   out_415125029152466339[39] = 0;
   out_415125029152466339[40] = 1;
   out_415125029152466339[41] = 0;
   out_415125029152466339[42] = 0;
   out_415125029152466339[43] = 0;
   out_415125029152466339[44] = 0;
   out_415125029152466339[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_415125029152466339[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_415125029152466339[47] = -dt*stiffness_front*state[0]/state[1];
   out_415125029152466339[48] = -dt*stiffness_front*state[0]/state[1];
   out_415125029152466339[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_415125029152466339[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_415125029152466339[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_415125029152466339[52] = dt*stiffness_front*state[0]/state[1];
   out_415125029152466339[53] = -9.8100000000000005*dt;
   out_415125029152466339[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_415125029152466339[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_415125029152466339[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_415125029152466339[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_415125029152466339[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_415125029152466339[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_415125029152466339[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_415125029152466339[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_415125029152466339[62] = 0;
   out_415125029152466339[63] = 0;
   out_415125029152466339[64] = 0;
   out_415125029152466339[65] = 0;
   out_415125029152466339[66] = 0;
   out_415125029152466339[67] = 0;
   out_415125029152466339[68] = 0;
   out_415125029152466339[69] = 0;
   out_415125029152466339[70] = 1;
   out_415125029152466339[71] = 0;
   out_415125029152466339[72] = 0;
   out_415125029152466339[73] = 0;
   out_415125029152466339[74] = 0;
   out_415125029152466339[75] = 0;
   out_415125029152466339[76] = 0;
   out_415125029152466339[77] = 0;
   out_415125029152466339[78] = 0;
   out_415125029152466339[79] = 0;
   out_415125029152466339[80] = 1;
}
void h_25(double *state, double *unused, double *out_4795770764444923556) {
   out_4795770764444923556[0] = state[6];
}
void H_25(double *state, double *unused, double *out_2522191029611540754) {
   out_2522191029611540754[0] = 0;
   out_2522191029611540754[1] = 0;
   out_2522191029611540754[2] = 0;
   out_2522191029611540754[3] = 0;
   out_2522191029611540754[4] = 0;
   out_2522191029611540754[5] = 0;
   out_2522191029611540754[6] = 1;
   out_2522191029611540754[7] = 0;
   out_2522191029611540754[8] = 0;
}
void h_24(double *state, double *unused, double *out_6599817667004600482) {
   out_6599817667004600482[0] = state[4];
   out_6599817667004600482[1] = state[5];
}
void H_24(double *state, double *unused, double *out_6252701378200630140) {
   out_6252701378200630140[0] = 0;
   out_6252701378200630140[1] = 0;
   out_6252701378200630140[2] = 0;
   out_6252701378200630140[3] = 0;
   out_6252701378200630140[4] = 1;
   out_6252701378200630140[5] = 0;
   out_6252701378200630140[6] = 0;
   out_6252701378200630140[7] = 0;
   out_6252701378200630140[8] = 0;
   out_6252701378200630140[9] = 0;
   out_6252701378200630140[10] = 0;
   out_6252701378200630140[11] = 0;
   out_6252701378200630140[12] = 0;
   out_6252701378200630140[13] = 0;
   out_6252701378200630140[14] = 1;
   out_6252701378200630140[15] = 0;
   out_6252701378200630140[16] = 0;
   out_6252701378200630140[17] = 0;
}
void h_30(double *state, double *unused, double *out_6887458760493917913) {
   out_6887458760493917913[0] = state[4];
}
void H_30(double *state, double *unused, double *out_2392852082468300684) {
   out_2392852082468300684[0] = 0;
   out_2392852082468300684[1] = 0;
   out_2392852082468300684[2] = 0;
   out_2392852082468300684[3] = 0;
   out_2392852082468300684[4] = 1;
   out_2392852082468300684[5] = 0;
   out_2392852082468300684[6] = 0;
   out_2392852082468300684[7] = 0;
   out_2392852082468300684[8] = 0;
}
void h_26(double *state, double *unused, double *out_5943948907592255478) {
   out_5943948907592255478[0] = state[7];
}
void H_26(double *state, double *unused, double *out_1219312289262515470) {
   out_1219312289262515470[0] = 0;
   out_1219312289262515470[1] = 0;
   out_1219312289262515470[2] = 0;
   out_1219312289262515470[3] = 0;
   out_1219312289262515470[4] = 0;
   out_1219312289262515470[5] = 0;
   out_1219312289262515470[6] = 0;
   out_1219312289262515470[7] = 1;
   out_1219312289262515470[8] = 0;
}
void h_27(double *state, double *unused, double *out_8679685930122430163) {
   out_8679685930122430163[0] = state[3];
}
void H_27(double *state, double *unused, double *out_218088770667875773) {
   out_218088770667875773[0] = 0;
   out_218088770667875773[1] = 0;
   out_218088770667875773[2] = 0;
   out_218088770667875773[3] = 1;
   out_218088770667875773[4] = 0;
   out_218088770667875773[5] = 0;
   out_218088770667875773[6] = 0;
   out_218088770667875773[7] = 0;
   out_218088770667875773[8] = 0;
}
void h_29(double *state, double *unused, double *out_3153628493571407050) {
   out_3153628493571407050[0] = state[1];
}
void H_29(double *state, double *unused, double *out_2903083426782692868) {
   out_2903083426782692868[0] = 0;
   out_2903083426782692868[1] = 1;
   out_2903083426782692868[2] = 0;
   out_2903083426782692868[3] = 0;
   out_2903083426782692868[4] = 0;
   out_2903083426782692868[5] = 0;
   out_2903083426782692868[6] = 0;
   out_2903083426782692868[7] = 0;
   out_2903083426782692868[8] = 0;
}
void h_28(double *state, double *unused, double *out_8009781937011597724) {
   out_8009781937011597724[0] = state[0];
}
void H_28(double *state, double *unused, double *out_4866713698348019119) {
   out_4866713698348019119[0] = 1;
   out_4866713698348019119[1] = 0;
   out_4866713698348019119[2] = 0;
   out_4866713698348019119[3] = 0;
   out_4866713698348019119[4] = 0;
   out_4866713698348019119[5] = 0;
   out_4866713698348019119[6] = 0;
   out_4866713698348019119[7] = 0;
   out_4866713698348019119[8] = 0;
}
void h_31(double *state, double *unused, double *out_5128644055090467685) {
   out_5128644055090467685[0] = state[8];
}
void H_31(double *state, double *unused, double *out_2552836991488501182) {
   out_2552836991488501182[0] = 0;
   out_2552836991488501182[1] = 0;
   out_2552836991488501182[2] = 0;
   out_2552836991488501182[3] = 0;
   out_2552836991488501182[4] = 0;
   out_2552836991488501182[5] = 0;
   out_2552836991488501182[6] = 0;
   out_2552836991488501182[7] = 0;
   out_2552836991488501182[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_8879023178729925980) {
  err_fun(nom_x, delta_x, out_8879023178729925980);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_917364609278737637) {
  inv_err_fun(nom_x, true_x, out_917364609278737637);
}
void car_H_mod_fun(double *state, double *out_6731364818185773625) {
  H_mod_fun(state, out_6731364818185773625);
}
void car_f_fun(double *state, double dt, double *out_4909689618608652368) {
  f_fun(state,  dt, out_4909689618608652368);
}
void car_F_fun(double *state, double dt, double *out_415125029152466339) {
  F_fun(state,  dt, out_415125029152466339);
}
void car_h_25(double *state, double *unused, double *out_4795770764444923556) {
  h_25(state, unused, out_4795770764444923556);
}
void car_H_25(double *state, double *unused, double *out_2522191029611540754) {
  H_25(state, unused, out_2522191029611540754);
}
void car_h_24(double *state, double *unused, double *out_6599817667004600482) {
  h_24(state, unused, out_6599817667004600482);
}
void car_H_24(double *state, double *unused, double *out_6252701378200630140) {
  H_24(state, unused, out_6252701378200630140);
}
void car_h_30(double *state, double *unused, double *out_6887458760493917913) {
  h_30(state, unused, out_6887458760493917913);
}
void car_H_30(double *state, double *unused, double *out_2392852082468300684) {
  H_30(state, unused, out_2392852082468300684);
}
void car_h_26(double *state, double *unused, double *out_5943948907592255478) {
  h_26(state, unused, out_5943948907592255478);
}
void car_H_26(double *state, double *unused, double *out_1219312289262515470) {
  H_26(state, unused, out_1219312289262515470);
}
void car_h_27(double *state, double *unused, double *out_8679685930122430163) {
  h_27(state, unused, out_8679685930122430163);
}
void car_H_27(double *state, double *unused, double *out_218088770667875773) {
  H_27(state, unused, out_218088770667875773);
}
void car_h_29(double *state, double *unused, double *out_3153628493571407050) {
  h_29(state, unused, out_3153628493571407050);
}
void car_H_29(double *state, double *unused, double *out_2903083426782692868) {
  H_29(state, unused, out_2903083426782692868);
}
void car_h_28(double *state, double *unused, double *out_8009781937011597724) {
  h_28(state, unused, out_8009781937011597724);
}
void car_H_28(double *state, double *unused, double *out_4866713698348019119) {
  H_28(state, unused, out_4866713698348019119);
}
void car_h_31(double *state, double *unused, double *out_5128644055090467685) {
  h_31(state, unused, out_5128644055090467685);
}
void car_H_31(double *state, double *unused, double *out_2552836991488501182) {
  H_31(state, unused, out_2552836991488501182);
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
