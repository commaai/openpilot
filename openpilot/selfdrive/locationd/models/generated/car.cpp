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
void err_fun(double *nom_x, double *delta_x, double *out_2996852071044234202) {
   out_2996852071044234202[0] = delta_x[0] + nom_x[0];
   out_2996852071044234202[1] = delta_x[1] + nom_x[1];
   out_2996852071044234202[2] = delta_x[2] + nom_x[2];
   out_2996852071044234202[3] = delta_x[3] + nom_x[3];
   out_2996852071044234202[4] = delta_x[4] + nom_x[4];
   out_2996852071044234202[5] = delta_x[5] + nom_x[5];
   out_2996852071044234202[6] = delta_x[6] + nom_x[6];
   out_2996852071044234202[7] = delta_x[7] + nom_x[7];
   out_2996852071044234202[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_5428601996215229963) {
   out_5428601996215229963[0] = -nom_x[0] + true_x[0];
   out_5428601996215229963[1] = -nom_x[1] + true_x[1];
   out_5428601996215229963[2] = -nom_x[2] + true_x[2];
   out_5428601996215229963[3] = -nom_x[3] + true_x[3];
   out_5428601996215229963[4] = -nom_x[4] + true_x[4];
   out_5428601996215229963[5] = -nom_x[5] + true_x[5];
   out_5428601996215229963[6] = -nom_x[6] + true_x[6];
   out_5428601996215229963[7] = -nom_x[7] + true_x[7];
   out_5428601996215229963[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_6840182513517023115) {
   out_6840182513517023115[0] = 1.0;
   out_6840182513517023115[1] = 0.0;
   out_6840182513517023115[2] = 0.0;
   out_6840182513517023115[3] = 0.0;
   out_6840182513517023115[4] = 0.0;
   out_6840182513517023115[5] = 0.0;
   out_6840182513517023115[6] = 0.0;
   out_6840182513517023115[7] = 0.0;
   out_6840182513517023115[8] = 0.0;
   out_6840182513517023115[9] = 0.0;
   out_6840182513517023115[10] = 1.0;
   out_6840182513517023115[11] = 0.0;
   out_6840182513517023115[12] = 0.0;
   out_6840182513517023115[13] = 0.0;
   out_6840182513517023115[14] = 0.0;
   out_6840182513517023115[15] = 0.0;
   out_6840182513517023115[16] = 0.0;
   out_6840182513517023115[17] = 0.0;
   out_6840182513517023115[18] = 0.0;
   out_6840182513517023115[19] = 0.0;
   out_6840182513517023115[20] = 1.0;
   out_6840182513517023115[21] = 0.0;
   out_6840182513517023115[22] = 0.0;
   out_6840182513517023115[23] = 0.0;
   out_6840182513517023115[24] = 0.0;
   out_6840182513517023115[25] = 0.0;
   out_6840182513517023115[26] = 0.0;
   out_6840182513517023115[27] = 0.0;
   out_6840182513517023115[28] = 0.0;
   out_6840182513517023115[29] = 0.0;
   out_6840182513517023115[30] = 1.0;
   out_6840182513517023115[31] = 0.0;
   out_6840182513517023115[32] = 0.0;
   out_6840182513517023115[33] = 0.0;
   out_6840182513517023115[34] = 0.0;
   out_6840182513517023115[35] = 0.0;
   out_6840182513517023115[36] = 0.0;
   out_6840182513517023115[37] = 0.0;
   out_6840182513517023115[38] = 0.0;
   out_6840182513517023115[39] = 0.0;
   out_6840182513517023115[40] = 1.0;
   out_6840182513517023115[41] = 0.0;
   out_6840182513517023115[42] = 0.0;
   out_6840182513517023115[43] = 0.0;
   out_6840182513517023115[44] = 0.0;
   out_6840182513517023115[45] = 0.0;
   out_6840182513517023115[46] = 0.0;
   out_6840182513517023115[47] = 0.0;
   out_6840182513517023115[48] = 0.0;
   out_6840182513517023115[49] = 0.0;
   out_6840182513517023115[50] = 1.0;
   out_6840182513517023115[51] = 0.0;
   out_6840182513517023115[52] = 0.0;
   out_6840182513517023115[53] = 0.0;
   out_6840182513517023115[54] = 0.0;
   out_6840182513517023115[55] = 0.0;
   out_6840182513517023115[56] = 0.0;
   out_6840182513517023115[57] = 0.0;
   out_6840182513517023115[58] = 0.0;
   out_6840182513517023115[59] = 0.0;
   out_6840182513517023115[60] = 1.0;
   out_6840182513517023115[61] = 0.0;
   out_6840182513517023115[62] = 0.0;
   out_6840182513517023115[63] = 0.0;
   out_6840182513517023115[64] = 0.0;
   out_6840182513517023115[65] = 0.0;
   out_6840182513517023115[66] = 0.0;
   out_6840182513517023115[67] = 0.0;
   out_6840182513517023115[68] = 0.0;
   out_6840182513517023115[69] = 0.0;
   out_6840182513517023115[70] = 1.0;
   out_6840182513517023115[71] = 0.0;
   out_6840182513517023115[72] = 0.0;
   out_6840182513517023115[73] = 0.0;
   out_6840182513517023115[74] = 0.0;
   out_6840182513517023115[75] = 0.0;
   out_6840182513517023115[76] = 0.0;
   out_6840182513517023115[77] = 0.0;
   out_6840182513517023115[78] = 0.0;
   out_6840182513517023115[79] = 0.0;
   out_6840182513517023115[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_4503671108966703887) {
   out_4503671108966703887[0] = state[0];
   out_4503671108966703887[1] = state[1];
   out_4503671108966703887[2] = state[2];
   out_4503671108966703887[3] = state[3];
   out_4503671108966703887[4] = state[4];
   out_4503671108966703887[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_4503671108966703887[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_4503671108966703887[7] = state[7];
   out_4503671108966703887[8] = state[8];
}
void F_fun(double *state, double dt, double *out_4770600575889623320) {
   out_4770600575889623320[0] = 1;
   out_4770600575889623320[1] = 0;
   out_4770600575889623320[2] = 0;
   out_4770600575889623320[3] = 0;
   out_4770600575889623320[4] = 0;
   out_4770600575889623320[5] = 0;
   out_4770600575889623320[6] = 0;
   out_4770600575889623320[7] = 0;
   out_4770600575889623320[8] = 0;
   out_4770600575889623320[9] = 0;
   out_4770600575889623320[10] = 1;
   out_4770600575889623320[11] = 0;
   out_4770600575889623320[12] = 0;
   out_4770600575889623320[13] = 0;
   out_4770600575889623320[14] = 0;
   out_4770600575889623320[15] = 0;
   out_4770600575889623320[16] = 0;
   out_4770600575889623320[17] = 0;
   out_4770600575889623320[18] = 0;
   out_4770600575889623320[19] = 0;
   out_4770600575889623320[20] = 1;
   out_4770600575889623320[21] = 0;
   out_4770600575889623320[22] = 0;
   out_4770600575889623320[23] = 0;
   out_4770600575889623320[24] = 0;
   out_4770600575889623320[25] = 0;
   out_4770600575889623320[26] = 0;
   out_4770600575889623320[27] = 0;
   out_4770600575889623320[28] = 0;
   out_4770600575889623320[29] = 0;
   out_4770600575889623320[30] = 1;
   out_4770600575889623320[31] = 0;
   out_4770600575889623320[32] = 0;
   out_4770600575889623320[33] = 0;
   out_4770600575889623320[34] = 0;
   out_4770600575889623320[35] = 0;
   out_4770600575889623320[36] = 0;
   out_4770600575889623320[37] = 0;
   out_4770600575889623320[38] = 0;
   out_4770600575889623320[39] = 0;
   out_4770600575889623320[40] = 1;
   out_4770600575889623320[41] = 0;
   out_4770600575889623320[42] = 0;
   out_4770600575889623320[43] = 0;
   out_4770600575889623320[44] = 0;
   out_4770600575889623320[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_4770600575889623320[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_4770600575889623320[47] = -dt*stiffness_front*state[0]/state[1];
   out_4770600575889623320[48] = -dt*stiffness_front*state[0]/state[1];
   out_4770600575889623320[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_4770600575889623320[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_4770600575889623320[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_4770600575889623320[52] = dt*stiffness_front*state[0]/state[1];
   out_4770600575889623320[53] = -9.8100000000000005*dt;
   out_4770600575889623320[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_4770600575889623320[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_4770600575889623320[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4770600575889623320[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4770600575889623320[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_4770600575889623320[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_4770600575889623320[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_4770600575889623320[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4770600575889623320[62] = 0;
   out_4770600575889623320[63] = 0;
   out_4770600575889623320[64] = 0;
   out_4770600575889623320[65] = 0;
   out_4770600575889623320[66] = 0;
   out_4770600575889623320[67] = 0;
   out_4770600575889623320[68] = 0;
   out_4770600575889623320[69] = 0;
   out_4770600575889623320[70] = 1;
   out_4770600575889623320[71] = 0;
   out_4770600575889623320[72] = 0;
   out_4770600575889623320[73] = 0;
   out_4770600575889623320[74] = 0;
   out_4770600575889623320[75] = 0;
   out_4770600575889623320[76] = 0;
   out_4770600575889623320[77] = 0;
   out_4770600575889623320[78] = 0;
   out_4770600575889623320[79] = 0;
   out_4770600575889623320[80] = 1;
}
void h_25(double *state, double *unused, double *out_5459418218260901103) {
   out_5459418218260901103[0] = state[6];
}
void H_25(double *state, double *unused, double *out_1936490688332358275) {
   out_1936490688332358275[0] = 0;
   out_1936490688332358275[1] = 0;
   out_1936490688332358275[2] = 0;
   out_1936490688332358275[3] = 0;
   out_1936490688332358275[4] = 0;
   out_1936490688332358275[5] = 0;
   out_1936490688332358275[6] = 1;
   out_1936490688332358275[7] = 0;
   out_1936490688332358275[8] = 0;
}
void h_24(double *state, double *unused, double *out_4053235228616768809) {
   out_4053235228616768809[0] = state[4];
   out_4053235228616768809[1] = state[5];
}
void H_24(double *state, double *unused, double *out_2333135992686217770) {
   out_2333135992686217770[0] = 0;
   out_2333135992686217770[1] = 0;
   out_2333135992686217770[2] = 0;
   out_2333135992686217770[3] = 0;
   out_2333135992686217770[4] = 1;
   out_2333135992686217770[5] = 0;
   out_2333135992686217770[6] = 0;
   out_2333135992686217770[7] = 0;
   out_2333135992686217770[8] = 0;
   out_2333135992686217770[9] = 0;
   out_2333135992686217770[10] = 0;
   out_2333135992686217770[11] = 0;
   out_2333135992686217770[12] = 0;
   out_2333135992686217770[13] = 0;
   out_2333135992686217770[14] = 1;
   out_2333135992686217770[15] = 0;
   out_2333135992686217770[16] = 0;
   out_2333135992686217770[17] = 0;
}
void h_30(double *state, double *unused, double *out_5302231549909771958) {
   out_5302231549909771958[0] = state[4];
}
void H_30(double *state, double *unused, double *out_4980199653159258480) {
   out_4980199653159258480[0] = 0;
   out_4980199653159258480[1] = 0;
   out_4980199653159258480[2] = 0;
   out_4980199653159258480[3] = 0;
   out_4980199653159258480[4] = 1;
   out_4980199653159258480[5] = 0;
   out_4980199653159258480[6] = 0;
   out_4980199653159258480[7] = 0;
   out_4980199653159258480[8] = 0;
}
void h_26(double *state, double *unused, double *out_4698915665233294441) {
   out_4698915665233294441[0] = state[7];
}
void H_26(double *state, double *unused, double *out_5677994007206414499) {
   out_5677994007206414499[0] = 0;
   out_5677994007206414499[1] = 0;
   out_5677994007206414499[2] = 0;
   out_5677994007206414499[3] = 0;
   out_5677994007206414499[4] = 0;
   out_5677994007206414499[5] = 0;
   out_5677994007206414499[6] = 0;
   out_5677994007206414499[7] = 1;
   out_5677994007206414499[8] = 0;
}
void h_27(double *state, double *unused, double *out_8561964940150418980) {
   out_8561964940150418980[0] = state[3];
}
void H_27(double *state, double *unused, double *out_2805436341358833569) {
   out_2805436341358833569[0] = 0;
   out_2805436341358833569[1] = 0;
   out_2805436341358833569[2] = 0;
   out_2805436341358833569[3] = 1;
   out_2805436341358833569[4] = 0;
   out_2805436341358833569[5] = 0;
   out_2805436341358833569[6] = 0;
   out_2805436341358833569[7] = 0;
   out_2805436341358833569[8] = 0;
}
void h_29(double *state, double *unused, double *out_4358721697008109523) {
   out_4358721697008109523[0] = state[1];
}
void H_29(double *state, double *unused, double *out_1092073614489282536) {
   out_1092073614489282536[0] = 0;
   out_1092073614489282536[1] = 1;
   out_1092073614489282536[2] = 0;
   out_1092073614489282536[3] = 0;
   out_1092073614489282536[4] = 0;
   out_1092073614489282536[5] = 0;
   out_1092073614489282536[6] = 0;
   out_1092073614489282536[7] = 0;
   out_1092073614489282536[8] = 0;
}
void h_28(double *state, double *unused, double *out_1479122313914905983) {
   out_1479122313914905983[0] = state[0];
}
void H_28(double *state, double *unused, double *out_3990325402580248038) {
   out_3990325402580248038[0] = 1;
   out_3990325402580248038[1] = 0;
   out_3990325402580248038[2] = 0;
   out_3990325402580248038[3] = 0;
   out_3990325402580248038[4] = 0;
   out_3990325402580248038[5] = 0;
   out_3990325402580248038[6] = 0;
   out_3990325402580248038[7] = 0;
   out_3990325402580248038[8] = 0;
}
void h_31(double *state, double *unused, double *out_1402015568006224056) {
   out_1402015568006224056[0] = state[8];
}
void H_31(double *state, double *unused, double *out_1905844726455397847) {
   out_1905844726455397847[0] = 0;
   out_1905844726455397847[1] = 0;
   out_1905844726455397847[2] = 0;
   out_1905844726455397847[3] = 0;
   out_1905844726455397847[4] = 0;
   out_1905844726455397847[5] = 0;
   out_1905844726455397847[6] = 0;
   out_1905844726455397847[7] = 0;
   out_1905844726455397847[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_2996852071044234202) {
  err_fun(nom_x, delta_x, out_2996852071044234202);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5428601996215229963) {
  inv_err_fun(nom_x, true_x, out_5428601996215229963);
}
void car_H_mod_fun(double *state, double *out_6840182513517023115) {
  H_mod_fun(state, out_6840182513517023115);
}
void car_f_fun(double *state, double dt, double *out_4503671108966703887) {
  f_fun(state,  dt, out_4503671108966703887);
}
void car_F_fun(double *state, double dt, double *out_4770600575889623320) {
  F_fun(state,  dt, out_4770600575889623320);
}
void car_h_25(double *state, double *unused, double *out_5459418218260901103) {
  h_25(state, unused, out_5459418218260901103);
}
void car_H_25(double *state, double *unused, double *out_1936490688332358275) {
  H_25(state, unused, out_1936490688332358275);
}
void car_h_24(double *state, double *unused, double *out_4053235228616768809) {
  h_24(state, unused, out_4053235228616768809);
}
void car_H_24(double *state, double *unused, double *out_2333135992686217770) {
  H_24(state, unused, out_2333135992686217770);
}
void car_h_30(double *state, double *unused, double *out_5302231549909771958) {
  h_30(state, unused, out_5302231549909771958);
}
void car_H_30(double *state, double *unused, double *out_4980199653159258480) {
  H_30(state, unused, out_4980199653159258480);
}
void car_h_26(double *state, double *unused, double *out_4698915665233294441) {
  h_26(state, unused, out_4698915665233294441);
}
void car_H_26(double *state, double *unused, double *out_5677994007206414499) {
  H_26(state, unused, out_5677994007206414499);
}
void car_h_27(double *state, double *unused, double *out_8561964940150418980) {
  h_27(state, unused, out_8561964940150418980);
}
void car_H_27(double *state, double *unused, double *out_2805436341358833569) {
  H_27(state, unused, out_2805436341358833569);
}
void car_h_29(double *state, double *unused, double *out_4358721697008109523) {
  h_29(state, unused, out_4358721697008109523);
}
void car_H_29(double *state, double *unused, double *out_1092073614489282536) {
  H_29(state, unused, out_1092073614489282536);
}
void car_h_28(double *state, double *unused, double *out_1479122313914905983) {
  h_28(state, unused, out_1479122313914905983);
}
void car_H_28(double *state, double *unused, double *out_3990325402580248038) {
  H_28(state, unused, out_3990325402580248038);
}
void car_h_31(double *state, double *unused, double *out_1402015568006224056) {
  h_31(state, unused, out_1402015568006224056);
}
void car_H_31(double *state, double *unused, double *out_1905844726455397847) {
  H_31(state, unused, out_1905844726455397847);
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
