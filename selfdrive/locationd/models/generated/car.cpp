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
void err_fun(double *nom_x, double *delta_x, double *out_2637591915669889113) {
   out_2637591915669889113[0] = delta_x[0] + nom_x[0];
   out_2637591915669889113[1] = delta_x[1] + nom_x[1];
   out_2637591915669889113[2] = delta_x[2] + nom_x[2];
   out_2637591915669889113[3] = delta_x[3] + nom_x[3];
   out_2637591915669889113[4] = delta_x[4] + nom_x[4];
   out_2637591915669889113[5] = delta_x[5] + nom_x[5];
   out_2637591915669889113[6] = delta_x[6] + nom_x[6];
   out_2637591915669889113[7] = delta_x[7] + nom_x[7];
   out_2637591915669889113[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_7341338816872437038) {
   out_7341338816872437038[0] = -nom_x[0] + true_x[0];
   out_7341338816872437038[1] = -nom_x[1] + true_x[1];
   out_7341338816872437038[2] = -nom_x[2] + true_x[2];
   out_7341338816872437038[3] = -nom_x[3] + true_x[3];
   out_7341338816872437038[4] = -nom_x[4] + true_x[4];
   out_7341338816872437038[5] = -nom_x[5] + true_x[5];
   out_7341338816872437038[6] = -nom_x[6] + true_x[6];
   out_7341338816872437038[7] = -nom_x[7] + true_x[7];
   out_7341338816872437038[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_3673445024695914894) {
   out_3673445024695914894[0] = 1.0;
   out_3673445024695914894[1] = 0;
   out_3673445024695914894[2] = 0;
   out_3673445024695914894[3] = 0;
   out_3673445024695914894[4] = 0;
   out_3673445024695914894[5] = 0;
   out_3673445024695914894[6] = 0;
   out_3673445024695914894[7] = 0;
   out_3673445024695914894[8] = 0;
   out_3673445024695914894[9] = 0;
   out_3673445024695914894[10] = 1.0;
   out_3673445024695914894[11] = 0;
   out_3673445024695914894[12] = 0;
   out_3673445024695914894[13] = 0;
   out_3673445024695914894[14] = 0;
   out_3673445024695914894[15] = 0;
   out_3673445024695914894[16] = 0;
   out_3673445024695914894[17] = 0;
   out_3673445024695914894[18] = 0;
   out_3673445024695914894[19] = 0;
   out_3673445024695914894[20] = 1.0;
   out_3673445024695914894[21] = 0;
   out_3673445024695914894[22] = 0;
   out_3673445024695914894[23] = 0;
   out_3673445024695914894[24] = 0;
   out_3673445024695914894[25] = 0;
   out_3673445024695914894[26] = 0;
   out_3673445024695914894[27] = 0;
   out_3673445024695914894[28] = 0;
   out_3673445024695914894[29] = 0;
   out_3673445024695914894[30] = 1.0;
   out_3673445024695914894[31] = 0;
   out_3673445024695914894[32] = 0;
   out_3673445024695914894[33] = 0;
   out_3673445024695914894[34] = 0;
   out_3673445024695914894[35] = 0;
   out_3673445024695914894[36] = 0;
   out_3673445024695914894[37] = 0;
   out_3673445024695914894[38] = 0;
   out_3673445024695914894[39] = 0;
   out_3673445024695914894[40] = 1.0;
   out_3673445024695914894[41] = 0;
   out_3673445024695914894[42] = 0;
   out_3673445024695914894[43] = 0;
   out_3673445024695914894[44] = 0;
   out_3673445024695914894[45] = 0;
   out_3673445024695914894[46] = 0;
   out_3673445024695914894[47] = 0;
   out_3673445024695914894[48] = 0;
   out_3673445024695914894[49] = 0;
   out_3673445024695914894[50] = 1.0;
   out_3673445024695914894[51] = 0;
   out_3673445024695914894[52] = 0;
   out_3673445024695914894[53] = 0;
   out_3673445024695914894[54] = 0;
   out_3673445024695914894[55] = 0;
   out_3673445024695914894[56] = 0;
   out_3673445024695914894[57] = 0;
   out_3673445024695914894[58] = 0;
   out_3673445024695914894[59] = 0;
   out_3673445024695914894[60] = 1.0;
   out_3673445024695914894[61] = 0;
   out_3673445024695914894[62] = 0;
   out_3673445024695914894[63] = 0;
   out_3673445024695914894[64] = 0;
   out_3673445024695914894[65] = 0;
   out_3673445024695914894[66] = 0;
   out_3673445024695914894[67] = 0;
   out_3673445024695914894[68] = 0;
   out_3673445024695914894[69] = 0;
   out_3673445024695914894[70] = 1.0;
   out_3673445024695914894[71] = 0;
   out_3673445024695914894[72] = 0;
   out_3673445024695914894[73] = 0;
   out_3673445024695914894[74] = 0;
   out_3673445024695914894[75] = 0;
   out_3673445024695914894[76] = 0;
   out_3673445024695914894[77] = 0;
   out_3673445024695914894[78] = 0;
   out_3673445024695914894[79] = 0;
   out_3673445024695914894[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_8303322641241644895) {
   out_8303322641241644895[0] = state[0];
   out_8303322641241644895[1] = state[1];
   out_8303322641241644895[2] = state[2];
   out_8303322641241644895[3] = state[3];
   out_8303322641241644895[4] = state[4];
   out_8303322641241644895[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_8303322641241644895[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_8303322641241644895[7] = state[7];
   out_8303322641241644895[8] = state[8];
}
void F_fun(double *state, double dt, double *out_8675149220450121470) {
   out_8675149220450121470[0] = 1;
   out_8675149220450121470[1] = 0;
   out_8675149220450121470[2] = 0;
   out_8675149220450121470[3] = 0;
   out_8675149220450121470[4] = 0;
   out_8675149220450121470[5] = 0;
   out_8675149220450121470[6] = 0;
   out_8675149220450121470[7] = 0;
   out_8675149220450121470[8] = 0;
   out_8675149220450121470[9] = 0;
   out_8675149220450121470[10] = 1;
   out_8675149220450121470[11] = 0;
   out_8675149220450121470[12] = 0;
   out_8675149220450121470[13] = 0;
   out_8675149220450121470[14] = 0;
   out_8675149220450121470[15] = 0;
   out_8675149220450121470[16] = 0;
   out_8675149220450121470[17] = 0;
   out_8675149220450121470[18] = 0;
   out_8675149220450121470[19] = 0;
   out_8675149220450121470[20] = 1;
   out_8675149220450121470[21] = 0;
   out_8675149220450121470[22] = 0;
   out_8675149220450121470[23] = 0;
   out_8675149220450121470[24] = 0;
   out_8675149220450121470[25] = 0;
   out_8675149220450121470[26] = 0;
   out_8675149220450121470[27] = 0;
   out_8675149220450121470[28] = 0;
   out_8675149220450121470[29] = 0;
   out_8675149220450121470[30] = 1;
   out_8675149220450121470[31] = 0;
   out_8675149220450121470[32] = 0;
   out_8675149220450121470[33] = 0;
   out_8675149220450121470[34] = 0;
   out_8675149220450121470[35] = 0;
   out_8675149220450121470[36] = 0;
   out_8675149220450121470[37] = 0;
   out_8675149220450121470[38] = 0;
   out_8675149220450121470[39] = 0;
   out_8675149220450121470[40] = 1;
   out_8675149220450121470[41] = 0;
   out_8675149220450121470[42] = 0;
   out_8675149220450121470[43] = 0;
   out_8675149220450121470[44] = 0;
   out_8675149220450121470[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_8675149220450121470[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_8675149220450121470[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8675149220450121470[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8675149220450121470[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_8675149220450121470[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_8675149220450121470[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_8675149220450121470[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_8675149220450121470[53] = -9.8000000000000007*dt;
   out_8675149220450121470[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_8675149220450121470[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_8675149220450121470[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8675149220450121470[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8675149220450121470[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_8675149220450121470[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_8675149220450121470[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_8675149220450121470[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8675149220450121470[62] = 0;
   out_8675149220450121470[63] = 0;
   out_8675149220450121470[64] = 0;
   out_8675149220450121470[65] = 0;
   out_8675149220450121470[66] = 0;
   out_8675149220450121470[67] = 0;
   out_8675149220450121470[68] = 0;
   out_8675149220450121470[69] = 0;
   out_8675149220450121470[70] = 1;
   out_8675149220450121470[71] = 0;
   out_8675149220450121470[72] = 0;
   out_8675149220450121470[73] = 0;
   out_8675149220450121470[74] = 0;
   out_8675149220450121470[75] = 0;
   out_8675149220450121470[76] = 0;
   out_8675149220450121470[77] = 0;
   out_8675149220450121470[78] = 0;
   out_8675149220450121470[79] = 0;
   out_8675149220450121470[80] = 1;
}
void h_25(double *state, double *unused, double *out_6066814403066876528) {
   out_6066814403066876528[0] = state[6];
}
void H_25(double *state, double *unused, double *out_132270908145005372) {
   out_132270908145005372[0] = 0;
   out_132270908145005372[1] = 0;
   out_132270908145005372[2] = 0;
   out_132270908145005372[3] = 0;
   out_132270908145005372[4] = 0;
   out_132270908145005372[5] = 0;
   out_132270908145005372[6] = 1;
   out_132270908145005372[7] = 0;
   out_132270908145005372[8] = 0;
}
void h_24(double *state, double *unused, double *out_186230791854934322) {
   out_186230791854934322[0] = state[4];
   out_186230791854934322[1] = state[5];
}
void H_24(double *state, double *unused, double *out_2040378690860494194) {
   out_2040378690860494194[0] = 0;
   out_2040378690860494194[1] = 0;
   out_2040378690860494194[2] = 0;
   out_2040378690860494194[3] = 0;
   out_2040378690860494194[4] = 1;
   out_2040378690860494194[5] = 0;
   out_2040378690860494194[6] = 0;
   out_2040378690860494194[7] = 0;
   out_2040378690860494194[8] = 0;
   out_2040378690860494194[9] = 0;
   out_2040378690860494194[10] = 0;
   out_2040378690860494194[11] = 0;
   out_2040378690860494194[12] = 0;
   out_2040378690860494194[13] = 0;
   out_2040378690860494194[14] = 1;
   out_2040378690860494194[15] = 0;
   out_2040378690860494194[16] = 0;
   out_2040378690860494194[17] = 0;
}
void h_30(double *state, double *unused, double *out_8871672977068494370) {
   out_8871672977068494370[0] = state[4];
}
void H_30(double *state, double *unused, double *out_4395425421982602826) {
   out_4395425421982602826[0] = 0;
   out_4395425421982602826[1] = 0;
   out_4395425421982602826[2] = 0;
   out_4395425421982602826[3] = 0;
   out_4395425421982602826[4] = 1;
   out_4395425421982602826[5] = 0;
   out_4395425421982602826[6] = 0;
   out_4395425421982602826[7] = 0;
   out_4395425421982602826[8] = 0;
}
void h_26(double *state, double *unused, double *out_7271548445356717599) {
   out_7271548445356717599[0] = state[7];
}
void H_26(double *state, double *unused, double *out_3609232410729050852) {
   out_3609232410729050852[0] = 0;
   out_3609232410729050852[1] = 0;
   out_3609232410729050852[2] = 0;
   out_3609232410729050852[3] = 0;
   out_3609232410729050852[4] = 0;
   out_3609232410729050852[5] = 0;
   out_3609232410729050852[6] = 0;
   out_3609232410729050852[7] = 1;
   out_3609232410729050852[8] = 0;
}
void h_27(double *state, double *unused, double *out_1065141135406030883) {
   out_1065141135406030883[0] = state[3];
}
void H_27(double *state, double *unused, double *out_2171831350798659609) {
   out_2171831350798659609[0] = 0;
   out_2171831350798659609[1] = 0;
   out_2171831350798659609[2] = 0;
   out_2171831350798659609[3] = 1;
   out_2171831350798659609[4] = 0;
   out_2171831350798659609[5] = 0;
   out_2171831350798659609[6] = 0;
   out_2171831350798659609[7] = 0;
   out_2171831350798659609[8] = 0;
}
void h_29(double *state, double *unused, double *out_2796542341022470769) {
   out_2796542341022470769[0] = state[1];
}
void H_29(double *state, double *unused, double *out_3885194077668210642) {
   out_3885194077668210642[0] = 0;
   out_3885194077668210642[1] = 1;
   out_3885194077668210642[2] = 0;
   out_3885194077668210642[3] = 0;
   out_3885194077668210642[4] = 0;
   out_3885194077668210642[5] = 0;
   out_3885194077668210642[6] = 0;
   out_3885194077668210642[7] = 0;
   out_3885194077668210642[8] = 0;
}
void h_28(double *state, double *unused, double *out_3187969700777346877) {
   out_3187969700777346877[0] = state[0];
}
void H_28(double *state, double *unused, double *out_1921563806102884391) {
   out_1921563806102884391[0] = 1;
   out_1921563806102884391[1] = 0;
   out_1921563806102884391[2] = 0;
   out_1921563806102884391[3] = 0;
   out_1921563806102884391[4] = 0;
   out_1921563806102884391[5] = 0;
   out_1921563806102884391[6] = 0;
   out_1921563806102884391[7] = 0;
   out_1921563806102884391[8] = 0;
}
void h_31(double *state, double *unused, double *out_5792101744166379925) {
   out_5792101744166379925[0] = state[8];
}
void H_31(double *state, double *unused, double *out_162916870021965800) {
   out_162916870021965800[0] = 0;
   out_162916870021965800[1] = 0;
   out_162916870021965800[2] = 0;
   out_162916870021965800[3] = 0;
   out_162916870021965800[4] = 0;
   out_162916870021965800[5] = 0;
   out_162916870021965800[6] = 0;
   out_162916870021965800[7] = 0;
   out_162916870021965800[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_2637591915669889113) {
  err_fun(nom_x, delta_x, out_2637591915669889113);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7341338816872437038) {
  inv_err_fun(nom_x, true_x, out_7341338816872437038);
}
void car_H_mod_fun(double *state, double *out_3673445024695914894) {
  H_mod_fun(state, out_3673445024695914894);
}
void car_f_fun(double *state, double dt, double *out_8303322641241644895) {
  f_fun(state,  dt, out_8303322641241644895);
}
void car_F_fun(double *state, double dt, double *out_8675149220450121470) {
  F_fun(state,  dt, out_8675149220450121470);
}
void car_h_25(double *state, double *unused, double *out_6066814403066876528) {
  h_25(state, unused, out_6066814403066876528);
}
void car_H_25(double *state, double *unused, double *out_132270908145005372) {
  H_25(state, unused, out_132270908145005372);
}
void car_h_24(double *state, double *unused, double *out_186230791854934322) {
  h_24(state, unused, out_186230791854934322);
}
void car_H_24(double *state, double *unused, double *out_2040378690860494194) {
  H_24(state, unused, out_2040378690860494194);
}
void car_h_30(double *state, double *unused, double *out_8871672977068494370) {
  h_30(state, unused, out_8871672977068494370);
}
void car_H_30(double *state, double *unused, double *out_4395425421982602826) {
  H_30(state, unused, out_4395425421982602826);
}
void car_h_26(double *state, double *unused, double *out_7271548445356717599) {
  h_26(state, unused, out_7271548445356717599);
}
void car_H_26(double *state, double *unused, double *out_3609232410729050852) {
  H_26(state, unused, out_3609232410729050852);
}
void car_h_27(double *state, double *unused, double *out_1065141135406030883) {
  h_27(state, unused, out_1065141135406030883);
}
void car_H_27(double *state, double *unused, double *out_2171831350798659609) {
  H_27(state, unused, out_2171831350798659609);
}
void car_h_29(double *state, double *unused, double *out_2796542341022470769) {
  h_29(state, unused, out_2796542341022470769);
}
void car_H_29(double *state, double *unused, double *out_3885194077668210642) {
  H_29(state, unused, out_3885194077668210642);
}
void car_h_28(double *state, double *unused, double *out_3187969700777346877) {
  h_28(state, unused, out_3187969700777346877);
}
void car_H_28(double *state, double *unused, double *out_1921563806102884391) {
  H_28(state, unused, out_1921563806102884391);
}
void car_h_31(double *state, double *unused, double *out_5792101744166379925) {
  h_31(state, unused, out_5792101744166379925);
}
void car_H_31(double *state, double *unused, double *out_162916870021965800) {
  H_31(state, unused, out_162916870021965800);
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
