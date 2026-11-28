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
void err_fun(double *nom_x, double *delta_x, double *out_52242666316804912) {
   out_52242666316804912[0] = delta_x[0] + nom_x[0];
   out_52242666316804912[1] = delta_x[1] + nom_x[1];
   out_52242666316804912[2] = delta_x[2] + nom_x[2];
   out_52242666316804912[3] = delta_x[3] + nom_x[3];
   out_52242666316804912[4] = delta_x[4] + nom_x[4];
   out_52242666316804912[5] = delta_x[5] + nom_x[5];
   out_52242666316804912[6] = delta_x[6] + nom_x[6];
   out_52242666316804912[7] = delta_x[7] + nom_x[7];
   out_52242666316804912[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_4570214122495112138) {
   out_4570214122495112138[0] = -nom_x[0] + true_x[0];
   out_4570214122495112138[1] = -nom_x[1] + true_x[1];
   out_4570214122495112138[2] = -nom_x[2] + true_x[2];
   out_4570214122495112138[3] = -nom_x[3] + true_x[3];
   out_4570214122495112138[4] = -nom_x[4] + true_x[4];
   out_4570214122495112138[5] = -nom_x[5] + true_x[5];
   out_4570214122495112138[6] = -nom_x[6] + true_x[6];
   out_4570214122495112138[7] = -nom_x[7] + true_x[7];
   out_4570214122495112138[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_1673187515601145384) {
   out_1673187515601145384[0] = 1.0;
   out_1673187515601145384[1] = 0.0;
   out_1673187515601145384[2] = 0.0;
   out_1673187515601145384[3] = 0.0;
   out_1673187515601145384[4] = 0.0;
   out_1673187515601145384[5] = 0.0;
   out_1673187515601145384[6] = 0.0;
   out_1673187515601145384[7] = 0.0;
   out_1673187515601145384[8] = 0.0;
   out_1673187515601145384[9] = 0.0;
   out_1673187515601145384[10] = 1.0;
   out_1673187515601145384[11] = 0.0;
   out_1673187515601145384[12] = 0.0;
   out_1673187515601145384[13] = 0.0;
   out_1673187515601145384[14] = 0.0;
   out_1673187515601145384[15] = 0.0;
   out_1673187515601145384[16] = 0.0;
   out_1673187515601145384[17] = 0.0;
   out_1673187515601145384[18] = 0.0;
   out_1673187515601145384[19] = 0.0;
   out_1673187515601145384[20] = 1.0;
   out_1673187515601145384[21] = 0.0;
   out_1673187515601145384[22] = 0.0;
   out_1673187515601145384[23] = 0.0;
   out_1673187515601145384[24] = 0.0;
   out_1673187515601145384[25] = 0.0;
   out_1673187515601145384[26] = 0.0;
   out_1673187515601145384[27] = 0.0;
   out_1673187515601145384[28] = 0.0;
   out_1673187515601145384[29] = 0.0;
   out_1673187515601145384[30] = 1.0;
   out_1673187515601145384[31] = 0.0;
   out_1673187515601145384[32] = 0.0;
   out_1673187515601145384[33] = 0.0;
   out_1673187515601145384[34] = 0.0;
   out_1673187515601145384[35] = 0.0;
   out_1673187515601145384[36] = 0.0;
   out_1673187515601145384[37] = 0.0;
   out_1673187515601145384[38] = 0.0;
   out_1673187515601145384[39] = 0.0;
   out_1673187515601145384[40] = 1.0;
   out_1673187515601145384[41] = 0.0;
   out_1673187515601145384[42] = 0.0;
   out_1673187515601145384[43] = 0.0;
   out_1673187515601145384[44] = 0.0;
   out_1673187515601145384[45] = 0.0;
   out_1673187515601145384[46] = 0.0;
   out_1673187515601145384[47] = 0.0;
   out_1673187515601145384[48] = 0.0;
   out_1673187515601145384[49] = 0.0;
   out_1673187515601145384[50] = 1.0;
   out_1673187515601145384[51] = 0.0;
   out_1673187515601145384[52] = 0.0;
   out_1673187515601145384[53] = 0.0;
   out_1673187515601145384[54] = 0.0;
   out_1673187515601145384[55] = 0.0;
   out_1673187515601145384[56] = 0.0;
   out_1673187515601145384[57] = 0.0;
   out_1673187515601145384[58] = 0.0;
   out_1673187515601145384[59] = 0.0;
   out_1673187515601145384[60] = 1.0;
   out_1673187515601145384[61] = 0.0;
   out_1673187515601145384[62] = 0.0;
   out_1673187515601145384[63] = 0.0;
   out_1673187515601145384[64] = 0.0;
   out_1673187515601145384[65] = 0.0;
   out_1673187515601145384[66] = 0.0;
   out_1673187515601145384[67] = 0.0;
   out_1673187515601145384[68] = 0.0;
   out_1673187515601145384[69] = 0.0;
   out_1673187515601145384[70] = 1.0;
   out_1673187515601145384[71] = 0.0;
   out_1673187515601145384[72] = 0.0;
   out_1673187515601145384[73] = 0.0;
   out_1673187515601145384[74] = 0.0;
   out_1673187515601145384[75] = 0.0;
   out_1673187515601145384[76] = 0.0;
   out_1673187515601145384[77] = 0.0;
   out_1673187515601145384[78] = 0.0;
   out_1673187515601145384[79] = 0.0;
   out_1673187515601145384[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_609770788174560294) {
   out_609770788174560294[0] = state[0];
   out_609770788174560294[1] = state[1];
   out_609770788174560294[2] = state[2];
   out_609770788174560294[3] = state[3];
   out_609770788174560294[4] = state[4];
   out_609770788174560294[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_609770788174560294[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_609770788174560294[7] = state[7];
   out_609770788174560294[8] = state[8];
}
void F_fun(double *state, double dt, double *out_4074794192207865105) {
   out_4074794192207865105[0] = 1;
   out_4074794192207865105[1] = 0;
   out_4074794192207865105[2] = 0;
   out_4074794192207865105[3] = 0;
   out_4074794192207865105[4] = 0;
   out_4074794192207865105[5] = 0;
   out_4074794192207865105[6] = 0;
   out_4074794192207865105[7] = 0;
   out_4074794192207865105[8] = 0;
   out_4074794192207865105[9] = 0;
   out_4074794192207865105[10] = 1;
   out_4074794192207865105[11] = 0;
   out_4074794192207865105[12] = 0;
   out_4074794192207865105[13] = 0;
   out_4074794192207865105[14] = 0;
   out_4074794192207865105[15] = 0;
   out_4074794192207865105[16] = 0;
   out_4074794192207865105[17] = 0;
   out_4074794192207865105[18] = 0;
   out_4074794192207865105[19] = 0;
   out_4074794192207865105[20] = 1;
   out_4074794192207865105[21] = 0;
   out_4074794192207865105[22] = 0;
   out_4074794192207865105[23] = 0;
   out_4074794192207865105[24] = 0;
   out_4074794192207865105[25] = 0;
   out_4074794192207865105[26] = 0;
   out_4074794192207865105[27] = 0;
   out_4074794192207865105[28] = 0;
   out_4074794192207865105[29] = 0;
   out_4074794192207865105[30] = 1;
   out_4074794192207865105[31] = 0;
   out_4074794192207865105[32] = 0;
   out_4074794192207865105[33] = 0;
   out_4074794192207865105[34] = 0;
   out_4074794192207865105[35] = 0;
   out_4074794192207865105[36] = 0;
   out_4074794192207865105[37] = 0;
   out_4074794192207865105[38] = 0;
   out_4074794192207865105[39] = 0;
   out_4074794192207865105[40] = 1;
   out_4074794192207865105[41] = 0;
   out_4074794192207865105[42] = 0;
   out_4074794192207865105[43] = 0;
   out_4074794192207865105[44] = 0;
   out_4074794192207865105[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_4074794192207865105[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_4074794192207865105[47] = -dt*stiffness_front*state[0]/state[1];
   out_4074794192207865105[48] = -dt*stiffness_front*state[0]/state[1];
   out_4074794192207865105[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_4074794192207865105[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_4074794192207865105[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_4074794192207865105[52] = dt*stiffness_front*state[0]/state[1];
   out_4074794192207865105[53] = -9.8100000000000005*dt;
   out_4074794192207865105[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_4074794192207865105[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_4074794192207865105[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4074794192207865105[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4074794192207865105[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_4074794192207865105[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_4074794192207865105[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_4074794192207865105[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4074794192207865105[62] = 0;
   out_4074794192207865105[63] = 0;
   out_4074794192207865105[64] = 0;
   out_4074794192207865105[65] = 0;
   out_4074794192207865105[66] = 0;
   out_4074794192207865105[67] = 0;
   out_4074794192207865105[68] = 0;
   out_4074794192207865105[69] = 0;
   out_4074794192207865105[70] = 1;
   out_4074794192207865105[71] = 0;
   out_4074794192207865105[72] = 0;
   out_4074794192207865105[73] = 0;
   out_4074794192207865105[74] = 0;
   out_4074794192207865105[75] = 0;
   out_4074794192207865105[76] = 0;
   out_4074794192207865105[77] = 0;
   out_4074794192207865105[78] = 0;
   out_4074794192207865105[79] = 0;
   out_4074794192207865105[80] = 1;
}
void h_25(double *state, double *unused, double *out_5272859061006755531) {
   out_5272859061006755531[0] = state[6];
}
void H_25(double *state, double *unused, double *out_7520000710311091853) {
   out_7520000710311091853[0] = 0;
   out_7520000710311091853[1] = 0;
   out_7520000710311091853[2] = 0;
   out_7520000710311091853[3] = 0;
   out_7520000710311091853[4] = 0;
   out_7520000710311091853[5] = 0;
   out_7520000710311091853[6] = 1;
   out_7520000710311091853[7] = 0;
   out_7520000710311091853[8] = 0;
}
void h_24(double *state, double *unused, double *out_2158751071353599023) {
   out_2158751071353599023[0] = state[4];
   out_2158751071353599023[1] = state[5];
}
void H_24(double *state, double *unused, double *out_8754093764392960197) {
   out_8754093764392960197[0] = 0;
   out_8754093764392960197[1] = 0;
   out_8754093764392960197[2] = 0;
   out_8754093764392960197[3] = 0;
   out_8754093764392960197[4] = 1;
   out_8754093764392960197[5] = 0;
   out_8754093764392960197[6] = 0;
   out_8754093764392960197[7] = 0;
   out_8754093764392960197[8] = 0;
   out_8754093764392960197[9] = 0;
   out_8754093764392960197[10] = 0;
   out_8754093764392960197[11] = 0;
   out_8754093764392960197[12] = 0;
   out_8754093764392960197[13] = 0;
   out_8754093764392960197[14] = 1;
   out_8754093764392960197[15] = 0;
   out_8754093764392960197[16] = 0;
   out_8754093764392960197[17] = 0;
}
void h_30(double *state, double *unused, double *out_6403049786352445149) {
   out_6403049786352445149[0] = state[4];
}
void H_30(double *state, double *unused, double *out_603310368819475098) {
   out_603310368819475098[0] = 0;
   out_603310368819475098[1] = 0;
   out_603310368819475098[2] = 0;
   out_603310368819475098[3] = 0;
   out_603310368819475098[4] = 1;
   out_603310368819475098[5] = 0;
   out_603310368819475098[6] = 0;
   out_603310368819475098[7] = 0;
   out_603310368819475098[8] = 0;
}
void h_26(double *state, double *unused, double *out_3695140238103757981) {
   out_3695140238103757981[0] = state[7];
}
void H_26(double *state, double *unused, double *out_7185240044524403539) {
   out_7185240044524403539[0] = 0;
   out_7185240044524403539[1] = 0;
   out_7185240044524403539[2] = 0;
   out_7185240044524403539[3] = 0;
   out_7185240044524403539[4] = 0;
   out_7185240044524403539[5] = 0;
   out_7185240044524403539[6] = 0;
   out_7185240044524403539[7] = 1;
   out_7185240044524403539[8] = 0;
}
void h_27(double *state, double *unused, double *out_7607783828642286220) {
   out_7607783828642286220[0] = state[3];
}
void H_27(double *state, double *unused, double *out_2778073680619900009) {
   out_2778073680619900009[0] = 0;
   out_2778073680619900009[1] = 0;
   out_2778073680619900009[2] = 0;
   out_2778073680619900009[3] = 1;
   out_2778073680619900009[4] = 0;
   out_2778073680619900009[5] = 0;
   out_2778073680619900009[6] = 0;
   out_2778073680619900009[7] = 0;
   out_2778073680619900009[8] = 0;
}
void h_29(double *state, double *unused, double *out_645064872358707193) {
   out_645064872358707193[0] = state[1];
}
void H_29(double *state, double *unused, double *out_4491436407489451042) {
   out_4491436407489451042[0] = 0;
   out_4491436407489451042[1] = 1;
   out_4491436407489451042[2] = 0;
   out_4491436407489451042[3] = 0;
   out_4491436407489451042[4] = 0;
   out_4491436407489451042[5] = 0;
   out_4491436407489451042[6] = 0;
   out_4491436407489451042[7] = 0;
   out_4491436407489451042[8] = 0;
}
void h_28(double *state, double *unused, double *out_5613053029231940449) {
   out_5613053029231940449[0] = state[0];
}
void H_28(double *state, double *unused, double *out_8872908649150570000) {
   out_8872908649150570000[0] = 1;
   out_8872908649150570000[1] = 0;
   out_8872908649150570000[2] = 0;
   out_8872908649150570000[3] = 0;
   out_8872908649150570000[4] = 0;
   out_8872908649150570000[5] = 0;
   out_8872908649150570000[6] = 0;
   out_8872908649150570000[7] = 0;
   out_8872908649150570000[8] = 0;
}
void h_31(double *state, double *unused, double *out_2490406195813916910) {
   out_2490406195813916910[0] = state[8];
}
void H_31(double *state, double *unused, double *out_7489354748434131425) {
   out_7489354748434131425[0] = 0;
   out_7489354748434131425[1] = 0;
   out_7489354748434131425[2] = 0;
   out_7489354748434131425[3] = 0;
   out_7489354748434131425[4] = 0;
   out_7489354748434131425[5] = 0;
   out_7489354748434131425[6] = 0;
   out_7489354748434131425[7] = 0;
   out_7489354748434131425[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_52242666316804912) {
  err_fun(nom_x, delta_x, out_52242666316804912);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_4570214122495112138) {
  inv_err_fun(nom_x, true_x, out_4570214122495112138);
}
void car_H_mod_fun(double *state, double *out_1673187515601145384) {
  H_mod_fun(state, out_1673187515601145384);
}
void car_f_fun(double *state, double dt, double *out_609770788174560294) {
  f_fun(state,  dt, out_609770788174560294);
}
void car_F_fun(double *state, double dt, double *out_4074794192207865105) {
  F_fun(state,  dt, out_4074794192207865105);
}
void car_h_25(double *state, double *unused, double *out_5272859061006755531) {
  h_25(state, unused, out_5272859061006755531);
}
void car_H_25(double *state, double *unused, double *out_7520000710311091853) {
  H_25(state, unused, out_7520000710311091853);
}
void car_h_24(double *state, double *unused, double *out_2158751071353599023) {
  h_24(state, unused, out_2158751071353599023);
}
void car_H_24(double *state, double *unused, double *out_8754093764392960197) {
  H_24(state, unused, out_8754093764392960197);
}
void car_h_30(double *state, double *unused, double *out_6403049786352445149) {
  h_30(state, unused, out_6403049786352445149);
}
void car_H_30(double *state, double *unused, double *out_603310368819475098) {
  H_30(state, unused, out_603310368819475098);
}
void car_h_26(double *state, double *unused, double *out_3695140238103757981) {
  h_26(state, unused, out_3695140238103757981);
}
void car_H_26(double *state, double *unused, double *out_7185240044524403539) {
  H_26(state, unused, out_7185240044524403539);
}
void car_h_27(double *state, double *unused, double *out_7607783828642286220) {
  h_27(state, unused, out_7607783828642286220);
}
void car_H_27(double *state, double *unused, double *out_2778073680619900009) {
  H_27(state, unused, out_2778073680619900009);
}
void car_h_29(double *state, double *unused, double *out_645064872358707193) {
  h_29(state, unused, out_645064872358707193);
}
void car_H_29(double *state, double *unused, double *out_4491436407489451042) {
  H_29(state, unused, out_4491436407489451042);
}
void car_h_28(double *state, double *unused, double *out_5613053029231940449) {
  h_28(state, unused, out_5613053029231940449);
}
void car_H_28(double *state, double *unused, double *out_8872908649150570000) {
  H_28(state, unused, out_8872908649150570000);
}
void car_h_31(double *state, double *unused, double *out_2490406195813916910) {
  h_31(state, unused, out_2490406195813916910);
}
void car_H_31(double *state, double *unused, double *out_7489354748434131425) {
  H_31(state, unused, out_7489354748434131425);
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
