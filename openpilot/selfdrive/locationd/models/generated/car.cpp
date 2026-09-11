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
 *                      Code generated with SymPy 1.14.0                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_9092401817951937801) {
   out_9092401817951937801[0] = delta_x[0] + nom_x[0];
   out_9092401817951937801[1] = delta_x[1] + nom_x[1];
   out_9092401817951937801[2] = delta_x[2] + nom_x[2];
   out_9092401817951937801[3] = delta_x[3] + nom_x[3];
   out_9092401817951937801[4] = delta_x[4] + nom_x[4];
   out_9092401817951937801[5] = delta_x[5] + nom_x[5];
   out_9092401817951937801[6] = delta_x[6] + nom_x[6];
   out_9092401817951937801[7] = delta_x[7] + nom_x[7];
   out_9092401817951937801[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_6018369249920624325) {
   out_6018369249920624325[0] = -nom_x[0] + true_x[0];
   out_6018369249920624325[1] = -nom_x[1] + true_x[1];
   out_6018369249920624325[2] = -nom_x[2] + true_x[2];
   out_6018369249920624325[3] = -nom_x[3] + true_x[3];
   out_6018369249920624325[4] = -nom_x[4] + true_x[4];
   out_6018369249920624325[5] = -nom_x[5] + true_x[5];
   out_6018369249920624325[6] = -nom_x[6] + true_x[6];
   out_6018369249920624325[7] = -nom_x[7] + true_x[7];
   out_6018369249920624325[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_2017416824247113709) {
   out_2017416824247113709[0] = 1.0;
   out_2017416824247113709[1] = 0.0;
   out_2017416824247113709[2] = 0.0;
   out_2017416824247113709[3] = 0.0;
   out_2017416824247113709[4] = 0.0;
   out_2017416824247113709[5] = 0.0;
   out_2017416824247113709[6] = 0.0;
   out_2017416824247113709[7] = 0.0;
   out_2017416824247113709[8] = 0.0;
   out_2017416824247113709[9] = 0.0;
   out_2017416824247113709[10] = 1.0;
   out_2017416824247113709[11] = 0.0;
   out_2017416824247113709[12] = 0.0;
   out_2017416824247113709[13] = 0.0;
   out_2017416824247113709[14] = 0.0;
   out_2017416824247113709[15] = 0.0;
   out_2017416824247113709[16] = 0.0;
   out_2017416824247113709[17] = 0.0;
   out_2017416824247113709[18] = 0.0;
   out_2017416824247113709[19] = 0.0;
   out_2017416824247113709[20] = 1.0;
   out_2017416824247113709[21] = 0.0;
   out_2017416824247113709[22] = 0.0;
   out_2017416824247113709[23] = 0.0;
   out_2017416824247113709[24] = 0.0;
   out_2017416824247113709[25] = 0.0;
   out_2017416824247113709[26] = 0.0;
   out_2017416824247113709[27] = 0.0;
   out_2017416824247113709[28] = 0.0;
   out_2017416824247113709[29] = 0.0;
   out_2017416824247113709[30] = 1.0;
   out_2017416824247113709[31] = 0.0;
   out_2017416824247113709[32] = 0.0;
   out_2017416824247113709[33] = 0.0;
   out_2017416824247113709[34] = 0.0;
   out_2017416824247113709[35] = 0.0;
   out_2017416824247113709[36] = 0.0;
   out_2017416824247113709[37] = 0.0;
   out_2017416824247113709[38] = 0.0;
   out_2017416824247113709[39] = 0.0;
   out_2017416824247113709[40] = 1.0;
   out_2017416824247113709[41] = 0.0;
   out_2017416824247113709[42] = 0.0;
   out_2017416824247113709[43] = 0.0;
   out_2017416824247113709[44] = 0.0;
   out_2017416824247113709[45] = 0.0;
   out_2017416824247113709[46] = 0.0;
   out_2017416824247113709[47] = 0.0;
   out_2017416824247113709[48] = 0.0;
   out_2017416824247113709[49] = 0.0;
   out_2017416824247113709[50] = 1.0;
   out_2017416824247113709[51] = 0.0;
   out_2017416824247113709[52] = 0.0;
   out_2017416824247113709[53] = 0.0;
   out_2017416824247113709[54] = 0.0;
   out_2017416824247113709[55] = 0.0;
   out_2017416824247113709[56] = 0.0;
   out_2017416824247113709[57] = 0.0;
   out_2017416824247113709[58] = 0.0;
   out_2017416824247113709[59] = 0.0;
   out_2017416824247113709[60] = 1.0;
   out_2017416824247113709[61] = 0.0;
   out_2017416824247113709[62] = 0.0;
   out_2017416824247113709[63] = 0.0;
   out_2017416824247113709[64] = 0.0;
   out_2017416824247113709[65] = 0.0;
   out_2017416824247113709[66] = 0.0;
   out_2017416824247113709[67] = 0.0;
   out_2017416824247113709[68] = 0.0;
   out_2017416824247113709[69] = 0.0;
   out_2017416824247113709[70] = 1.0;
   out_2017416824247113709[71] = 0.0;
   out_2017416824247113709[72] = 0.0;
   out_2017416824247113709[73] = 0.0;
   out_2017416824247113709[74] = 0.0;
   out_2017416824247113709[75] = 0.0;
   out_2017416824247113709[76] = 0.0;
   out_2017416824247113709[77] = 0.0;
   out_2017416824247113709[78] = 0.0;
   out_2017416824247113709[79] = 0.0;
   out_2017416824247113709[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_9091538470711422403) {
   out_9091538470711422403[0] = state[0];
   out_9091538470711422403[1] = state[1];
   out_9091538470711422403[2] = state[2];
   out_9091538470711422403[3] = state[3];
   out_9091538470711422403[4] = state[4];
   out_9091538470711422403[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8100000000000005*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_9091538470711422403[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_9091538470711422403[7] = state[7];
   out_9091538470711422403[8] = state[8];
}
void F_fun(double *state, double dt, double *out_6396802349455434445) {
   out_6396802349455434445[0] = 1;
   out_6396802349455434445[1] = 0;
   out_6396802349455434445[2] = 0;
   out_6396802349455434445[3] = 0;
   out_6396802349455434445[4] = 0;
   out_6396802349455434445[5] = 0;
   out_6396802349455434445[6] = 0;
   out_6396802349455434445[7] = 0;
   out_6396802349455434445[8] = 0;
   out_6396802349455434445[9] = 0;
   out_6396802349455434445[10] = 1;
   out_6396802349455434445[11] = 0;
   out_6396802349455434445[12] = 0;
   out_6396802349455434445[13] = 0;
   out_6396802349455434445[14] = 0;
   out_6396802349455434445[15] = 0;
   out_6396802349455434445[16] = 0;
   out_6396802349455434445[17] = 0;
   out_6396802349455434445[18] = 0;
   out_6396802349455434445[19] = 0;
   out_6396802349455434445[20] = 1;
   out_6396802349455434445[21] = 0;
   out_6396802349455434445[22] = 0;
   out_6396802349455434445[23] = 0;
   out_6396802349455434445[24] = 0;
   out_6396802349455434445[25] = 0;
   out_6396802349455434445[26] = 0;
   out_6396802349455434445[27] = 0;
   out_6396802349455434445[28] = 0;
   out_6396802349455434445[29] = 0;
   out_6396802349455434445[30] = 1;
   out_6396802349455434445[31] = 0;
   out_6396802349455434445[32] = 0;
   out_6396802349455434445[33] = 0;
   out_6396802349455434445[34] = 0;
   out_6396802349455434445[35] = 0;
   out_6396802349455434445[36] = 0;
   out_6396802349455434445[37] = 0;
   out_6396802349455434445[38] = 0;
   out_6396802349455434445[39] = 0;
   out_6396802349455434445[40] = 1;
   out_6396802349455434445[41] = 0;
   out_6396802349455434445[42] = 0;
   out_6396802349455434445[43] = 0;
   out_6396802349455434445[44] = 0;
   out_6396802349455434445[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_6396802349455434445[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_6396802349455434445[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_6396802349455434445[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_6396802349455434445[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_6396802349455434445[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_6396802349455434445[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_6396802349455434445[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_6396802349455434445[53] = -9.8100000000000005*dt;
   out_6396802349455434445[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_6396802349455434445[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_6396802349455434445[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6396802349455434445[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6396802349455434445[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_6396802349455434445[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_6396802349455434445[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_6396802349455434445[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6396802349455434445[62] = 0;
   out_6396802349455434445[63] = 0;
   out_6396802349455434445[64] = 0;
   out_6396802349455434445[65] = 0;
   out_6396802349455434445[66] = 0;
   out_6396802349455434445[67] = 0;
   out_6396802349455434445[68] = 0;
   out_6396802349455434445[69] = 0;
   out_6396802349455434445[70] = 1;
   out_6396802349455434445[71] = 0;
   out_6396802349455434445[72] = 0;
   out_6396802349455434445[73] = 0;
   out_6396802349455434445[74] = 0;
   out_6396802349455434445[75] = 0;
   out_6396802349455434445[76] = 0;
   out_6396802349455434445[77] = 0;
   out_6396802349455434445[78] = 0;
   out_6396802349455434445[79] = 0;
   out_6396802349455434445[80] = 1;
}
void h_25(double *state, double *unused, double *out_7942966278303983884) {
   out_7942966278303983884[0] = state[6];
}
void H_25(double *state, double *unused, double *out_6921108649431778549) {
   out_6921108649431778549[0] = 0;
   out_6921108649431778549[1] = 0;
   out_6921108649431778549[2] = 0;
   out_6921108649431778549[3] = 0;
   out_6921108649431778549[4] = 0;
   out_6921108649431778549[5] = 0;
   out_6921108649431778549[6] = 1;
   out_6921108649431778549[7] = 0;
   out_6921108649431778549[8] = 0;
}
void h_24(double *state, double *unused, double *out_6744365657582572191) {
   out_6744365657582572191[0] = state[4];
   out_6744365657582572191[1] = state[5];
}
void H_24(double *state, double *unused, double *out_3367747875231105820) {
   out_3367747875231105820[0] = 0;
   out_3367747875231105820[1] = 0;
   out_3367747875231105820[2] = 0;
   out_3367747875231105820[3] = 0;
   out_3367747875231105820[4] = 1;
   out_3367747875231105820[5] = 0;
   out_3367747875231105820[6] = 0;
   out_3367747875231105820[7] = 0;
   out_3367747875231105820[8] = 0;
   out_3367747875231105820[9] = 0;
   out_3367747875231105820[10] = 0;
   out_3367747875231105820[11] = 0;
   out_3367747875231105820[12] = 0;
   out_3367747875231105820[13] = 0;
   out_3367747875231105820[14] = 1;
   out_3367747875231105820[15] = 0;
   out_3367747875231105820[16] = 0;
   out_3367747875231105820[17] = 0;
}
void h_30(double *state, double *unused, double *out_7785779609952854739) {
   out_7785779609952854739[0] = state[4];
}
void H_30(double *state, double *unused, double *out_9007302465770524440) {
   out_9007302465770524440[0] = 0;
   out_9007302465770524440[1] = 0;
   out_9007302465770524440[2] = 0;
   out_9007302465770524440[3] = 0;
   out_9007302465770524440[4] = 1;
   out_9007302465770524440[5] = 0;
   out_9007302465770524440[6] = 0;
   out_9007302465770524440[7] = 0;
   out_9007302465770524440[8] = 0;
}
void h_26(double *state, double *unused, double *out_2784106342292009094) {
   out_2784106342292009094[0] = state[7];
}
void H_26(double *state, double *unused, double *out_3179605330557722325) {
   out_3179605330557722325[0] = 0;
   out_3179605330557722325[1] = 0;
   out_3179605330557722325[2] = 0;
   out_3179605330557722325[3] = 0;
   out_3179605330557722325[4] = 0;
   out_3179605330557722325[5] = 0;
   out_3179605330557722325[6] = 0;
   out_3179605330557722325[7] = 1;
   out_3179605330557722325[8] = 0;
}
void h_27(double *state, double *unused, double *out_507822843875283071) {
   out_507822843875283071[0] = state[3];
}
void H_27(double *state, double *unused, double *out_7264678296138602265) {
   out_7264678296138602265[0] = 0;
   out_7264678296138602265[1] = 0;
   out_7264678296138602265[2] = 0;
   out_7264678296138602265[3] = 1;
   out_7264678296138602265[4] = 0;
   out_7264678296138602265[5] = 0;
   out_7264678296138602265[6] = 0;
   out_7264678296138602265[7] = 0;
   out_7264678296138602265[8] = 0;
}
void h_29(double *state, double *unused, double *out_1864654999490699737) {
   out_1864654999490699737[0] = state[1];
}
void H_29(double *state, double *unused, double *out_8497071121456132256) {
   out_8497071121456132256[0] = 0;
   out_8497071121456132256[1] = 1;
   out_8497071121456132256[2] = 0;
   out_8497071121456132256[3] = 0;
   out_8497071121456132256[4] = 0;
   out_8497071121456132256[5] = 0;
   out_8497071121456132256[6] = 0;
   out_8497071121456132256[7] = 0;
   out_8497071121456132256[8] = 0;
}
void h_28(double *state, double *unused, double *out_435687009026379364) {
   out_435687009026379364[0] = state[0];
}
void H_28(double *state, double *unused, double *out_4867273935183888786) {
   out_4867273935183888786[0] = 1;
   out_4867273935183888786[1] = 0;
   out_4867273935183888786[2] = 0;
   out_4867273935183888786[3] = 0;
   out_4867273935183888786[4] = 0;
   out_4867273935183888786[5] = 0;
   out_4867273935183888786[6] = 0;
   out_4867273935183888786[7] = 0;
   out_4867273935183888786[8] = 0;
}
void h_31(double *state, double *unused, double *out_7667772216019477995) {
   out_7667772216019477995[0] = state[8];
}
void H_31(double *state, double *unused, double *out_2553397228324370849) {
   out_2553397228324370849[0] = 0;
   out_2553397228324370849[1] = 0;
   out_2553397228324370849[2] = 0;
   out_2553397228324370849[3] = 0;
   out_2553397228324370849[4] = 0;
   out_2553397228324370849[5] = 0;
   out_2553397228324370849[6] = 0;
   out_2553397228324370849[7] = 0;
   out_2553397228324370849[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_9092401817951937801) {
  err_fun(nom_x, delta_x, out_9092401817951937801);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_6018369249920624325) {
  inv_err_fun(nom_x, true_x, out_6018369249920624325);
}
void car_H_mod_fun(double *state, double *out_2017416824247113709) {
  H_mod_fun(state, out_2017416824247113709);
}
void car_f_fun(double *state, double dt, double *out_9091538470711422403) {
  f_fun(state,  dt, out_9091538470711422403);
}
void car_F_fun(double *state, double dt, double *out_6396802349455434445) {
  F_fun(state,  dt, out_6396802349455434445);
}
void car_h_25(double *state, double *unused, double *out_7942966278303983884) {
  h_25(state, unused, out_7942966278303983884);
}
void car_H_25(double *state, double *unused, double *out_6921108649431778549) {
  H_25(state, unused, out_6921108649431778549);
}
void car_h_24(double *state, double *unused, double *out_6744365657582572191) {
  h_24(state, unused, out_6744365657582572191);
}
void car_H_24(double *state, double *unused, double *out_3367747875231105820) {
  H_24(state, unused, out_3367747875231105820);
}
void car_h_30(double *state, double *unused, double *out_7785779609952854739) {
  h_30(state, unused, out_7785779609952854739);
}
void car_H_30(double *state, double *unused, double *out_9007302465770524440) {
  H_30(state, unused, out_9007302465770524440);
}
void car_h_26(double *state, double *unused, double *out_2784106342292009094) {
  h_26(state, unused, out_2784106342292009094);
}
void car_H_26(double *state, double *unused, double *out_3179605330557722325) {
  H_26(state, unused, out_3179605330557722325);
}
void car_h_27(double *state, double *unused, double *out_507822843875283071) {
  h_27(state, unused, out_507822843875283071);
}
void car_H_27(double *state, double *unused, double *out_7264678296138602265) {
  H_27(state, unused, out_7264678296138602265);
}
void car_h_29(double *state, double *unused, double *out_1864654999490699737) {
  h_29(state, unused, out_1864654999490699737);
}
void car_H_29(double *state, double *unused, double *out_8497071121456132256) {
  H_29(state, unused, out_8497071121456132256);
}
void car_h_28(double *state, double *unused, double *out_435687009026379364) {
  h_28(state, unused, out_435687009026379364);
}
void car_H_28(double *state, double *unused, double *out_4867273935183888786) {
  H_28(state, unused, out_4867273935183888786);
}
void car_h_31(double *state, double *unused, double *out_7667772216019477995) {
  h_31(state, unused, out_7667772216019477995);
}
void car_H_31(double *state, double *unused, double *out_2553397228324370849) {
  H_31(state, unused, out_2553397228324370849);
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
