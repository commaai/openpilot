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
 *                       Code generated with sympy 1.9                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_6604114408220947675) {
   out_6604114408220947675[0] = delta_x[0] + nom_x[0];
   out_6604114408220947675[1] = delta_x[1] + nom_x[1];
   out_6604114408220947675[2] = delta_x[2] + nom_x[2];
   out_6604114408220947675[3] = delta_x[3] + nom_x[3];
   out_6604114408220947675[4] = delta_x[4] + nom_x[4];
   out_6604114408220947675[5] = delta_x[5] + nom_x[5];
   out_6604114408220947675[6] = delta_x[6] + nom_x[6];
   out_6604114408220947675[7] = delta_x[7] + nom_x[7];
   out_6604114408220947675[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_8451809277703177441) {
   out_8451809277703177441[0] = -nom_x[0] + true_x[0];
   out_8451809277703177441[1] = -nom_x[1] + true_x[1];
   out_8451809277703177441[2] = -nom_x[2] + true_x[2];
   out_8451809277703177441[3] = -nom_x[3] + true_x[3];
   out_8451809277703177441[4] = -nom_x[4] + true_x[4];
   out_8451809277703177441[5] = -nom_x[5] + true_x[5];
   out_8451809277703177441[6] = -nom_x[6] + true_x[6];
   out_8451809277703177441[7] = -nom_x[7] + true_x[7];
   out_8451809277703177441[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_8861781352220683421) {
   out_8861781352220683421[0] = 1.0;
   out_8861781352220683421[1] = 0;
   out_8861781352220683421[2] = 0;
   out_8861781352220683421[3] = 0;
   out_8861781352220683421[4] = 0;
   out_8861781352220683421[5] = 0;
   out_8861781352220683421[6] = 0;
   out_8861781352220683421[7] = 0;
   out_8861781352220683421[8] = 0;
   out_8861781352220683421[9] = 0;
   out_8861781352220683421[10] = 1.0;
   out_8861781352220683421[11] = 0;
   out_8861781352220683421[12] = 0;
   out_8861781352220683421[13] = 0;
   out_8861781352220683421[14] = 0;
   out_8861781352220683421[15] = 0;
   out_8861781352220683421[16] = 0;
   out_8861781352220683421[17] = 0;
   out_8861781352220683421[18] = 0;
   out_8861781352220683421[19] = 0;
   out_8861781352220683421[20] = 1.0;
   out_8861781352220683421[21] = 0;
   out_8861781352220683421[22] = 0;
   out_8861781352220683421[23] = 0;
   out_8861781352220683421[24] = 0;
   out_8861781352220683421[25] = 0;
   out_8861781352220683421[26] = 0;
   out_8861781352220683421[27] = 0;
   out_8861781352220683421[28] = 0;
   out_8861781352220683421[29] = 0;
   out_8861781352220683421[30] = 1.0;
   out_8861781352220683421[31] = 0;
   out_8861781352220683421[32] = 0;
   out_8861781352220683421[33] = 0;
   out_8861781352220683421[34] = 0;
   out_8861781352220683421[35] = 0;
   out_8861781352220683421[36] = 0;
   out_8861781352220683421[37] = 0;
   out_8861781352220683421[38] = 0;
   out_8861781352220683421[39] = 0;
   out_8861781352220683421[40] = 1.0;
   out_8861781352220683421[41] = 0;
   out_8861781352220683421[42] = 0;
   out_8861781352220683421[43] = 0;
   out_8861781352220683421[44] = 0;
   out_8861781352220683421[45] = 0;
   out_8861781352220683421[46] = 0;
   out_8861781352220683421[47] = 0;
   out_8861781352220683421[48] = 0;
   out_8861781352220683421[49] = 0;
   out_8861781352220683421[50] = 1.0;
   out_8861781352220683421[51] = 0;
   out_8861781352220683421[52] = 0;
   out_8861781352220683421[53] = 0;
   out_8861781352220683421[54] = 0;
   out_8861781352220683421[55] = 0;
   out_8861781352220683421[56] = 0;
   out_8861781352220683421[57] = 0;
   out_8861781352220683421[58] = 0;
   out_8861781352220683421[59] = 0;
   out_8861781352220683421[60] = 1.0;
   out_8861781352220683421[61] = 0;
   out_8861781352220683421[62] = 0;
   out_8861781352220683421[63] = 0;
   out_8861781352220683421[64] = 0;
   out_8861781352220683421[65] = 0;
   out_8861781352220683421[66] = 0;
   out_8861781352220683421[67] = 0;
   out_8861781352220683421[68] = 0;
   out_8861781352220683421[69] = 0;
   out_8861781352220683421[70] = 1.0;
   out_8861781352220683421[71] = 0;
   out_8861781352220683421[72] = 0;
   out_8861781352220683421[73] = 0;
   out_8861781352220683421[74] = 0;
   out_8861781352220683421[75] = 0;
   out_8861781352220683421[76] = 0;
   out_8861781352220683421[77] = 0;
   out_8861781352220683421[78] = 0;
   out_8861781352220683421[79] = 0;
   out_8861781352220683421[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_8082967332842324869) {
   out_8082967332842324869[0] = state[0];
   out_8082967332842324869[1] = state[1];
   out_8082967332842324869[2] = state[2];
   out_8082967332842324869[3] = state[3];
   out_8082967332842324869[4] = state[4];
   out_8082967332842324869[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_8082967332842324869[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_8082967332842324869[7] = state[7];
   out_8082967332842324869[8] = state[8];
}
void F_fun(double *state, double dt, double *out_4095617844735775990) {
   out_4095617844735775990[0] = 1;
   out_4095617844735775990[1] = 0;
   out_4095617844735775990[2] = 0;
   out_4095617844735775990[3] = 0;
   out_4095617844735775990[4] = 0;
   out_4095617844735775990[5] = 0;
   out_4095617844735775990[6] = 0;
   out_4095617844735775990[7] = 0;
   out_4095617844735775990[8] = 0;
   out_4095617844735775990[9] = 0;
   out_4095617844735775990[10] = 1;
   out_4095617844735775990[11] = 0;
   out_4095617844735775990[12] = 0;
   out_4095617844735775990[13] = 0;
   out_4095617844735775990[14] = 0;
   out_4095617844735775990[15] = 0;
   out_4095617844735775990[16] = 0;
   out_4095617844735775990[17] = 0;
   out_4095617844735775990[18] = 0;
   out_4095617844735775990[19] = 0;
   out_4095617844735775990[20] = 1;
   out_4095617844735775990[21] = 0;
   out_4095617844735775990[22] = 0;
   out_4095617844735775990[23] = 0;
   out_4095617844735775990[24] = 0;
   out_4095617844735775990[25] = 0;
   out_4095617844735775990[26] = 0;
   out_4095617844735775990[27] = 0;
   out_4095617844735775990[28] = 0;
   out_4095617844735775990[29] = 0;
   out_4095617844735775990[30] = 1;
   out_4095617844735775990[31] = 0;
   out_4095617844735775990[32] = 0;
   out_4095617844735775990[33] = 0;
   out_4095617844735775990[34] = 0;
   out_4095617844735775990[35] = 0;
   out_4095617844735775990[36] = 0;
   out_4095617844735775990[37] = 0;
   out_4095617844735775990[38] = 0;
   out_4095617844735775990[39] = 0;
   out_4095617844735775990[40] = 1;
   out_4095617844735775990[41] = 0;
   out_4095617844735775990[42] = 0;
   out_4095617844735775990[43] = 0;
   out_4095617844735775990[44] = 0;
   out_4095617844735775990[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_4095617844735775990[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_4095617844735775990[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_4095617844735775990[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_4095617844735775990[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_4095617844735775990[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_4095617844735775990[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_4095617844735775990[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_4095617844735775990[53] = -9.8000000000000007*dt;
   out_4095617844735775990[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_4095617844735775990[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_4095617844735775990[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4095617844735775990[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4095617844735775990[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_4095617844735775990[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_4095617844735775990[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_4095617844735775990[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4095617844735775990[62] = 0;
   out_4095617844735775990[63] = 0;
   out_4095617844735775990[64] = 0;
   out_4095617844735775990[65] = 0;
   out_4095617844735775990[66] = 0;
   out_4095617844735775990[67] = 0;
   out_4095617844735775990[68] = 0;
   out_4095617844735775990[69] = 0;
   out_4095617844735775990[70] = 1;
   out_4095617844735775990[71] = 0;
   out_4095617844735775990[72] = 0;
   out_4095617844735775990[73] = 0;
   out_4095617844735775990[74] = 0;
   out_4095617844735775990[75] = 0;
   out_4095617844735775990[76] = 0;
   out_4095617844735775990[77] = 0;
   out_4095617844735775990[78] = 0;
   out_4095617844735775990[79] = 0;
   out_4095617844735775990[80] = 1;
}
void h_25(double *state, double *unused, double *out_6175323206303387376) {
   out_6175323206303387376[0] = state[6];
}
void H_25(double *state, double *unused, double *out_487413547395389457) {
   out_487413547395389457[0] = 0;
   out_487413547395389457[1] = 0;
   out_487413547395389457[2] = 0;
   out_487413547395389457[3] = 0;
   out_487413547395389457[4] = 0;
   out_487413547395389457[5] = 0;
   out_487413547395389457[6] = 1;
   out_487413547395389457[7] = 0;
   out_487413547395389457[8] = 0;
}
void h_24(double *state, double *unused, double *out_1658736045300990593) {
   out_1658736045300990593[0] = state[4];
   out_1658736045300990593[1] = state[5];
}
void H_24(double *state, double *unused, double *out_5233922874361898491) {
   out_5233922874361898491[0] = 0;
   out_5233922874361898491[1] = 0;
   out_5233922874361898491[2] = 0;
   out_5233922874361898491[3] = 0;
   out_5233922874361898491[4] = 1;
   out_5233922874361898491[5] = 0;
   out_5233922874361898491[6] = 0;
   out_5233922874361898491[7] = 0;
   out_5233922874361898491[8] = 0;
   out_5233922874361898491[9] = 0;
   out_5233922874361898491[10] = 0;
   out_5233922874361898491[11] = 0;
   out_5233922874361898491[12] = 0;
   out_5233922874361898491[13] = 0;
   out_5233922874361898491[14] = 1;
   out_5233922874361898491[15] = 0;
   out_5233922874361898491[16] = 0;
   out_5233922874361898491[17] = 0;
}
void h_30(double *state, double *unused, double *out_5292438858594657572) {
   out_5292438858594657572[0] = state[4];
}
void H_30(double *state, double *unused, double *out_2030919411111859170) {
   out_2030919411111859170[0] = 0;
   out_2030919411111859170[1] = 0;
   out_2030919411111859170[2] = 0;
   out_2030919411111859170[3] = 0;
   out_2030919411111859170[4] = 1;
   out_2030919411111859170[5] = 0;
   out_2030919411111859170[6] = 0;
   out_2030919411111859170[7] = 0;
   out_2030919411111859170[8] = 0;
}
void h_26(double *state, double *unused, double *out_66531642759332625) {
   out_66531642759332625[0] = state[7];
}
void H_26(double *state, double *unused, double *out_4228916866269445681) {
   out_4228916866269445681[0] = 0;
   out_4228916866269445681[1] = 0;
   out_4228916866269445681[2] = 0;
   out_4228916866269445681[3] = 0;
   out_4228916866269445681[4] = 0;
   out_4228916866269445681[5] = 0;
   out_4228916866269445681[6] = 0;
   out_4228916866269445681[7] = 1;
   out_4228916866269445681[8] = 0;
}
void h_27(double *state, double *unused, double *out_2412839475501454032) {
   out_2412839475501454032[0] = state[3];
}
void H_27(double *state, double *unused, double *out_7189873189323422566) {
   out_7189873189323422566[0] = 0;
   out_7189873189323422566[1] = 0;
   out_7189873189323422566[2] = 0;
   out_7189873189323422566[3] = 1;
   out_7189873189323422566[4] = 0;
   out_7189873189323422566[5] = 0;
   out_7189873189323422566[6] = 0;
   out_7189873189323422566[7] = 0;
   out_7189873189323422566[8] = 0;
}
void h_29(double *state, double *unused, double *out_2688033537785959921) {
   out_2688033537785959921[0] = state[1];
}
void H_29(double *state, double *unused, double *out_2541150755426251354) {
   out_2541150755426251354[0] = 0;
   out_2541150755426251354[1] = 1;
   out_2541150755426251354[2] = 0;
   out_2541150755426251354[3] = 0;
   out_2541150755426251354[4] = 0;
   out_2541150755426251354[5] = 0;
   out_2541150755426251354[6] = 0;
   out_2541150755426251354[7] = 0;
   out_2541150755426251354[8] = 0;
}
void h_28(double *state, double *unused, double *out_3892767580075800992) {
   out_3892767580075800992[0] = state[0];
}
void H_28(double *state, double *unused, double *out_2541248261643279220) {
   out_2541248261643279220[0] = 1;
   out_2541248261643279220[1] = 0;
   out_2541248261643279220[2] = 0;
   out_2541248261643279220[3] = 0;
   out_2541248261643279220[4] = 0;
   out_2541248261643279220[5] = 0;
   out_2541248261643279220[6] = 0;
   out_2541248261643279220[7] = 0;
   out_2541248261643279220[8] = 0;
}
void h_31(double *state, double *unused, double *out_5536559782514876574) {
   out_5536559782514876574[0] = state[8];
}
void H_31(double *state, double *unused, double *out_4855124968502797157) {
   out_4855124968502797157[0] = 0;
   out_4855124968502797157[1] = 0;
   out_4855124968502797157[2] = 0;
   out_4855124968502797157[3] = 0;
   out_4855124968502797157[4] = 0;
   out_4855124968502797157[5] = 0;
   out_4855124968502797157[6] = 0;
   out_4855124968502797157[7] = 0;
   out_4855124968502797157[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_6604114408220947675) {
  err_fun(nom_x, delta_x, out_6604114408220947675);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_8451809277703177441) {
  inv_err_fun(nom_x, true_x, out_8451809277703177441);
}
void car_H_mod_fun(double *state, double *out_8861781352220683421) {
  H_mod_fun(state, out_8861781352220683421);
}
void car_f_fun(double *state, double dt, double *out_8082967332842324869) {
  f_fun(state,  dt, out_8082967332842324869);
}
void car_F_fun(double *state, double dt, double *out_4095617844735775990) {
  F_fun(state,  dt, out_4095617844735775990);
}
void car_h_25(double *state, double *unused, double *out_6175323206303387376) {
  h_25(state, unused, out_6175323206303387376);
}
void car_H_25(double *state, double *unused, double *out_487413547395389457) {
  H_25(state, unused, out_487413547395389457);
}
void car_h_24(double *state, double *unused, double *out_1658736045300990593) {
  h_24(state, unused, out_1658736045300990593);
}
void car_H_24(double *state, double *unused, double *out_5233922874361898491) {
  H_24(state, unused, out_5233922874361898491);
}
void car_h_30(double *state, double *unused, double *out_5292438858594657572) {
  h_30(state, unused, out_5292438858594657572);
}
void car_H_30(double *state, double *unused, double *out_2030919411111859170) {
  H_30(state, unused, out_2030919411111859170);
}
void car_h_26(double *state, double *unused, double *out_66531642759332625) {
  h_26(state, unused, out_66531642759332625);
}
void car_H_26(double *state, double *unused, double *out_4228916866269445681) {
  H_26(state, unused, out_4228916866269445681);
}
void car_h_27(double *state, double *unused, double *out_2412839475501454032) {
  h_27(state, unused, out_2412839475501454032);
}
void car_H_27(double *state, double *unused, double *out_7189873189323422566) {
  H_27(state, unused, out_7189873189323422566);
}
void car_h_29(double *state, double *unused, double *out_2688033537785959921) {
  h_29(state, unused, out_2688033537785959921);
}
void car_H_29(double *state, double *unused, double *out_2541150755426251354) {
  H_29(state, unused, out_2541150755426251354);
}
void car_h_28(double *state, double *unused, double *out_3892767580075800992) {
  h_28(state, unused, out_3892767580075800992);
}
void car_H_28(double *state, double *unused, double *out_2541248261643279220) {
  H_28(state, unused, out_2541248261643279220);
}
void car_h_31(double *state, double *unused, double *out_5536559782514876574) {
  h_31(state, unused, out_5536559782514876574);
}
void car_H_31(double *state, double *unused, double *out_4855124968502797157) {
  H_31(state, unused, out_4855124968502797157);
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

ekf_init(car);
