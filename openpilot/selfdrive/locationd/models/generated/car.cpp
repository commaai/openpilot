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
void err_fun(double *nom_x, double *delta_x, double *out_5852520466696422965) {
   out_5852520466696422965[0] = delta_x[0] + nom_x[0];
   out_5852520466696422965[1] = delta_x[1] + nom_x[1];
   out_5852520466696422965[2] = delta_x[2] + nom_x[2];
   out_5852520466696422965[3] = delta_x[3] + nom_x[3];
   out_5852520466696422965[4] = delta_x[4] + nom_x[4];
   out_5852520466696422965[5] = delta_x[5] + nom_x[5];
   out_5852520466696422965[6] = delta_x[6] + nom_x[6];
   out_5852520466696422965[7] = delta_x[7] + nom_x[7];
   out_5852520466696422965[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_1553841940871946652) {
   out_1553841940871946652[0] = -nom_x[0] + true_x[0];
   out_1553841940871946652[1] = -nom_x[1] + true_x[1];
   out_1553841940871946652[2] = -nom_x[2] + true_x[2];
   out_1553841940871946652[3] = -nom_x[3] + true_x[3];
   out_1553841940871946652[4] = -nom_x[4] + true_x[4];
   out_1553841940871946652[5] = -nom_x[5] + true_x[5];
   out_1553841940871946652[6] = -nom_x[6] + true_x[6];
   out_1553841940871946652[7] = -nom_x[7] + true_x[7];
   out_1553841940871946652[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_3868422467346524374) {
   out_3868422467346524374[0] = 1.0;
   out_3868422467346524374[1] = 0.0;
   out_3868422467346524374[2] = 0.0;
   out_3868422467346524374[3] = 0.0;
   out_3868422467346524374[4] = 0.0;
   out_3868422467346524374[5] = 0.0;
   out_3868422467346524374[6] = 0.0;
   out_3868422467346524374[7] = 0.0;
   out_3868422467346524374[8] = 0.0;
   out_3868422467346524374[9] = 0.0;
   out_3868422467346524374[10] = 1.0;
   out_3868422467346524374[11] = 0.0;
   out_3868422467346524374[12] = 0.0;
   out_3868422467346524374[13] = 0.0;
   out_3868422467346524374[14] = 0.0;
   out_3868422467346524374[15] = 0.0;
   out_3868422467346524374[16] = 0.0;
   out_3868422467346524374[17] = 0.0;
   out_3868422467346524374[18] = 0.0;
   out_3868422467346524374[19] = 0.0;
   out_3868422467346524374[20] = 1.0;
   out_3868422467346524374[21] = 0.0;
   out_3868422467346524374[22] = 0.0;
   out_3868422467346524374[23] = 0.0;
   out_3868422467346524374[24] = 0.0;
   out_3868422467346524374[25] = 0.0;
   out_3868422467346524374[26] = 0.0;
   out_3868422467346524374[27] = 0.0;
   out_3868422467346524374[28] = 0.0;
   out_3868422467346524374[29] = 0.0;
   out_3868422467346524374[30] = 1.0;
   out_3868422467346524374[31] = 0.0;
   out_3868422467346524374[32] = 0.0;
   out_3868422467346524374[33] = 0.0;
   out_3868422467346524374[34] = 0.0;
   out_3868422467346524374[35] = 0.0;
   out_3868422467346524374[36] = 0.0;
   out_3868422467346524374[37] = 0.0;
   out_3868422467346524374[38] = 0.0;
   out_3868422467346524374[39] = 0.0;
   out_3868422467346524374[40] = 1.0;
   out_3868422467346524374[41] = 0.0;
   out_3868422467346524374[42] = 0.0;
   out_3868422467346524374[43] = 0.0;
   out_3868422467346524374[44] = 0.0;
   out_3868422467346524374[45] = 0.0;
   out_3868422467346524374[46] = 0.0;
   out_3868422467346524374[47] = 0.0;
   out_3868422467346524374[48] = 0.0;
   out_3868422467346524374[49] = 0.0;
   out_3868422467346524374[50] = 1.0;
   out_3868422467346524374[51] = 0.0;
   out_3868422467346524374[52] = 0.0;
   out_3868422467346524374[53] = 0.0;
   out_3868422467346524374[54] = 0.0;
   out_3868422467346524374[55] = 0.0;
   out_3868422467346524374[56] = 0.0;
   out_3868422467346524374[57] = 0.0;
   out_3868422467346524374[58] = 0.0;
   out_3868422467346524374[59] = 0.0;
   out_3868422467346524374[60] = 1.0;
   out_3868422467346524374[61] = 0.0;
   out_3868422467346524374[62] = 0.0;
   out_3868422467346524374[63] = 0.0;
   out_3868422467346524374[64] = 0.0;
   out_3868422467346524374[65] = 0.0;
   out_3868422467346524374[66] = 0.0;
   out_3868422467346524374[67] = 0.0;
   out_3868422467346524374[68] = 0.0;
   out_3868422467346524374[69] = 0.0;
   out_3868422467346524374[70] = 1.0;
   out_3868422467346524374[71] = 0.0;
   out_3868422467346524374[72] = 0.0;
   out_3868422467346524374[73] = 0.0;
   out_3868422467346524374[74] = 0.0;
   out_3868422467346524374[75] = 0.0;
   out_3868422467346524374[76] = 0.0;
   out_3868422467346524374[77] = 0.0;
   out_3868422467346524374[78] = 0.0;
   out_3868422467346524374[79] = 0.0;
   out_3868422467346524374[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_9105879954051592652) {
   out_9105879954051592652[0] = state[0];
   out_9105879954051592652[1] = state[1];
   out_9105879954051592652[2] = state[2];
   out_9105879954051592652[3] = state[3];
   out_9105879954051592652[4] = state[4];
   out_9105879954051592652[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_9105879954051592652[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_9105879954051592652[7] = state[7];
   out_9105879954051592652[8] = state[8];
}
void F_fun(double *state, double dt, double *out_1694872647985395104) {
   out_1694872647985395104[0] = 1;
   out_1694872647985395104[1] = 0;
   out_1694872647985395104[2] = 0;
   out_1694872647985395104[3] = 0;
   out_1694872647985395104[4] = 0;
   out_1694872647985395104[5] = 0;
   out_1694872647985395104[6] = 0;
   out_1694872647985395104[7] = 0;
   out_1694872647985395104[8] = 0;
   out_1694872647985395104[9] = 0;
   out_1694872647985395104[10] = 1;
   out_1694872647985395104[11] = 0;
   out_1694872647985395104[12] = 0;
   out_1694872647985395104[13] = 0;
   out_1694872647985395104[14] = 0;
   out_1694872647985395104[15] = 0;
   out_1694872647985395104[16] = 0;
   out_1694872647985395104[17] = 0;
   out_1694872647985395104[18] = 0;
   out_1694872647985395104[19] = 0;
   out_1694872647985395104[20] = 1;
   out_1694872647985395104[21] = 0;
   out_1694872647985395104[22] = 0;
   out_1694872647985395104[23] = 0;
   out_1694872647985395104[24] = 0;
   out_1694872647985395104[25] = 0;
   out_1694872647985395104[26] = 0;
   out_1694872647985395104[27] = 0;
   out_1694872647985395104[28] = 0;
   out_1694872647985395104[29] = 0;
   out_1694872647985395104[30] = 1;
   out_1694872647985395104[31] = 0;
   out_1694872647985395104[32] = 0;
   out_1694872647985395104[33] = 0;
   out_1694872647985395104[34] = 0;
   out_1694872647985395104[35] = 0;
   out_1694872647985395104[36] = 0;
   out_1694872647985395104[37] = 0;
   out_1694872647985395104[38] = 0;
   out_1694872647985395104[39] = 0;
   out_1694872647985395104[40] = 1;
   out_1694872647985395104[41] = 0;
   out_1694872647985395104[42] = 0;
   out_1694872647985395104[43] = 0;
   out_1694872647985395104[44] = 0;
   out_1694872647985395104[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_1694872647985395104[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_1694872647985395104[47] = -dt*stiffness_front*state[0]/state[1];
   out_1694872647985395104[48] = -dt*stiffness_front*state[0]/state[1];
   out_1694872647985395104[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_1694872647985395104[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_1694872647985395104[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_1694872647985395104[52] = dt*stiffness_front*state[0]/state[1];
   out_1694872647985395104[53] = -9.8100000000000005*dt;
   out_1694872647985395104[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_1694872647985395104[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_1694872647985395104[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1694872647985395104[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1694872647985395104[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_1694872647985395104[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_1694872647985395104[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_1694872647985395104[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1694872647985395104[62] = 0;
   out_1694872647985395104[63] = 0;
   out_1694872647985395104[64] = 0;
   out_1694872647985395104[65] = 0;
   out_1694872647985395104[66] = 0;
   out_1694872647985395104[67] = 0;
   out_1694872647985395104[68] = 0;
   out_1694872647985395104[69] = 0;
   out_1694872647985395104[70] = 1;
   out_1694872647985395104[71] = 0;
   out_1694872647985395104[72] = 0;
   out_1694872647985395104[73] = 0;
   out_1694872647985395104[74] = 0;
   out_1694872647985395104[75] = 0;
   out_1694872647985395104[76] = 0;
   out_1694872647985395104[77] = 0;
   out_1694872647985395104[78] = 0;
   out_1694872647985395104[79] = 0;
   out_1694872647985395104[80] = 1;
}
void h_25(double *state, double *unused, double *out_4995790709594599065) {
   out_4995790709594599065[0] = state[6];
}
void H_25(double *state, double *unused, double *out_345441817357032697) {
   out_345441817357032697[0] = 0;
   out_345441817357032697[1] = 0;
   out_345441817357032697[2] = 0;
   out_345441817357032697[3] = 0;
   out_345441817357032697[4] = 0;
   out_345441817357032697[5] = 0;
   out_345441817357032697[6] = 1;
   out_345441817357032697[7] = 0;
   out_345441817357032697[8] = 0;
}
void h_24(double *state, double *unused, double *out_1084295506565986845) {
   out_1084295506565986845[0] = state[4];
   out_1084295506565986845[1] = state[5];
}
void H_24(double *state, double *unused, double *out_3212483781445290439) {
   out_3212483781445290439[0] = 0;
   out_3212483781445290439[1] = 0;
   out_3212483781445290439[2] = 0;
   out_3212483781445290439[3] = 0;
   out_3212483781445290439[4] = 1;
   out_3212483781445290439[5] = 0;
   out_3212483781445290439[6] = 0;
   out_3212483781445290439[7] = 0;
   out_3212483781445290439[8] = 0;
   out_3212483781445290439[9] = 0;
   out_3212483781445290439[10] = 0;
   out_3212483781445290439[11] = 0;
   out_3212483781445290439[12] = 0;
   out_3212483781445290439[13] = 0;
   out_3212483781445290439[14] = 1;
   out_3212483781445290439[15] = 0;
   out_3212483781445290439[16] = 0;
   out_3212483781445290439[17] = 0;
}
void h_30(double *state, double *unused, double *out_4720596647310093176) {
   out_4720596647310093176[0] = state[4];
}
void H_30(double *state, double *unused, double *out_6571248524134584058) {
   out_6571248524134584058[0] = 0;
   out_6571248524134584058[1] = 0;
   out_6571248524134584058[2] = 0;
   out_6571248524134584058[3] = 0;
   out_6571248524134584058[4] = 1;
   out_6571248524134584058[5] = 0;
   out_6571248524134584058[6] = 0;
   out_6571248524134584058[7] = 0;
   out_6571248524134584058[8] = 0;
}
void h_26(double *state, double *unused, double *out_8724474158738612795) {
   out_8724474158738612795[0] = state[7];
}
void H_26(double *state, double *unused, double *out_4086945136231088921) {
   out_4086945136231088921[0] = 0;
   out_4086945136231088921[1] = 0;
   out_4086945136231088921[2] = 0;
   out_4086945136231088921[3] = 0;
   out_4086945136231088921[4] = 0;
   out_4086945136231088921[5] = 0;
   out_4086945136231088921[6] = 0;
   out_4086945136231088921[7] = 1;
   out_4086945136231088921[8] = 0;
}
void h_27(double *state, double *unused, double *out_5939300562496261500) {
   out_5939300562496261500[0] = state[3];
}
void H_27(double *state, double *unused, double *out_4396485212334159147) {
   out_4396485212334159147[0] = 0;
   out_4396485212334159147[1] = 0;
   out_4396485212334159147[2] = 0;
   out_4396485212334159147[3] = 1;
   out_4396485212334159147[4] = 0;
   out_4396485212334159147[5] = 0;
   out_4396485212334159147[6] = 0;
   out_4396485212334159147[7] = 0;
   out_4396485212334159147[8] = 0;
}
void h_29(double *state, double *unused, double *out_7757382698754425861) {
   out_7757382698754425861[0] = state[1];
}
void H_29(double *state, double *unused, double *out_2683122485464608114) {
   out_2683122485464608114[0] = 0;
   out_2683122485464608114[1] = 1;
   out_2683122485464608114[2] = 0;
   out_2683122485464608114[3] = 0;
   out_2683122485464608114[4] = 0;
   out_2683122485464608114[5] = 0;
   out_2683122485464608114[6] = 0;
   out_2683122485464608114[7] = 0;
   out_2683122485464608114[8] = 0;
}
void h_28(double *state, double *unused, double *out_6108401993799440970) {
   out_6108401993799440970[0] = state[0];
}
void H_28(double *state, double *unused, double *out_2399276531604922460) {
   out_2399276531604922460[0] = 1;
   out_2399276531604922460[1] = 0;
   out_2399276531604922460[2] = 0;
   out_2399276531604922460[3] = 0;
   out_2399276531604922460[4] = 0;
   out_2399276531604922460[5] = 0;
   out_2399276531604922460[6] = 0;
   out_2399276531604922460[7] = 0;
   out_2399276531604922460[8] = 0;
}
void h_31(double *state, double *unused, double *out_4838604041243469920) {
   out_4838604041243469920[0] = state[8];
}
void H_31(double *state, double *unused, double *out_314795855480072269) {
   out_314795855480072269[0] = 0;
   out_314795855480072269[1] = 0;
   out_314795855480072269[2] = 0;
   out_314795855480072269[3] = 0;
   out_314795855480072269[4] = 0;
   out_314795855480072269[5] = 0;
   out_314795855480072269[6] = 0;
   out_314795855480072269[7] = 0;
   out_314795855480072269[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_5852520466696422965) {
  err_fun(nom_x, delta_x, out_5852520466696422965);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_1553841940871946652) {
  inv_err_fun(nom_x, true_x, out_1553841940871946652);
}
void car_H_mod_fun(double *state, double *out_3868422467346524374) {
  H_mod_fun(state, out_3868422467346524374);
}
void car_f_fun(double *state, double dt, double *out_9105879954051592652) {
  f_fun(state,  dt, out_9105879954051592652);
}
void car_F_fun(double *state, double dt, double *out_1694872647985395104) {
  F_fun(state,  dt, out_1694872647985395104);
}
void car_h_25(double *state, double *unused, double *out_4995790709594599065) {
  h_25(state, unused, out_4995790709594599065);
}
void car_H_25(double *state, double *unused, double *out_345441817357032697) {
  H_25(state, unused, out_345441817357032697);
}
void car_h_24(double *state, double *unused, double *out_1084295506565986845) {
  h_24(state, unused, out_1084295506565986845);
}
void car_H_24(double *state, double *unused, double *out_3212483781445290439) {
  H_24(state, unused, out_3212483781445290439);
}
void car_h_30(double *state, double *unused, double *out_4720596647310093176) {
  h_30(state, unused, out_4720596647310093176);
}
void car_H_30(double *state, double *unused, double *out_6571248524134584058) {
  H_30(state, unused, out_6571248524134584058);
}
void car_h_26(double *state, double *unused, double *out_8724474158738612795) {
  h_26(state, unused, out_8724474158738612795);
}
void car_H_26(double *state, double *unused, double *out_4086945136231088921) {
  H_26(state, unused, out_4086945136231088921);
}
void car_h_27(double *state, double *unused, double *out_5939300562496261500) {
  h_27(state, unused, out_5939300562496261500);
}
void car_H_27(double *state, double *unused, double *out_4396485212334159147) {
  H_27(state, unused, out_4396485212334159147);
}
void car_h_29(double *state, double *unused, double *out_7757382698754425861) {
  h_29(state, unused, out_7757382698754425861);
}
void car_H_29(double *state, double *unused, double *out_2683122485464608114) {
  H_29(state, unused, out_2683122485464608114);
}
void car_h_28(double *state, double *unused, double *out_6108401993799440970) {
  h_28(state, unused, out_6108401993799440970);
}
void car_H_28(double *state, double *unused, double *out_2399276531604922460) {
  H_28(state, unused, out_2399276531604922460);
}
void car_h_31(double *state, double *unused, double *out_4838604041243469920) {
  h_31(state, unused, out_4838604041243469920);
}
void car_H_31(double *state, double *unused, double *out_314795855480072269) {
  H_31(state, unused, out_314795855480072269);
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
