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
void err_fun(double *nom_x, double *delta_x, double *out_23523217466685504) {
   out_23523217466685504[0] = delta_x[0] + nom_x[0];
   out_23523217466685504[1] = delta_x[1] + nom_x[1];
   out_23523217466685504[2] = delta_x[2] + nom_x[2];
   out_23523217466685504[3] = delta_x[3] + nom_x[3];
   out_23523217466685504[4] = delta_x[4] + nom_x[4];
   out_23523217466685504[5] = delta_x[5] + nom_x[5];
   out_23523217466685504[6] = delta_x[6] + nom_x[6];
   out_23523217466685504[7] = delta_x[7] + nom_x[7];
   out_23523217466685504[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_490723024718224285) {
   out_490723024718224285[0] = -nom_x[0] + true_x[0];
   out_490723024718224285[1] = -nom_x[1] + true_x[1];
   out_490723024718224285[2] = -nom_x[2] + true_x[2];
   out_490723024718224285[3] = -nom_x[3] + true_x[3];
   out_490723024718224285[4] = -nom_x[4] + true_x[4];
   out_490723024718224285[5] = -nom_x[5] + true_x[5];
   out_490723024718224285[6] = -nom_x[6] + true_x[6];
   out_490723024718224285[7] = -nom_x[7] + true_x[7];
   out_490723024718224285[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_4030678833409053918) {
   out_4030678833409053918[0] = 1.0;
   out_4030678833409053918[1] = 0.0;
   out_4030678833409053918[2] = 0.0;
   out_4030678833409053918[3] = 0.0;
   out_4030678833409053918[4] = 0.0;
   out_4030678833409053918[5] = 0.0;
   out_4030678833409053918[6] = 0.0;
   out_4030678833409053918[7] = 0.0;
   out_4030678833409053918[8] = 0.0;
   out_4030678833409053918[9] = 0.0;
   out_4030678833409053918[10] = 1.0;
   out_4030678833409053918[11] = 0.0;
   out_4030678833409053918[12] = 0.0;
   out_4030678833409053918[13] = 0.0;
   out_4030678833409053918[14] = 0.0;
   out_4030678833409053918[15] = 0.0;
   out_4030678833409053918[16] = 0.0;
   out_4030678833409053918[17] = 0.0;
   out_4030678833409053918[18] = 0.0;
   out_4030678833409053918[19] = 0.0;
   out_4030678833409053918[20] = 1.0;
   out_4030678833409053918[21] = 0.0;
   out_4030678833409053918[22] = 0.0;
   out_4030678833409053918[23] = 0.0;
   out_4030678833409053918[24] = 0.0;
   out_4030678833409053918[25] = 0.0;
   out_4030678833409053918[26] = 0.0;
   out_4030678833409053918[27] = 0.0;
   out_4030678833409053918[28] = 0.0;
   out_4030678833409053918[29] = 0.0;
   out_4030678833409053918[30] = 1.0;
   out_4030678833409053918[31] = 0.0;
   out_4030678833409053918[32] = 0.0;
   out_4030678833409053918[33] = 0.0;
   out_4030678833409053918[34] = 0.0;
   out_4030678833409053918[35] = 0.0;
   out_4030678833409053918[36] = 0.0;
   out_4030678833409053918[37] = 0.0;
   out_4030678833409053918[38] = 0.0;
   out_4030678833409053918[39] = 0.0;
   out_4030678833409053918[40] = 1.0;
   out_4030678833409053918[41] = 0.0;
   out_4030678833409053918[42] = 0.0;
   out_4030678833409053918[43] = 0.0;
   out_4030678833409053918[44] = 0.0;
   out_4030678833409053918[45] = 0.0;
   out_4030678833409053918[46] = 0.0;
   out_4030678833409053918[47] = 0.0;
   out_4030678833409053918[48] = 0.0;
   out_4030678833409053918[49] = 0.0;
   out_4030678833409053918[50] = 1.0;
   out_4030678833409053918[51] = 0.0;
   out_4030678833409053918[52] = 0.0;
   out_4030678833409053918[53] = 0.0;
   out_4030678833409053918[54] = 0.0;
   out_4030678833409053918[55] = 0.0;
   out_4030678833409053918[56] = 0.0;
   out_4030678833409053918[57] = 0.0;
   out_4030678833409053918[58] = 0.0;
   out_4030678833409053918[59] = 0.0;
   out_4030678833409053918[60] = 1.0;
   out_4030678833409053918[61] = 0.0;
   out_4030678833409053918[62] = 0.0;
   out_4030678833409053918[63] = 0.0;
   out_4030678833409053918[64] = 0.0;
   out_4030678833409053918[65] = 0.0;
   out_4030678833409053918[66] = 0.0;
   out_4030678833409053918[67] = 0.0;
   out_4030678833409053918[68] = 0.0;
   out_4030678833409053918[69] = 0.0;
   out_4030678833409053918[70] = 1.0;
   out_4030678833409053918[71] = 0.0;
   out_4030678833409053918[72] = 0.0;
   out_4030678833409053918[73] = 0.0;
   out_4030678833409053918[74] = 0.0;
   out_4030678833409053918[75] = 0.0;
   out_4030678833409053918[76] = 0.0;
   out_4030678833409053918[77] = 0.0;
   out_4030678833409053918[78] = 0.0;
   out_4030678833409053918[79] = 0.0;
   out_4030678833409053918[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_7376853175200153600) {
   out_7376853175200153600[0] = state[0];
   out_7376853175200153600[1] = state[1];
   out_7376853175200153600[2] = state[2];
   out_7376853175200153600[3] = state[3];
   out_7376853175200153600[4] = state[4];
   out_7376853175200153600[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_7376853175200153600[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_7376853175200153600[7] = state[7];
   out_7376853175200153600[8] = state[8];
}
void F_fun(double *state, double dt, double *out_732326874719211251) {
   out_732326874719211251[0] = 1;
   out_732326874719211251[1] = 0;
   out_732326874719211251[2] = 0;
   out_732326874719211251[3] = 0;
   out_732326874719211251[4] = 0;
   out_732326874719211251[5] = 0;
   out_732326874719211251[6] = 0;
   out_732326874719211251[7] = 0;
   out_732326874719211251[8] = 0;
   out_732326874719211251[9] = 0;
   out_732326874719211251[10] = 1;
   out_732326874719211251[11] = 0;
   out_732326874719211251[12] = 0;
   out_732326874719211251[13] = 0;
   out_732326874719211251[14] = 0;
   out_732326874719211251[15] = 0;
   out_732326874719211251[16] = 0;
   out_732326874719211251[17] = 0;
   out_732326874719211251[18] = 0;
   out_732326874719211251[19] = 0;
   out_732326874719211251[20] = 1;
   out_732326874719211251[21] = 0;
   out_732326874719211251[22] = 0;
   out_732326874719211251[23] = 0;
   out_732326874719211251[24] = 0;
   out_732326874719211251[25] = 0;
   out_732326874719211251[26] = 0;
   out_732326874719211251[27] = 0;
   out_732326874719211251[28] = 0;
   out_732326874719211251[29] = 0;
   out_732326874719211251[30] = 1;
   out_732326874719211251[31] = 0;
   out_732326874719211251[32] = 0;
   out_732326874719211251[33] = 0;
   out_732326874719211251[34] = 0;
   out_732326874719211251[35] = 0;
   out_732326874719211251[36] = 0;
   out_732326874719211251[37] = 0;
   out_732326874719211251[38] = 0;
   out_732326874719211251[39] = 0;
   out_732326874719211251[40] = 1;
   out_732326874719211251[41] = 0;
   out_732326874719211251[42] = 0;
   out_732326874719211251[43] = 0;
   out_732326874719211251[44] = 0;
   out_732326874719211251[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_732326874719211251[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_732326874719211251[47] = -dt*stiffness_front*state[0]/state[1];
   out_732326874719211251[48] = -dt*stiffness_front*state[0]/state[1];
   out_732326874719211251[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_732326874719211251[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_732326874719211251[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_732326874719211251[52] = dt*stiffness_front*state[0]/state[1];
   out_732326874719211251[53] = -9.8100000000000005*dt;
   out_732326874719211251[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_732326874719211251[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_732326874719211251[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_732326874719211251[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_732326874719211251[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_732326874719211251[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_732326874719211251[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_732326874719211251[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_732326874719211251[62] = 0;
   out_732326874719211251[63] = 0;
   out_732326874719211251[64] = 0;
   out_732326874719211251[65] = 0;
   out_732326874719211251[66] = 0;
   out_732326874719211251[67] = 0;
   out_732326874719211251[68] = 0;
   out_732326874719211251[69] = 0;
   out_732326874719211251[70] = 1;
   out_732326874719211251[71] = 0;
   out_732326874719211251[72] = 0;
   out_732326874719211251[73] = 0;
   out_732326874719211251[74] = 0;
   out_732326874719211251[75] = 0;
   out_732326874719211251[76] = 0;
   out_732326874719211251[77] = 0;
   out_732326874719211251[78] = 0;
   out_732326874719211251[79] = 0;
   out_732326874719211251[80] = 1;
}
void h_25(double *state, double *unused, double *out_1982449565641168842) {
   out_1982449565641168842[0] = state[6];
}
void H_25(double *state, double *unused, double *out_5162509392503183319) {
   out_5162509392503183319[0] = 0;
   out_5162509392503183319[1] = 0;
   out_5162509392503183319[2] = 0;
   out_5162509392503183319[3] = 0;
   out_5162509392503183319[4] = 0;
   out_5162509392503183319[5] = 0;
   out_5162509392503183319[6] = 1;
   out_5162509392503183319[7] = 0;
   out_5162509392503183319[8] = 0;
}
void h_24(double *state, double *unused, double *out_5761352688166713048) {
   out_5761352688166713048[0] = state[4];
   out_5761352688166713048[1] = state[5];
}
void H_24(double *state, double *unused, double *out_8537725354239859263) {
   out_8537725354239859263[0] = 0;
   out_8537725354239859263[1] = 0;
   out_8537725354239859263[2] = 0;
   out_8537725354239859263[3] = 0;
   out_8537725354239859263[4] = 1;
   out_8537725354239859263[5] = 0;
   out_8537725354239859263[6] = 0;
   out_8537725354239859263[7] = 0;
   out_8537725354239859263[8] = 0;
   out_8537725354239859263[9] = 0;
   out_8537725354239859263[10] = 0;
   out_8537725354239859263[11] = 0;
   out_8537725354239859263[12] = 0;
   out_8537725354239859263[13] = 0;
   out_8537725354239859263[14] = 1;
   out_8537725354239859263[15] = 0;
   out_8537725354239859263[16] = 0;
   out_8537725354239859263[17] = 0;
}
void h_30(double *state, double *unused, double *out_2257643627925674731) {
   out_2257643627925674731[0] = state[4];
}
void H_30(double *state, double *unused, double *out_1754180948988433436) {
   out_1754180948988433436[0] = 0;
   out_1754180948988433436[1] = 0;
   out_1754180948988433436[2] = 0;
   out_1754180948988433436[3] = 0;
   out_1754180948988433436[4] = 1;
   out_1754180948988433436[5] = 0;
   out_1754180948988433436[6] = 0;
   out_1754180948988433436[7] = 0;
   out_1754180948988433436[8] = 0;
}
void h_26(double *state, double *unused, double *out_3425387429001816468) {
   out_3425387429001816468[0] = state[7];
}
void H_26(double *state, double *unused, double *out_8904012711377239543) {
   out_8904012711377239543[0] = 0;
   out_8904012711377239543[1] = 0;
   out_8904012711377239543[2] = 0;
   out_8904012711377239543[3] = 0;
   out_8904012711377239543[4] = 0;
   out_8904012711377239543[5] = 0;
   out_8904012711377239543[6] = 0;
   out_8904012711377239543[7] = 1;
   out_8904012711377239543[8] = 0;
}
void h_27(double *state, double *unused, double *out_7338031019540344707) {
   out_7338031019540344707[0] = state[3];
}
void H_27(double *state, double *unused, double *out_420582362811991475) {
   out_420582362811991475[0] = 0;
   out_420582362811991475[1] = 0;
   out_420582362811991475[2] = 0;
   out_420582362811991475[3] = 1;
   out_420582362811991475[4] = 0;
   out_420582362811991475[5] = 0;
   out_420582362811991475[6] = 0;
   out_420582362811991475[7] = 0;
   out_420582362811991475[8] = 0;
}
void h_29(double *state, double *unused, double *out_4953265858328546780) {
   out_4953265858328546780[0] = state[1];
}
void H_29(double *state, double *unused, double *out_2133945089681542508) {
   out_2133945089681542508[0] = 0;
   out_2133945089681542508[1] = 1;
   out_2133945089681542508[2] = 0;
   out_2133945089681542508[3] = 0;
   out_2133945089681542508[4] = 0;
   out_2133945089681542508[5] = 0;
   out_2133945089681542508[6] = 0;
   out_2133945089681542508[7] = 0;
   out_2133945089681542508[8] = 0;
}
void h_28(double *state, double *unused, double *out_6107039786508361264) {
   out_6107039786508361264[0] = state[0];
}
void H_28(double *state, double *unused, double *out_7216344106751073082) {
   out_7216344106751073082[0] = 1;
   out_7216344106751073082[1] = 0;
   out_7216344106751073082[2] = 0;
   out_7216344106751073082[3] = 0;
   out_7216344106751073082[4] = 0;
   out_7216344106751073082[5] = 0;
   out_7216344106751073082[6] = 0;
   out_7216344106751073082[7] = 0;
   out_7216344106751073082[8] = 0;
}
void h_31(double *state, double *unused, double *out_6080752863920540706) {
   out_6080752863920540706[0] = state[8];
}
void H_31(double *state, double *unused, double *out_5131863430626222891) {
   out_5131863430626222891[0] = 0;
   out_5131863430626222891[1] = 0;
   out_5131863430626222891[2] = 0;
   out_5131863430626222891[3] = 0;
   out_5131863430626222891[4] = 0;
   out_5131863430626222891[5] = 0;
   out_5131863430626222891[6] = 0;
   out_5131863430626222891[7] = 0;
   out_5131863430626222891[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_23523217466685504) {
  err_fun(nom_x, delta_x, out_23523217466685504);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_490723024718224285) {
  inv_err_fun(nom_x, true_x, out_490723024718224285);
}
void car_H_mod_fun(double *state, double *out_4030678833409053918) {
  H_mod_fun(state, out_4030678833409053918);
}
void car_f_fun(double *state, double dt, double *out_7376853175200153600) {
  f_fun(state,  dt, out_7376853175200153600);
}
void car_F_fun(double *state, double dt, double *out_732326874719211251) {
  F_fun(state,  dt, out_732326874719211251);
}
void car_h_25(double *state, double *unused, double *out_1982449565641168842) {
  h_25(state, unused, out_1982449565641168842);
}
void car_H_25(double *state, double *unused, double *out_5162509392503183319) {
  H_25(state, unused, out_5162509392503183319);
}
void car_h_24(double *state, double *unused, double *out_5761352688166713048) {
  h_24(state, unused, out_5761352688166713048);
}
void car_H_24(double *state, double *unused, double *out_8537725354239859263) {
  H_24(state, unused, out_8537725354239859263);
}
void car_h_30(double *state, double *unused, double *out_2257643627925674731) {
  h_30(state, unused, out_2257643627925674731);
}
void car_H_30(double *state, double *unused, double *out_1754180948988433436) {
  H_30(state, unused, out_1754180948988433436);
}
void car_h_26(double *state, double *unused, double *out_3425387429001816468) {
  h_26(state, unused, out_3425387429001816468);
}
void car_H_26(double *state, double *unused, double *out_8904012711377239543) {
  H_26(state, unused, out_8904012711377239543);
}
void car_h_27(double *state, double *unused, double *out_7338031019540344707) {
  h_27(state, unused, out_7338031019540344707);
}
void car_H_27(double *state, double *unused, double *out_420582362811991475) {
  H_27(state, unused, out_420582362811991475);
}
void car_h_29(double *state, double *unused, double *out_4953265858328546780) {
  h_29(state, unused, out_4953265858328546780);
}
void car_H_29(double *state, double *unused, double *out_2133945089681542508) {
  H_29(state, unused, out_2133945089681542508);
}
void car_h_28(double *state, double *unused, double *out_6107039786508361264) {
  h_28(state, unused, out_6107039786508361264);
}
void car_H_28(double *state, double *unused, double *out_7216344106751073082) {
  H_28(state, unused, out_7216344106751073082);
}
void car_h_31(double *state, double *unused, double *out_6080752863920540706) {
  h_31(state, unused, out_6080752863920540706);
}
void car_H_31(double *state, double *unused, double *out_5131863430626222891) {
  H_31(state, unused, out_5131863430626222891);
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
