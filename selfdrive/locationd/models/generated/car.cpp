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
 *                      Code generated with SymPy 1.13.2                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_6387606906665158225) {
   out_6387606906665158225[0] = delta_x[0] + nom_x[0];
   out_6387606906665158225[1] = delta_x[1] + nom_x[1];
   out_6387606906665158225[2] = delta_x[2] + nom_x[2];
   out_6387606906665158225[3] = delta_x[3] + nom_x[3];
   out_6387606906665158225[4] = delta_x[4] + nom_x[4];
   out_6387606906665158225[5] = delta_x[5] + nom_x[5];
   out_6387606906665158225[6] = delta_x[6] + nom_x[6];
   out_6387606906665158225[7] = delta_x[7] + nom_x[7];
   out_6387606906665158225[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_1940653918789291055) {
   out_1940653918789291055[0] = -nom_x[0] + true_x[0];
   out_1940653918789291055[1] = -nom_x[1] + true_x[1];
   out_1940653918789291055[2] = -nom_x[2] + true_x[2];
   out_1940653918789291055[3] = -nom_x[3] + true_x[3];
   out_1940653918789291055[4] = -nom_x[4] + true_x[4];
   out_1940653918789291055[5] = -nom_x[5] + true_x[5];
   out_1940653918789291055[6] = -nom_x[6] + true_x[6];
   out_1940653918789291055[7] = -nom_x[7] + true_x[7];
   out_1940653918789291055[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_4168849259471386666) {
   out_4168849259471386666[0] = 1.0;
   out_4168849259471386666[1] = 0.0;
   out_4168849259471386666[2] = 0.0;
   out_4168849259471386666[3] = 0.0;
   out_4168849259471386666[4] = 0.0;
   out_4168849259471386666[5] = 0.0;
   out_4168849259471386666[6] = 0.0;
   out_4168849259471386666[7] = 0.0;
   out_4168849259471386666[8] = 0.0;
   out_4168849259471386666[9] = 0.0;
   out_4168849259471386666[10] = 1.0;
   out_4168849259471386666[11] = 0.0;
   out_4168849259471386666[12] = 0.0;
   out_4168849259471386666[13] = 0.0;
   out_4168849259471386666[14] = 0.0;
   out_4168849259471386666[15] = 0.0;
   out_4168849259471386666[16] = 0.0;
   out_4168849259471386666[17] = 0.0;
   out_4168849259471386666[18] = 0.0;
   out_4168849259471386666[19] = 0.0;
   out_4168849259471386666[20] = 1.0;
   out_4168849259471386666[21] = 0.0;
   out_4168849259471386666[22] = 0.0;
   out_4168849259471386666[23] = 0.0;
   out_4168849259471386666[24] = 0.0;
   out_4168849259471386666[25] = 0.0;
   out_4168849259471386666[26] = 0.0;
   out_4168849259471386666[27] = 0.0;
   out_4168849259471386666[28] = 0.0;
   out_4168849259471386666[29] = 0.0;
   out_4168849259471386666[30] = 1.0;
   out_4168849259471386666[31] = 0.0;
   out_4168849259471386666[32] = 0.0;
   out_4168849259471386666[33] = 0.0;
   out_4168849259471386666[34] = 0.0;
   out_4168849259471386666[35] = 0.0;
   out_4168849259471386666[36] = 0.0;
   out_4168849259471386666[37] = 0.0;
   out_4168849259471386666[38] = 0.0;
   out_4168849259471386666[39] = 0.0;
   out_4168849259471386666[40] = 1.0;
   out_4168849259471386666[41] = 0.0;
   out_4168849259471386666[42] = 0.0;
   out_4168849259471386666[43] = 0.0;
   out_4168849259471386666[44] = 0.0;
   out_4168849259471386666[45] = 0.0;
   out_4168849259471386666[46] = 0.0;
   out_4168849259471386666[47] = 0.0;
   out_4168849259471386666[48] = 0.0;
   out_4168849259471386666[49] = 0.0;
   out_4168849259471386666[50] = 1.0;
   out_4168849259471386666[51] = 0.0;
   out_4168849259471386666[52] = 0.0;
   out_4168849259471386666[53] = 0.0;
   out_4168849259471386666[54] = 0.0;
   out_4168849259471386666[55] = 0.0;
   out_4168849259471386666[56] = 0.0;
   out_4168849259471386666[57] = 0.0;
   out_4168849259471386666[58] = 0.0;
   out_4168849259471386666[59] = 0.0;
   out_4168849259471386666[60] = 1.0;
   out_4168849259471386666[61] = 0.0;
   out_4168849259471386666[62] = 0.0;
   out_4168849259471386666[63] = 0.0;
   out_4168849259471386666[64] = 0.0;
   out_4168849259471386666[65] = 0.0;
   out_4168849259471386666[66] = 0.0;
   out_4168849259471386666[67] = 0.0;
   out_4168849259471386666[68] = 0.0;
   out_4168849259471386666[69] = 0.0;
   out_4168849259471386666[70] = 1.0;
   out_4168849259471386666[71] = 0.0;
   out_4168849259471386666[72] = 0.0;
   out_4168849259471386666[73] = 0.0;
   out_4168849259471386666[74] = 0.0;
   out_4168849259471386666[75] = 0.0;
   out_4168849259471386666[76] = 0.0;
   out_4168849259471386666[77] = 0.0;
   out_4168849259471386666[78] = 0.0;
   out_4168849259471386666[79] = 0.0;
   out_4168849259471386666[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_7308117309923687076) {
   out_7308117309923687076[0] = state[0];
   out_7308117309923687076[1] = state[1];
   out_7308117309923687076[2] = state[2];
   out_7308117309923687076[3] = state[3];
   out_7308117309923687076[4] = state[4];
   out_7308117309923687076[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_7308117309923687076[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_7308117309923687076[7] = state[7];
   out_7308117309923687076[8] = state[8];
}
void F_fun(double *state, double dt, double *out_7708045199441947550) {
   out_7708045199441947550[0] = 1;
   out_7708045199441947550[1] = 0;
   out_7708045199441947550[2] = 0;
   out_7708045199441947550[3] = 0;
   out_7708045199441947550[4] = 0;
   out_7708045199441947550[5] = 0;
   out_7708045199441947550[6] = 0;
   out_7708045199441947550[7] = 0;
   out_7708045199441947550[8] = 0;
   out_7708045199441947550[9] = 0;
   out_7708045199441947550[10] = 1;
   out_7708045199441947550[11] = 0;
   out_7708045199441947550[12] = 0;
   out_7708045199441947550[13] = 0;
   out_7708045199441947550[14] = 0;
   out_7708045199441947550[15] = 0;
   out_7708045199441947550[16] = 0;
   out_7708045199441947550[17] = 0;
   out_7708045199441947550[18] = 0;
   out_7708045199441947550[19] = 0;
   out_7708045199441947550[20] = 1;
   out_7708045199441947550[21] = 0;
   out_7708045199441947550[22] = 0;
   out_7708045199441947550[23] = 0;
   out_7708045199441947550[24] = 0;
   out_7708045199441947550[25] = 0;
   out_7708045199441947550[26] = 0;
   out_7708045199441947550[27] = 0;
   out_7708045199441947550[28] = 0;
   out_7708045199441947550[29] = 0;
   out_7708045199441947550[30] = 1;
   out_7708045199441947550[31] = 0;
   out_7708045199441947550[32] = 0;
   out_7708045199441947550[33] = 0;
   out_7708045199441947550[34] = 0;
   out_7708045199441947550[35] = 0;
   out_7708045199441947550[36] = 0;
   out_7708045199441947550[37] = 0;
   out_7708045199441947550[38] = 0;
   out_7708045199441947550[39] = 0;
   out_7708045199441947550[40] = 1;
   out_7708045199441947550[41] = 0;
   out_7708045199441947550[42] = 0;
   out_7708045199441947550[43] = 0;
   out_7708045199441947550[44] = 0;
   out_7708045199441947550[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_7708045199441947550[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_7708045199441947550[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_7708045199441947550[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_7708045199441947550[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_7708045199441947550[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_7708045199441947550[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_7708045199441947550[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_7708045199441947550[53] = -9.8000000000000007*dt;
   out_7708045199441947550[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_7708045199441947550[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_7708045199441947550[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7708045199441947550[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7708045199441947550[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_7708045199441947550[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_7708045199441947550[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_7708045199441947550[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7708045199441947550[62] = 0;
   out_7708045199441947550[63] = 0;
   out_7708045199441947550[64] = 0;
   out_7708045199441947550[65] = 0;
   out_7708045199441947550[66] = 0;
   out_7708045199441947550[67] = 0;
   out_7708045199441947550[68] = 0;
   out_7708045199441947550[69] = 0;
   out_7708045199441947550[70] = 1;
   out_7708045199441947550[71] = 0;
   out_7708045199441947550[72] = 0;
   out_7708045199441947550[73] = 0;
   out_7708045199441947550[74] = 0;
   out_7708045199441947550[75] = 0;
   out_7708045199441947550[76] = 0;
   out_7708045199441947550[77] = 0;
   out_7708045199441947550[78] = 0;
   out_7708045199441947550[79] = 0;
   out_7708045199441947550[80] = 1;
}
void h_25(double *state, double *unused, double *out_8572254133923072558) {
   out_8572254133923072558[0] = state[6];
}
void H_25(double *state, double *unused, double *out_8963680102399255775) {
   out_8963680102399255775[0] = 0;
   out_8963680102399255775[1] = 0;
   out_8963680102399255775[2] = 0;
   out_8963680102399255775[3] = 0;
   out_8963680102399255775[4] = 0;
   out_8963680102399255775[5] = 0;
   out_8963680102399255775[6] = 1;
   out_8963680102399255775[7] = 0;
   out_8963680102399255775[8] = 0;
}
void h_24(double *state, double *unused, double *out_4030352194828355617) {
   out_4030352194828355617[0] = state[4];
   out_4030352194828355617[1] = state[5];
}
void H_24(double *state, double *unused, double *out_7310414372304796275) {
   out_7310414372304796275[0] = 0;
   out_7310414372304796275[1] = 0;
   out_7310414372304796275[2] = 0;
   out_7310414372304796275[3] = 0;
   out_7310414372304796275[4] = 1;
   out_7310414372304796275[5] = 0;
   out_7310414372304796275[6] = 0;
   out_7310414372304796275[7] = 0;
   out_7310414372304796275[8] = 0;
   out_7310414372304796275[9] = 0;
   out_7310414372304796275[10] = 0;
   out_7310414372304796275[11] = 0;
   out_7310414372304796275[12] = 0;
   out_7310414372304796275[13] = 0;
   out_7310414372304796275[14] = 1;
   out_7310414372304796275[15] = 0;
   out_7310414372304796275[16] = 0;
   out_7310414372304796275[17] = 0;
}
void h_30(double *state, double *unused, double *out_6926763949430994097) {
   out_6926763949430994097[0] = state[4];
}
void H_30(double *state, double *unused, double *out_6445347143892007148) {
   out_6445347143892007148[0] = 0;
   out_6445347143892007148[1] = 0;
   out_6445347143892007148[2] = 0;
   out_6445347143892007148[3] = 0;
   out_6445347143892007148[4] = 1;
   out_6445347143892007148[5] = 0;
   out_6445347143892007148[6] = 0;
   out_6445347143892007148[7] = 0;
   out_6445347143892007148[8] = 0;
}
void h_26(double *state, double *unused, double *out_5983254096529331662) {
   out_5983254096529331662[0] = state[7];
}
void H_26(double *state, double *unused, double *out_5741560652436239617) {
   out_5741560652436239617[0] = 0;
   out_5741560652436239617[1] = 0;
   out_5741560652436239617[2] = 0;
   out_5741560652436239617[3] = 0;
   out_5741560652436239617[4] = 0;
   out_5741560652436239617[5] = 0;
   out_5741560652436239617[6] = 0;
   out_5741560652436239617[7] = 1;
   out_5741560652436239617[8] = 0;
}
void h_27(double *state, double *unused, double *out_6598642127039022469) {
   out_6598642127039022469[0] = state[3];
}
void H_27(double *state, double *unused, double *out_2780604329382262732) {
   out_2780604329382262732[0] = 0;
   out_2780604329382262732[1] = 0;
   out_2780604329382262732[2] = 0;
   out_2780604329382262732[3] = 1;
   out_2780604329382262732[4] = 0;
   out_2780604329382262732[5] = 0;
   out_2780604329382262732[6] = 0;
   out_2780604329382262732[7] = 0;
   out_2780604329382262732[8] = 0;
}
void h_29(double *state, double *unused, double *out_6323448064754516580) {
   out_6323448064754516580[0] = state[1];
}
void H_29(double *state, double *unused, double *out_5935115799577614964) {
   out_5935115799577614964[0] = 0;
   out_5935115799577614964[1] = 1;
   out_5935115799577614964[2] = 0;
   out_5935115799577614964[3] = 0;
   out_5935115799577614964[4] = 0;
   out_5935115799577614964[5] = 0;
   out_5935115799577614964[6] = 0;
   out_5935115799577614964[7] = 0;
   out_5935115799577614964[8] = 0;
}
void h_28(double *state, double *unused, double *out_5449060429256680001) {
   out_5449060429256680001[0] = state[0];
}
void H_28(double *state, double *unused, double *out_7429229257062406078) {
   out_7429229257062406078[0] = 1;
   out_7429229257062406078[1] = 0;
   out_7429229257062406078[2] = 0;
   out_7429229257062406078[3] = 0;
   out_7429229257062406078[4] = 0;
   out_7429229257062406078[5] = 0;
   out_7429229257062406078[6] = 0;
   out_7429229257062406078[7] = 0;
   out_7429229257062406078[8] = 0;
}
void h_31(double *state, double *unused, double *out_3103654713436128122) {
   out_3103654713436128122[0] = state[8];
}
void H_31(double *state, double *unused, double *out_5115352550202888141) {
   out_5115352550202888141[0] = 0;
   out_5115352550202888141[1] = 0;
   out_5115352550202888141[2] = 0;
   out_5115352550202888141[3] = 0;
   out_5115352550202888141[4] = 0;
   out_5115352550202888141[5] = 0;
   out_5115352550202888141[6] = 0;
   out_5115352550202888141[7] = 0;
   out_5115352550202888141[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_6387606906665158225) {
  err_fun(nom_x, delta_x, out_6387606906665158225);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_1940653918789291055) {
  inv_err_fun(nom_x, true_x, out_1940653918789291055);
}
void car_H_mod_fun(double *state, double *out_4168849259471386666) {
  H_mod_fun(state, out_4168849259471386666);
}
void car_f_fun(double *state, double dt, double *out_7308117309923687076) {
  f_fun(state,  dt, out_7308117309923687076);
}
void car_F_fun(double *state, double dt, double *out_7708045199441947550) {
  F_fun(state,  dt, out_7708045199441947550);
}
void car_h_25(double *state, double *unused, double *out_8572254133923072558) {
  h_25(state, unused, out_8572254133923072558);
}
void car_H_25(double *state, double *unused, double *out_8963680102399255775) {
  H_25(state, unused, out_8963680102399255775);
}
void car_h_24(double *state, double *unused, double *out_4030352194828355617) {
  h_24(state, unused, out_4030352194828355617);
}
void car_H_24(double *state, double *unused, double *out_7310414372304796275) {
  H_24(state, unused, out_7310414372304796275);
}
void car_h_30(double *state, double *unused, double *out_6926763949430994097) {
  h_30(state, unused, out_6926763949430994097);
}
void car_H_30(double *state, double *unused, double *out_6445347143892007148) {
  H_30(state, unused, out_6445347143892007148);
}
void car_h_26(double *state, double *unused, double *out_5983254096529331662) {
  h_26(state, unused, out_5983254096529331662);
}
void car_H_26(double *state, double *unused, double *out_5741560652436239617) {
  H_26(state, unused, out_5741560652436239617);
}
void car_h_27(double *state, double *unused, double *out_6598642127039022469) {
  h_27(state, unused, out_6598642127039022469);
}
void car_H_27(double *state, double *unused, double *out_2780604329382262732) {
  H_27(state, unused, out_2780604329382262732);
}
void car_h_29(double *state, double *unused, double *out_6323448064754516580) {
  h_29(state, unused, out_6323448064754516580);
}
void car_H_29(double *state, double *unused, double *out_5935115799577614964) {
  H_29(state, unused, out_5935115799577614964);
}
void car_h_28(double *state, double *unused, double *out_5449060429256680001) {
  h_28(state, unused, out_5449060429256680001);
}
void car_H_28(double *state, double *unused, double *out_7429229257062406078) {
  H_28(state, unused, out_7429229257062406078);
}
void car_h_31(double *state, double *unused, double *out_3103654713436128122) {
  h_31(state, unused, out_3103654713436128122);
}
void car_H_31(double *state, double *unused, double *out_5115352550202888141) {
  H_31(state, unused, out_5115352550202888141);
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
