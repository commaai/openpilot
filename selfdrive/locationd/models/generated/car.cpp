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
void err_fun(double *nom_x, double *delta_x, double *out_2868104873166063940) {
   out_2868104873166063940[0] = delta_x[0] + nom_x[0];
   out_2868104873166063940[1] = delta_x[1] + nom_x[1];
   out_2868104873166063940[2] = delta_x[2] + nom_x[2];
   out_2868104873166063940[3] = delta_x[3] + nom_x[3];
   out_2868104873166063940[4] = delta_x[4] + nom_x[4];
   out_2868104873166063940[5] = delta_x[5] + nom_x[5];
   out_2868104873166063940[6] = delta_x[6] + nom_x[6];
   out_2868104873166063940[7] = delta_x[7] + nom_x[7];
   out_2868104873166063940[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_8159495437129412069) {
   out_8159495437129412069[0] = -nom_x[0] + true_x[0];
   out_8159495437129412069[1] = -nom_x[1] + true_x[1];
   out_8159495437129412069[2] = -nom_x[2] + true_x[2];
   out_8159495437129412069[3] = -nom_x[3] + true_x[3];
   out_8159495437129412069[4] = -nom_x[4] + true_x[4];
   out_8159495437129412069[5] = -nom_x[5] + true_x[5];
   out_8159495437129412069[6] = -nom_x[6] + true_x[6];
   out_8159495437129412069[7] = -nom_x[7] + true_x[7];
   out_8159495437129412069[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_8071738448770388550) {
   out_8071738448770388550[0] = 1.0;
   out_8071738448770388550[1] = 0.0;
   out_8071738448770388550[2] = 0.0;
   out_8071738448770388550[3] = 0.0;
   out_8071738448770388550[4] = 0.0;
   out_8071738448770388550[5] = 0.0;
   out_8071738448770388550[6] = 0.0;
   out_8071738448770388550[7] = 0.0;
   out_8071738448770388550[8] = 0.0;
   out_8071738448770388550[9] = 0.0;
   out_8071738448770388550[10] = 1.0;
   out_8071738448770388550[11] = 0.0;
   out_8071738448770388550[12] = 0.0;
   out_8071738448770388550[13] = 0.0;
   out_8071738448770388550[14] = 0.0;
   out_8071738448770388550[15] = 0.0;
   out_8071738448770388550[16] = 0.0;
   out_8071738448770388550[17] = 0.0;
   out_8071738448770388550[18] = 0.0;
   out_8071738448770388550[19] = 0.0;
   out_8071738448770388550[20] = 1.0;
   out_8071738448770388550[21] = 0.0;
   out_8071738448770388550[22] = 0.0;
   out_8071738448770388550[23] = 0.0;
   out_8071738448770388550[24] = 0.0;
   out_8071738448770388550[25] = 0.0;
   out_8071738448770388550[26] = 0.0;
   out_8071738448770388550[27] = 0.0;
   out_8071738448770388550[28] = 0.0;
   out_8071738448770388550[29] = 0.0;
   out_8071738448770388550[30] = 1.0;
   out_8071738448770388550[31] = 0.0;
   out_8071738448770388550[32] = 0.0;
   out_8071738448770388550[33] = 0.0;
   out_8071738448770388550[34] = 0.0;
   out_8071738448770388550[35] = 0.0;
   out_8071738448770388550[36] = 0.0;
   out_8071738448770388550[37] = 0.0;
   out_8071738448770388550[38] = 0.0;
   out_8071738448770388550[39] = 0.0;
   out_8071738448770388550[40] = 1.0;
   out_8071738448770388550[41] = 0.0;
   out_8071738448770388550[42] = 0.0;
   out_8071738448770388550[43] = 0.0;
   out_8071738448770388550[44] = 0.0;
   out_8071738448770388550[45] = 0.0;
   out_8071738448770388550[46] = 0.0;
   out_8071738448770388550[47] = 0.0;
   out_8071738448770388550[48] = 0.0;
   out_8071738448770388550[49] = 0.0;
   out_8071738448770388550[50] = 1.0;
   out_8071738448770388550[51] = 0.0;
   out_8071738448770388550[52] = 0.0;
   out_8071738448770388550[53] = 0.0;
   out_8071738448770388550[54] = 0.0;
   out_8071738448770388550[55] = 0.0;
   out_8071738448770388550[56] = 0.0;
   out_8071738448770388550[57] = 0.0;
   out_8071738448770388550[58] = 0.0;
   out_8071738448770388550[59] = 0.0;
   out_8071738448770388550[60] = 1.0;
   out_8071738448770388550[61] = 0.0;
   out_8071738448770388550[62] = 0.0;
   out_8071738448770388550[63] = 0.0;
   out_8071738448770388550[64] = 0.0;
   out_8071738448770388550[65] = 0.0;
   out_8071738448770388550[66] = 0.0;
   out_8071738448770388550[67] = 0.0;
   out_8071738448770388550[68] = 0.0;
   out_8071738448770388550[69] = 0.0;
   out_8071738448770388550[70] = 1.0;
   out_8071738448770388550[71] = 0.0;
   out_8071738448770388550[72] = 0.0;
   out_8071738448770388550[73] = 0.0;
   out_8071738448770388550[74] = 0.0;
   out_8071738448770388550[75] = 0.0;
   out_8071738448770388550[76] = 0.0;
   out_8071738448770388550[77] = 0.0;
   out_8071738448770388550[78] = 0.0;
   out_8071738448770388550[79] = 0.0;
   out_8071738448770388550[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_5610024755312278041) {
   out_5610024755312278041[0] = state[0];
   out_5610024755312278041[1] = state[1];
   out_5610024755312278041[2] = state[2];
   out_5610024755312278041[3] = state[3];
   out_5610024755312278041[4] = state[4];
   out_5610024755312278041[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_5610024755312278041[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_5610024755312278041[7] = state[7];
   out_5610024755312278041[8] = state[8];
}
void F_fun(double *state, double dt, double *out_8749456969336840526) {
   out_8749456969336840526[0] = 1;
   out_8749456969336840526[1] = 0;
   out_8749456969336840526[2] = 0;
   out_8749456969336840526[3] = 0;
   out_8749456969336840526[4] = 0;
   out_8749456969336840526[5] = 0;
   out_8749456969336840526[6] = 0;
   out_8749456969336840526[7] = 0;
   out_8749456969336840526[8] = 0;
   out_8749456969336840526[9] = 0;
   out_8749456969336840526[10] = 1;
   out_8749456969336840526[11] = 0;
   out_8749456969336840526[12] = 0;
   out_8749456969336840526[13] = 0;
   out_8749456969336840526[14] = 0;
   out_8749456969336840526[15] = 0;
   out_8749456969336840526[16] = 0;
   out_8749456969336840526[17] = 0;
   out_8749456969336840526[18] = 0;
   out_8749456969336840526[19] = 0;
   out_8749456969336840526[20] = 1;
   out_8749456969336840526[21] = 0;
   out_8749456969336840526[22] = 0;
   out_8749456969336840526[23] = 0;
   out_8749456969336840526[24] = 0;
   out_8749456969336840526[25] = 0;
   out_8749456969336840526[26] = 0;
   out_8749456969336840526[27] = 0;
   out_8749456969336840526[28] = 0;
   out_8749456969336840526[29] = 0;
   out_8749456969336840526[30] = 1;
   out_8749456969336840526[31] = 0;
   out_8749456969336840526[32] = 0;
   out_8749456969336840526[33] = 0;
   out_8749456969336840526[34] = 0;
   out_8749456969336840526[35] = 0;
   out_8749456969336840526[36] = 0;
   out_8749456969336840526[37] = 0;
   out_8749456969336840526[38] = 0;
   out_8749456969336840526[39] = 0;
   out_8749456969336840526[40] = 1;
   out_8749456969336840526[41] = 0;
   out_8749456969336840526[42] = 0;
   out_8749456969336840526[43] = 0;
   out_8749456969336840526[44] = 0;
   out_8749456969336840526[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_8749456969336840526[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_8749456969336840526[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8749456969336840526[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8749456969336840526[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_8749456969336840526[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_8749456969336840526[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_8749456969336840526[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_8749456969336840526[53] = -9.8000000000000007*dt;
   out_8749456969336840526[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_8749456969336840526[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_8749456969336840526[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8749456969336840526[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8749456969336840526[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_8749456969336840526[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_8749456969336840526[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_8749456969336840526[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8749456969336840526[62] = 0;
   out_8749456969336840526[63] = 0;
   out_8749456969336840526[64] = 0;
   out_8749456969336840526[65] = 0;
   out_8749456969336840526[66] = 0;
   out_8749456969336840526[67] = 0;
   out_8749456969336840526[68] = 0;
   out_8749456969336840526[69] = 0;
   out_8749456969336840526[70] = 1;
   out_8749456969336840526[71] = 0;
   out_8749456969336840526[72] = 0;
   out_8749456969336840526[73] = 0;
   out_8749456969336840526[74] = 0;
   out_8749456969336840526[75] = 0;
   out_8749456969336840526[76] = 0;
   out_8749456969336840526[77] = 0;
   out_8749456969336840526[78] = 0;
   out_8749456969336840526[79] = 0;
   out_8749456969336840526[80] = 1;
}
void h_25(double *state, double *unused, double *out_4495051692134795726) {
   out_4495051692134795726[0] = state[6];
}
void H_25(double *state, double *unused, double *out_692419411288880047) {
   out_692419411288880047[0] = 0;
   out_692419411288880047[1] = 0;
   out_692419411288880047[2] = 0;
   out_692419411288880047[3] = 0;
   out_692419411288880047[4] = 0;
   out_692419411288880047[5] = 0;
   out_692419411288880047[6] = 1;
   out_692419411288880047[7] = 0;
   out_692419411288880047[8] = 0;
}
void h_24(double *state, double *unused, double *out_2855344354432327028) {
   out_2855344354432327028[0] = state[4];
   out_2855344354432327028[1] = state[5];
}
void H_24(double *state, double *unused, double *out_4250345010091203183) {
   out_4250345010091203183[0] = 0;
   out_4250345010091203183[1] = 0;
   out_4250345010091203183[2] = 0;
   out_4250345010091203183[3] = 0;
   out_4250345010091203183[4] = 1;
   out_4250345010091203183[5] = 0;
   out_4250345010091203183[6] = 0;
   out_4250345010091203183[7] = 0;
   out_4250345010091203183[8] = 0;
   out_4250345010091203183[9] = 0;
   out_4250345010091203183[10] = 0;
   out_4250345010091203183[11] = 0;
   out_4250345010091203183[12] = 0;
   out_4250345010091203183[13] = 0;
   out_4250345010091203183[14] = 1;
   out_4250345010091203183[15] = 0;
   out_4250345010091203183[16] = 0;
   out_4250345010091203183[17] = 0;
}
void h_30(double *state, double *unused, double *out_4337865023783666581) {
   out_4337865023783666581[0] = state[4];
}
void H_30(double *state, double *unused, double *out_7609109752780496802) {
   out_7609109752780496802[0] = 0;
   out_7609109752780496802[1] = 0;
   out_7609109752780496802[2] = 0;
   out_7609109752780496802[3] = 0;
   out_7609109752780496802[4] = 1;
   out_7609109752780496802[5] = 0;
   out_7609109752780496802[6] = 0;
   out_7609109752780496802[7] = 0;
   out_7609109752780496802[8] = 0;
}
void h_26(double *state, double *unused, double *out_6460693589154982575) {
   out_6460693589154982575[0] = state[7];
}
void H_26(double *state, double *unused, double *out_3049083907585176177) {
   out_3049083907585176177[0] = 0;
   out_3049083907585176177[1] = 0;
   out_3049083907585176177[2] = 0;
   out_3049083907585176177[3] = 0;
   out_3049083907585176177[4] = 0;
   out_3049083907585176177[5] = 0;
   out_3049083907585176177[6] = 0;
   out_3049083907585176177[7] = 1;
   out_3049083907585176177[8] = 0;
}
void h_27(double *state, double *unused, double *out_8950019113913910245) {
   out_8950019113913910245[0] = state[3];
}
void H_27(double *state, double *unused, double *out_5434346440980071891) {
   out_5434346440980071891[0] = 0;
   out_5434346440980071891[1] = 0;
   out_5434346440980071891[2] = 0;
   out_5434346440980071891[3] = 1;
   out_5434346440980071891[4] = 0;
   out_5434346440980071891[5] = 0;
   out_5434346440980071891[6] = 0;
   out_5434346440980071891[7] = 0;
   out_5434346440980071891[8] = 0;
}
void h_29(double *state, double *unused, double *out_1733459702974968930) {
   out_1733459702974968930[0] = state[1];
}
void H_29(double *state, double *unused, double *out_3720983714110520858) {
   out_3720983714110520858[0] = 0;
   out_3720983714110520858[1] = 1;
   out_3720983714110520858[2] = 0;
   out_3720983714110520858[3] = 0;
   out_3720983714110520858[4] = 0;
   out_3720983714110520858[5] = 0;
   out_3720983714110520858[6] = 0;
   out_3720983714110520858[7] = 0;
   out_3720983714110520858[8] = 0;
}
void h_28(double *state, double *unused, double *out_514755787788800606) {
   out_514755787788800606[0] = state[0];
}
void H_28(double *state, double *unused, double *out_1361415302959009716) {
   out_1361415302959009716[0] = 1;
   out_1361415302959009716[1] = 0;
   out_1361415302959009716[2] = 0;
   out_1361415302959009716[3] = 0;
   out_1361415302959009716[4] = 0;
   out_1361415302959009716[5] = 0;
   out_1361415302959009716[6] = 0;
   out_1361415302959009716[7] = 0;
   out_1361415302959009716[8] = 0;
}
void h_31(double *state, double *unused, double *out_7180857155224404954) {
   out_7180857155224404954[0] = state[8];
}
void H_31(double *state, double *unused, double *out_723065373165840475) {
   out_723065373165840475[0] = 0;
   out_723065373165840475[1] = 0;
   out_723065373165840475[2] = 0;
   out_723065373165840475[3] = 0;
   out_723065373165840475[4] = 0;
   out_723065373165840475[5] = 0;
   out_723065373165840475[6] = 0;
   out_723065373165840475[7] = 0;
   out_723065373165840475[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_2868104873166063940) {
  err_fun(nom_x, delta_x, out_2868104873166063940);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_8159495437129412069) {
  inv_err_fun(nom_x, true_x, out_8159495437129412069);
}
void car_H_mod_fun(double *state, double *out_8071738448770388550) {
  H_mod_fun(state, out_8071738448770388550);
}
void car_f_fun(double *state, double dt, double *out_5610024755312278041) {
  f_fun(state,  dt, out_5610024755312278041);
}
void car_F_fun(double *state, double dt, double *out_8749456969336840526) {
  F_fun(state,  dt, out_8749456969336840526);
}
void car_h_25(double *state, double *unused, double *out_4495051692134795726) {
  h_25(state, unused, out_4495051692134795726);
}
void car_H_25(double *state, double *unused, double *out_692419411288880047) {
  H_25(state, unused, out_692419411288880047);
}
void car_h_24(double *state, double *unused, double *out_2855344354432327028) {
  h_24(state, unused, out_2855344354432327028);
}
void car_H_24(double *state, double *unused, double *out_4250345010091203183) {
  H_24(state, unused, out_4250345010091203183);
}
void car_h_30(double *state, double *unused, double *out_4337865023783666581) {
  h_30(state, unused, out_4337865023783666581);
}
void car_H_30(double *state, double *unused, double *out_7609109752780496802) {
  H_30(state, unused, out_7609109752780496802);
}
void car_h_26(double *state, double *unused, double *out_6460693589154982575) {
  h_26(state, unused, out_6460693589154982575);
}
void car_H_26(double *state, double *unused, double *out_3049083907585176177) {
  H_26(state, unused, out_3049083907585176177);
}
void car_h_27(double *state, double *unused, double *out_8950019113913910245) {
  h_27(state, unused, out_8950019113913910245);
}
void car_H_27(double *state, double *unused, double *out_5434346440980071891) {
  H_27(state, unused, out_5434346440980071891);
}
void car_h_29(double *state, double *unused, double *out_1733459702974968930) {
  h_29(state, unused, out_1733459702974968930);
}
void car_H_29(double *state, double *unused, double *out_3720983714110520858) {
  H_29(state, unused, out_3720983714110520858);
}
void car_h_28(double *state, double *unused, double *out_514755787788800606) {
  h_28(state, unused, out_514755787788800606);
}
void car_H_28(double *state, double *unused, double *out_1361415302959009716) {
  H_28(state, unused, out_1361415302959009716);
}
void car_h_31(double *state, double *unused, double *out_7180857155224404954) {
  h_31(state, unused, out_7180857155224404954);
}
void car_H_31(double *state, double *unused, double *out_723065373165840475) {
  H_31(state, unused, out_723065373165840475);
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
