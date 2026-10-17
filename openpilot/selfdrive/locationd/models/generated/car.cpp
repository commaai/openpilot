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
void err_fun(double *nom_x, double *delta_x, double *out_3479962638771710567) {
   out_3479962638771710567[0] = delta_x[0] + nom_x[0];
   out_3479962638771710567[1] = delta_x[1] + nom_x[1];
   out_3479962638771710567[2] = delta_x[2] + nom_x[2];
   out_3479962638771710567[3] = delta_x[3] + nom_x[3];
   out_3479962638771710567[4] = delta_x[4] + nom_x[4];
   out_3479962638771710567[5] = delta_x[5] + nom_x[5];
   out_3479962638771710567[6] = delta_x[6] + nom_x[6];
   out_3479962638771710567[7] = delta_x[7] + nom_x[7];
   out_3479962638771710567[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_3560122421157608575) {
   out_3560122421157608575[0] = -nom_x[0] + true_x[0];
   out_3560122421157608575[1] = -nom_x[1] + true_x[1];
   out_3560122421157608575[2] = -nom_x[2] + true_x[2];
   out_3560122421157608575[3] = -nom_x[3] + true_x[3];
   out_3560122421157608575[4] = -nom_x[4] + true_x[4];
   out_3560122421157608575[5] = -nom_x[5] + true_x[5];
   out_3560122421157608575[6] = -nom_x[6] + true_x[6];
   out_3560122421157608575[7] = -nom_x[7] + true_x[7];
   out_3560122421157608575[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_7086273587120705954) {
   out_7086273587120705954[0] = 1.0;
   out_7086273587120705954[1] = 0.0;
   out_7086273587120705954[2] = 0.0;
   out_7086273587120705954[3] = 0.0;
   out_7086273587120705954[4] = 0.0;
   out_7086273587120705954[5] = 0.0;
   out_7086273587120705954[6] = 0.0;
   out_7086273587120705954[7] = 0.0;
   out_7086273587120705954[8] = 0.0;
   out_7086273587120705954[9] = 0.0;
   out_7086273587120705954[10] = 1.0;
   out_7086273587120705954[11] = 0.0;
   out_7086273587120705954[12] = 0.0;
   out_7086273587120705954[13] = 0.0;
   out_7086273587120705954[14] = 0.0;
   out_7086273587120705954[15] = 0.0;
   out_7086273587120705954[16] = 0.0;
   out_7086273587120705954[17] = 0.0;
   out_7086273587120705954[18] = 0.0;
   out_7086273587120705954[19] = 0.0;
   out_7086273587120705954[20] = 1.0;
   out_7086273587120705954[21] = 0.0;
   out_7086273587120705954[22] = 0.0;
   out_7086273587120705954[23] = 0.0;
   out_7086273587120705954[24] = 0.0;
   out_7086273587120705954[25] = 0.0;
   out_7086273587120705954[26] = 0.0;
   out_7086273587120705954[27] = 0.0;
   out_7086273587120705954[28] = 0.0;
   out_7086273587120705954[29] = 0.0;
   out_7086273587120705954[30] = 1.0;
   out_7086273587120705954[31] = 0.0;
   out_7086273587120705954[32] = 0.0;
   out_7086273587120705954[33] = 0.0;
   out_7086273587120705954[34] = 0.0;
   out_7086273587120705954[35] = 0.0;
   out_7086273587120705954[36] = 0.0;
   out_7086273587120705954[37] = 0.0;
   out_7086273587120705954[38] = 0.0;
   out_7086273587120705954[39] = 0.0;
   out_7086273587120705954[40] = 1.0;
   out_7086273587120705954[41] = 0.0;
   out_7086273587120705954[42] = 0.0;
   out_7086273587120705954[43] = 0.0;
   out_7086273587120705954[44] = 0.0;
   out_7086273587120705954[45] = 0.0;
   out_7086273587120705954[46] = 0.0;
   out_7086273587120705954[47] = 0.0;
   out_7086273587120705954[48] = 0.0;
   out_7086273587120705954[49] = 0.0;
   out_7086273587120705954[50] = 1.0;
   out_7086273587120705954[51] = 0.0;
   out_7086273587120705954[52] = 0.0;
   out_7086273587120705954[53] = 0.0;
   out_7086273587120705954[54] = 0.0;
   out_7086273587120705954[55] = 0.0;
   out_7086273587120705954[56] = 0.0;
   out_7086273587120705954[57] = 0.0;
   out_7086273587120705954[58] = 0.0;
   out_7086273587120705954[59] = 0.0;
   out_7086273587120705954[60] = 1.0;
   out_7086273587120705954[61] = 0.0;
   out_7086273587120705954[62] = 0.0;
   out_7086273587120705954[63] = 0.0;
   out_7086273587120705954[64] = 0.0;
   out_7086273587120705954[65] = 0.0;
   out_7086273587120705954[66] = 0.0;
   out_7086273587120705954[67] = 0.0;
   out_7086273587120705954[68] = 0.0;
   out_7086273587120705954[69] = 0.0;
   out_7086273587120705954[70] = 1.0;
   out_7086273587120705954[71] = 0.0;
   out_7086273587120705954[72] = 0.0;
   out_7086273587120705954[73] = 0.0;
   out_7086273587120705954[74] = 0.0;
   out_7086273587120705954[75] = 0.0;
   out_7086273587120705954[76] = 0.0;
   out_7086273587120705954[77] = 0.0;
   out_7086273587120705954[78] = 0.0;
   out_7086273587120705954[79] = 0.0;
   out_7086273587120705954[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_6844953383171278992) {
   out_6844953383171278992[0] = state[0];
   out_6844953383171278992[1] = state[1];
   out_6844953383171278992[2] = state[2];
   out_6844953383171278992[3] = state[3];
   out_6844953383171278992[4] = state[4];
   out_6844953383171278992[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_6844953383171278992[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_6844953383171278992[7] = state[7];
   out_6844953383171278992[8] = state[8];
}
void F_fun(double *state, double dt, double *out_6778960402696977626) {
   out_6778960402696977626[0] = 1;
   out_6778960402696977626[1] = 0;
   out_6778960402696977626[2] = 0;
   out_6778960402696977626[3] = 0;
   out_6778960402696977626[4] = 0;
   out_6778960402696977626[5] = 0;
   out_6778960402696977626[6] = 0;
   out_6778960402696977626[7] = 0;
   out_6778960402696977626[8] = 0;
   out_6778960402696977626[9] = 0;
   out_6778960402696977626[10] = 1;
   out_6778960402696977626[11] = 0;
   out_6778960402696977626[12] = 0;
   out_6778960402696977626[13] = 0;
   out_6778960402696977626[14] = 0;
   out_6778960402696977626[15] = 0;
   out_6778960402696977626[16] = 0;
   out_6778960402696977626[17] = 0;
   out_6778960402696977626[18] = 0;
   out_6778960402696977626[19] = 0;
   out_6778960402696977626[20] = 1;
   out_6778960402696977626[21] = 0;
   out_6778960402696977626[22] = 0;
   out_6778960402696977626[23] = 0;
   out_6778960402696977626[24] = 0;
   out_6778960402696977626[25] = 0;
   out_6778960402696977626[26] = 0;
   out_6778960402696977626[27] = 0;
   out_6778960402696977626[28] = 0;
   out_6778960402696977626[29] = 0;
   out_6778960402696977626[30] = 1;
   out_6778960402696977626[31] = 0;
   out_6778960402696977626[32] = 0;
   out_6778960402696977626[33] = 0;
   out_6778960402696977626[34] = 0;
   out_6778960402696977626[35] = 0;
   out_6778960402696977626[36] = 0;
   out_6778960402696977626[37] = 0;
   out_6778960402696977626[38] = 0;
   out_6778960402696977626[39] = 0;
   out_6778960402696977626[40] = 1;
   out_6778960402696977626[41] = 0;
   out_6778960402696977626[42] = 0;
   out_6778960402696977626[43] = 0;
   out_6778960402696977626[44] = 0;
   out_6778960402696977626[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_6778960402696977626[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_6778960402696977626[47] = -dt*stiffness_front*state[0]/state[1];
   out_6778960402696977626[48] = -dt*stiffness_front*state[0]/state[1];
   out_6778960402696977626[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_6778960402696977626[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_6778960402696977626[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_6778960402696977626[52] = dt*stiffness_front*state[0]/state[1];
   out_6778960402696977626[53] = -9.8100000000000005*dt;
   out_6778960402696977626[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_6778960402696977626[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_6778960402696977626[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6778960402696977626[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6778960402696977626[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_6778960402696977626[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_6778960402696977626[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_6778960402696977626[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6778960402696977626[62] = 0;
   out_6778960402696977626[63] = 0;
   out_6778960402696977626[64] = 0;
   out_6778960402696977626[65] = 0;
   out_6778960402696977626[66] = 0;
   out_6778960402696977626[67] = 0;
   out_6778960402696977626[68] = 0;
   out_6778960402696977626[69] = 0;
   out_6778960402696977626[70] = 1;
   out_6778960402696977626[71] = 0;
   out_6778960402696977626[72] = 0;
   out_6778960402696977626[73] = 0;
   out_6778960402696977626[74] = 0;
   out_6778960402696977626[75] = 0;
   out_6778960402696977626[76] = 0;
   out_6778960402696977626[77] = 0;
   out_6778960402696977626[78] = 0;
   out_6778960402696977626[79] = 0;
   out_6778960402696977626[80] = 1;
}
void h_25(double *state, double *unused, double *out_5169296545033589210) {
   out_5169296545033589210[0] = state[6];
}
void H_25(double *state, double *unused, double *out_7837489836599353985) {
   out_7837489836599353985[0] = 0;
   out_7837489836599353985[1] = 0;
   out_7837489836599353985[2] = 0;
   out_7837489836599353985[3] = 0;
   out_7837489836599353985[4] = 0;
   out_7837489836599353985[5] = 0;
   out_7837489836599353985[6] = 1;
   out_7837489836599353985[7] = 0;
   out_7837489836599353985[8] = 0;
}
void h_24(double *state, double *unused, double *out_2794263254641248906) {
   out_2794263254641248906[0] = state[4];
   out_2794263254641248906[1] = state[5];
}
void H_24(double *state, double *unused, double *out_4279564237797030849) {
   out_4279564237797030849[0] = 0;
   out_4279564237797030849[1] = 0;
   out_4279564237797030849[2] = 0;
   out_4279564237797030849[3] = 0;
   out_4279564237797030849[4] = 1;
   out_4279564237797030849[5] = 0;
   out_4279564237797030849[6] = 0;
   out_4279564237797030849[7] = 0;
   out_4279564237797030849[8] = 0;
   out_4279564237797030849[9] = 0;
   out_4279564237797030849[10] = 0;
   out_4279564237797030849[11] = 0;
   out_4279564237797030849[12] = 0;
   out_4279564237797030849[13] = 0;
   out_4279564237797030849[14] = 1;
   out_4279564237797030849[15] = 0;
   out_4279564237797030849[16] = 0;
   out_4279564237797030849[17] = 0;
}
void h_30(double *state, double *unused, double *out_5012109876682460065) {
   out_5012109876682460065[0] = state[4];
}
void H_30(double *state, double *unused, double *out_920799495107737230) {
   out_920799495107737230[0] = 0;
   out_920799495107737230[1] = 0;
   out_920799495107737230[2] = 0;
   out_920799495107737230[3] = 0;
   out_920799495107737230[4] = 1;
   out_920799495107737230[5] = 0;
   out_920799495107737230[6] = 0;
   out_920799495107737230[7] = 0;
   out_920799495107737230[8] = 0;
}
void h_26(double *state, double *unused, double *out_903520877051402271) {
   out_903520877051402271[0] = state[7];
}
void H_26(double *state, double *unused, double *out_6867750918236141407) {
   out_6867750918236141407[0] = 0;
   out_6867750918236141407[1] = 0;
   out_6867750918236141407[2] = 0;
   out_6867750918236141407[3] = 0;
   out_6867750918236141407[4] = 0;
   out_6867750918236141407[5] = 0;
   out_6867750918236141407[6] = 0;
   out_6867750918236141407[7] = 1;
   out_6867750918236141407[8] = 0;
}
void h_27(double *state, double *unused, double *out_7831542771752882352) {
   out_7831542771752882352[0] = state[3];
}
void H_27(double *state, double *unused, double *out_3095562806908162141) {
   out_3095562806908162141[0] = 0;
   out_3095562806908162141[1] = 0;
   out_3095562806908162141[2] = 0;
   out_3095562806908162141[3] = 1;
   out_3095562806908162141[4] = 0;
   out_3095562806908162141[5] = 0;
   out_3095562806908162141[6] = 0;
   out_3095562806908162141[7] = 0;
   out_3095562806908162141[8] = 0;
}
void h_29(double *state, double *unused, double *out_8432960929366245906) {
   out_8432960929366245906[0] = state[1];
}
void H_29(double *state, double *unused, double *out_4808925533777713174) {
   out_4808925533777713174[0] = 0;
   out_4808925533777713174[1] = 1;
   out_4808925533777713174[2] = 0;
   out_4808925533777713174[3] = 0;
   out_4808925533777713174[4] = 0;
   out_4808925533777713174[5] = 0;
   out_4808925533777713174[6] = 0;
   out_4808925533777713174[7] = 0;
   out_4808925533777713174[8] = 0;
}
void h_28(double *state, double *unused, double *out_1189000640687594090) {
   out_1189000640687594090[0] = state[0];
}
void H_28(double *state, double *unused, double *out_8555419522862307868) {
   out_8555419522862307868[0] = 1;
   out_8555419522862307868[1] = 0;
   out_8555419522862307868[2] = 0;
   out_8555419522862307868[3] = 0;
   out_8555419522862307868[4] = 0;
   out_8555419522862307868[5] = 0;
   out_8555419522862307868[6] = 0;
   out_8555419522862307868[7] = 0;
   out_8555419522862307868[8] = 0;
}
void h_31(double *state, double *unused, double *out_6506612302325611470) {
   out_6506612302325611470[0] = state[8];
}
void H_31(double *state, double *unused, double *out_7806843874722393557) {
   out_7806843874722393557[0] = 0;
   out_7806843874722393557[1] = 0;
   out_7806843874722393557[2] = 0;
   out_7806843874722393557[3] = 0;
   out_7806843874722393557[4] = 0;
   out_7806843874722393557[5] = 0;
   out_7806843874722393557[6] = 0;
   out_7806843874722393557[7] = 0;
   out_7806843874722393557[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_3479962638771710567) {
  err_fun(nom_x, delta_x, out_3479962638771710567);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_3560122421157608575) {
  inv_err_fun(nom_x, true_x, out_3560122421157608575);
}
void car_H_mod_fun(double *state, double *out_7086273587120705954) {
  H_mod_fun(state, out_7086273587120705954);
}
void car_f_fun(double *state, double dt, double *out_6844953383171278992) {
  f_fun(state,  dt, out_6844953383171278992);
}
void car_F_fun(double *state, double dt, double *out_6778960402696977626) {
  F_fun(state,  dt, out_6778960402696977626);
}
void car_h_25(double *state, double *unused, double *out_5169296545033589210) {
  h_25(state, unused, out_5169296545033589210);
}
void car_H_25(double *state, double *unused, double *out_7837489836599353985) {
  H_25(state, unused, out_7837489836599353985);
}
void car_h_24(double *state, double *unused, double *out_2794263254641248906) {
  h_24(state, unused, out_2794263254641248906);
}
void car_H_24(double *state, double *unused, double *out_4279564237797030849) {
  H_24(state, unused, out_4279564237797030849);
}
void car_h_30(double *state, double *unused, double *out_5012109876682460065) {
  h_30(state, unused, out_5012109876682460065);
}
void car_H_30(double *state, double *unused, double *out_920799495107737230) {
  H_30(state, unused, out_920799495107737230);
}
void car_h_26(double *state, double *unused, double *out_903520877051402271) {
  h_26(state, unused, out_903520877051402271);
}
void car_H_26(double *state, double *unused, double *out_6867750918236141407) {
  H_26(state, unused, out_6867750918236141407);
}
void car_h_27(double *state, double *unused, double *out_7831542771752882352) {
  h_27(state, unused, out_7831542771752882352);
}
void car_H_27(double *state, double *unused, double *out_3095562806908162141) {
  H_27(state, unused, out_3095562806908162141);
}
void car_h_29(double *state, double *unused, double *out_8432960929366245906) {
  h_29(state, unused, out_8432960929366245906);
}
void car_H_29(double *state, double *unused, double *out_4808925533777713174) {
  H_29(state, unused, out_4808925533777713174);
}
void car_h_28(double *state, double *unused, double *out_1189000640687594090) {
  h_28(state, unused, out_1189000640687594090);
}
void car_H_28(double *state, double *unused, double *out_8555419522862307868) {
  H_28(state, unused, out_8555419522862307868);
}
void car_h_31(double *state, double *unused, double *out_6506612302325611470) {
  h_31(state, unused, out_6506612302325611470);
}
void car_H_31(double *state, double *unused, double *out_7806843874722393557) {
  H_31(state, unused, out_7806843874722393557);
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
