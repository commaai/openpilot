#include "car.h"

namespace {
#define DIM 8
#define EDIM 8
#define MEDIM 8
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
const static double MAHA_THRESH_28 = 5.991464547107981;

/******************************************************************************
 *                       Code generated with sympy 1.8                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_4238289070007684364) {
   out_4238289070007684364[0] = delta_x[0] + nom_x[0];
   out_4238289070007684364[1] = delta_x[1] + nom_x[1];
   out_4238289070007684364[2] = delta_x[2] + nom_x[2];
   out_4238289070007684364[3] = delta_x[3] + nom_x[3];
   out_4238289070007684364[4] = delta_x[4] + nom_x[4];
   out_4238289070007684364[5] = delta_x[5] + nom_x[5];
   out_4238289070007684364[6] = delta_x[6] + nom_x[6];
   out_4238289070007684364[7] = delta_x[7] + nom_x[7];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_3393139237430767898) {
   out_3393139237430767898[0] = -nom_x[0] + true_x[0];
   out_3393139237430767898[1] = -nom_x[1] + true_x[1];
   out_3393139237430767898[2] = -nom_x[2] + true_x[2];
   out_3393139237430767898[3] = -nom_x[3] + true_x[3];
   out_3393139237430767898[4] = -nom_x[4] + true_x[4];
   out_3393139237430767898[5] = -nom_x[5] + true_x[5];
   out_3393139237430767898[6] = -nom_x[6] + true_x[6];
   out_3393139237430767898[7] = -nom_x[7] + true_x[7];
}
void H_mod_fun(double *state, double *out_6977086138309336193) {
   out_6977086138309336193[0] = 1.0;
   out_6977086138309336193[1] = 0.0;
   out_6977086138309336193[2] = 0.0;
   out_6977086138309336193[3] = 0.0;
   out_6977086138309336193[4] = 0.0;
   out_6977086138309336193[5] = 0.0;
   out_6977086138309336193[6] = 0.0;
   out_6977086138309336193[7] = 0.0;
   out_6977086138309336193[8] = 0.0;
   out_6977086138309336193[9] = 1.0;
   out_6977086138309336193[10] = 0.0;
   out_6977086138309336193[11] = 0.0;
   out_6977086138309336193[12] = 0.0;
   out_6977086138309336193[13] = 0.0;
   out_6977086138309336193[14] = 0.0;
   out_6977086138309336193[15] = 0.0;
   out_6977086138309336193[16] = 0.0;
   out_6977086138309336193[17] = 0.0;
   out_6977086138309336193[18] = 1.0;
   out_6977086138309336193[19] = 0.0;
   out_6977086138309336193[20] = 0.0;
   out_6977086138309336193[21] = 0.0;
   out_6977086138309336193[22] = 0.0;
   out_6977086138309336193[23] = 0.0;
   out_6977086138309336193[24] = 0.0;
   out_6977086138309336193[25] = 0.0;
   out_6977086138309336193[26] = 0.0;
   out_6977086138309336193[27] = 1.0;
   out_6977086138309336193[28] = 0.0;
   out_6977086138309336193[29] = 0.0;
   out_6977086138309336193[30] = 0.0;
   out_6977086138309336193[31] = 0.0;
   out_6977086138309336193[32] = 0.0;
   out_6977086138309336193[33] = 0.0;
   out_6977086138309336193[34] = 0.0;
   out_6977086138309336193[35] = 0.0;
   out_6977086138309336193[36] = 1.0;
   out_6977086138309336193[37] = 0.0;
   out_6977086138309336193[38] = 0.0;
   out_6977086138309336193[39] = 0.0;
   out_6977086138309336193[40] = 0.0;
   out_6977086138309336193[41] = 0.0;
   out_6977086138309336193[42] = 0.0;
   out_6977086138309336193[43] = 0.0;
   out_6977086138309336193[44] = 0.0;
   out_6977086138309336193[45] = 1.0;
   out_6977086138309336193[46] = 0.0;
   out_6977086138309336193[47] = 0.0;
   out_6977086138309336193[48] = 0.0;
   out_6977086138309336193[49] = 0.0;
   out_6977086138309336193[50] = 0.0;
   out_6977086138309336193[51] = 0.0;
   out_6977086138309336193[52] = 0.0;
   out_6977086138309336193[53] = 0.0;
   out_6977086138309336193[54] = 1.0;
   out_6977086138309336193[55] = 0.0;
   out_6977086138309336193[56] = 0.0;
   out_6977086138309336193[57] = 0.0;
   out_6977086138309336193[58] = 0.0;
   out_6977086138309336193[59] = 0.0;
   out_6977086138309336193[60] = 0.0;
   out_6977086138309336193[61] = 0.0;
   out_6977086138309336193[62] = 0.0;
   out_6977086138309336193[63] = 1.0;
}
void f_fun(double *state, double dt, double *out_8805650115117454659) {
   out_8805650115117454659[0] = state[0];
   out_8805650115117454659[1] = state[1];
   out_8805650115117454659[2] = state[2];
   out_8805650115117454659[3] = state[3];
   out_8805650115117454659[4] = state[4];
   out_8805650115117454659[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_8805650115117454659[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_8805650115117454659[7] = state[7];
}
void F_fun(double *state, double dt, double *out_6686524369442219331) {
   out_6686524369442219331[0] = 1;
   out_6686524369442219331[1] = 0;
   out_6686524369442219331[2] = 0;
   out_6686524369442219331[3] = 0;
   out_6686524369442219331[4] = 0;
   out_6686524369442219331[5] = 0;
   out_6686524369442219331[6] = 0;
   out_6686524369442219331[7] = 0;
   out_6686524369442219331[8] = 0;
   out_6686524369442219331[9] = 1;
   out_6686524369442219331[10] = 0;
   out_6686524369442219331[11] = 0;
   out_6686524369442219331[12] = 0;
   out_6686524369442219331[13] = 0;
   out_6686524369442219331[14] = 0;
   out_6686524369442219331[15] = 0;
   out_6686524369442219331[16] = 0;
   out_6686524369442219331[17] = 0;
   out_6686524369442219331[18] = 1;
   out_6686524369442219331[19] = 0;
   out_6686524369442219331[20] = 0;
   out_6686524369442219331[21] = 0;
   out_6686524369442219331[22] = 0;
   out_6686524369442219331[23] = 0;
   out_6686524369442219331[24] = 0;
   out_6686524369442219331[25] = 0;
   out_6686524369442219331[26] = 0;
   out_6686524369442219331[27] = 1;
   out_6686524369442219331[28] = 0;
   out_6686524369442219331[29] = 0;
   out_6686524369442219331[30] = 0;
   out_6686524369442219331[31] = 0;
   out_6686524369442219331[32] = 0;
   out_6686524369442219331[33] = 0;
   out_6686524369442219331[34] = 0;
   out_6686524369442219331[35] = 0;
   out_6686524369442219331[36] = 1;
   out_6686524369442219331[37] = 0;
   out_6686524369442219331[38] = 0;
   out_6686524369442219331[39] = 0;
   out_6686524369442219331[40] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_6686524369442219331[41] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_6686524369442219331[42] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_6686524369442219331[43] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_6686524369442219331[44] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_6686524369442219331[45] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_6686524369442219331[46] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_6686524369442219331[47] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_6686524369442219331[48] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_6686524369442219331[49] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_6686524369442219331[50] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6686524369442219331[51] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6686524369442219331[52] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_6686524369442219331[53] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_6686524369442219331[54] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_6686524369442219331[55] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_6686524369442219331[56] = 0;
   out_6686524369442219331[57] = 0;
   out_6686524369442219331[58] = 0;
   out_6686524369442219331[59] = 0;
   out_6686524369442219331[60] = 0;
   out_6686524369442219331[61] = 0;
   out_6686524369442219331[62] = 0;
   out_6686524369442219331[63] = 1;
}
void h_25(double *state, double *unused, double *out_3742018707143453848) {
   out_3742018707143453848[0] = state[6];
}
void H_25(double *state, double *unused, double *out_5681767162363916568) {
   out_5681767162363916568[0] = 0;
   out_5681767162363916568[1] = 0;
   out_5681767162363916568[2] = 0;
   out_5681767162363916568[3] = 0;
   out_5681767162363916568[4] = 0;
   out_5681767162363916568[5] = 0;
   out_5681767162363916568[6] = 1;
   out_5681767162363916568[7] = 0;
}
void h_24(double *state, double *unused, double *out_5471145934099093935) {
   out_5471145934099093935[0] = state[4];
   out_5471145934099093935[1] = state[5];
}
void H_24(double *state, double *unused, double *out_8296790777733552478) {
   out_8296790777733552478[0] = 0;
   out_8296790777733552478[1] = 0;
   out_8296790777733552478[2] = 0;
   out_8296790777733552478[3] = 0;
   out_8296790777733552478[4] = 1;
   out_8296790777733552478[5] = 0;
   out_8296790777733552478[6] = 0;
   out_8296790777733552478[7] = 0;
   out_8296790777733552478[8] = 0;
   out_8296790777733552478[9] = 0;
   out_8296790777733552478[10] = 0;
   out_8296790777733552478[11] = 0;
   out_8296790777733552478[12] = 0;
   out_8296790777733552478[13] = 1;
   out_8296790777733552478[14] = 0;
   out_8296790777733552478[15] = 0;
}
void h_30(double *state, double *unused, double *out_7933890140215746832) {
   out_7933890140215746832[0] = state[4];
}
void H_30(double *state, double *unused, double *out_1247279382587916360) {
   out_1247279382587916360[0] = 0;
   out_1247279382587916360[1] = 0;
   out_1247279382587916360[2] = 0;
   out_1247279382587916360[3] = 0;
   out_1247279382587916360[4] = 1;
   out_1247279382587916360[5] = 0;
   out_1247279382587916360[6] = 0;
   out_1247279382587916360[7] = 0;
}
void h_26(double *state, double *unused, double *out_2932216872554901187) {
   out_2932216872554901187[0] = state[7];
}
void H_26(double *state, double *unused, double *out_4375531390409075423) {
   out_4375531390409075423[0] = 0;
   out_4375531390409075423[1] = 0;
   out_4375531390409075423[2] = 0;
   out_4375531390409075423[3] = 0;
   out_4375531390409075423[4] = 0;
   out_4375531390409075423[5] = 0;
   out_4375531390409075423[6] = 0;
   out_4375531390409075423[7] = 1;
}
void h_27(double *state, double *unused, double *out_7213026032994887806) {
   out_7213026032994887806[0] = state[3];
}
void H_27(double *state, double *unused, double *out_1863496012559826456) {
   out_1863496012559826456[0] = 0;
   out_1863496012559826456[1] = 0;
   out_1863496012559826456[2] = 0;
   out_1863496012559826456[3] = 1;
   out_1863496012559826456[4] = 0;
   out_1863496012559826456[5] = 0;
   out_1863496012559826456[6] = 0;
   out_1863496012559826456[7] = 0;
}
void h_29(double *state, double *unused, double *out_5121338036945893449) {
   out_5121338036945893449[0] = state[1];
}
void H_29(double *state, double *unused, double *out_6373308519902476640) {
   out_6373308519902476640[0] = 0;
   out_6373308519902476640[1] = 1;
   out_6373308519902476640[2] = 0;
   out_6373308519902476640[3] = 0;
   out_6373308519902476640[4] = 0;
   out_6373308519902476640[5] = 0;
   out_6373308519902476640[6] = 0;
   out_6373308519902476640[7] = 0;
}
void h_28(double *state, double *unused, double *out_4307520237050144928) {
   out_4307520237050144928[0] = state[5];
   out_4307520237050144928[1] = state[6];
}
void H_28(double *state, double *unused, double *out_3132120817739504272) {
   out_3132120817739504272[0] = 0;
   out_3132120817739504272[1] = 0;
   out_3132120817739504272[2] = 0;
   out_3132120817739504272[3] = 0;
   out_3132120817739504272[4] = 0;
   out_3132120817739504272[5] = 1;
   out_3132120817739504272[6] = 0;
   out_3132120817739504272[7] = 0;
   out_3132120817739504272[8] = 0;
   out_3132120817739504272[9] = 0;
   out_3132120817739504272[10] = 0;
   out_3132120817739504272[11] = 0;
   out_3132120817739504272[12] = 0;
   out_3132120817739504272[13] = 0;
   out_3132120817739504272[14] = 1;
   out_3132120817739504272[15] = 0;
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
  update<2, 3, 0>(in_x, in_P, h_28, H_28, NULL, in_z, in_R, in_ea, MAHA_THRESH_28);
}
void car_err_fun(double *nom_x, double *delta_x, double *out_4238289070007684364) {
  err_fun(nom_x, delta_x, out_4238289070007684364);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_3393139237430767898) {
  inv_err_fun(nom_x, true_x, out_3393139237430767898);
}
void car_H_mod_fun(double *state, double *out_6977086138309336193) {
  H_mod_fun(state, out_6977086138309336193);
}
void car_f_fun(double *state, double dt, double *out_8805650115117454659) {
  f_fun(state,  dt, out_8805650115117454659);
}
void car_F_fun(double *state, double dt, double *out_6686524369442219331) {
  F_fun(state,  dt, out_6686524369442219331);
}
void car_h_25(double *state, double *unused, double *out_3742018707143453848) {
  h_25(state, unused, out_3742018707143453848);
}
void car_H_25(double *state, double *unused, double *out_5681767162363916568) {
  H_25(state, unused, out_5681767162363916568);
}
void car_h_24(double *state, double *unused, double *out_5471145934099093935) {
  h_24(state, unused, out_5471145934099093935);
}
void car_H_24(double *state, double *unused, double *out_8296790777733552478) {
  H_24(state, unused, out_8296790777733552478);
}
void car_h_30(double *state, double *unused, double *out_7933890140215746832) {
  h_30(state, unused, out_7933890140215746832);
}
void car_H_30(double *state, double *unused, double *out_1247279382587916360) {
  H_30(state, unused, out_1247279382587916360);
}
void car_h_26(double *state, double *unused, double *out_2932216872554901187) {
  h_26(state, unused, out_2932216872554901187);
}
void car_H_26(double *state, double *unused, double *out_4375531390409075423) {
  H_26(state, unused, out_4375531390409075423);
}
void car_h_27(double *state, double *unused, double *out_7213026032994887806) {
  h_27(state, unused, out_7213026032994887806);
}
void car_H_27(double *state, double *unused, double *out_1863496012559826456) {
  H_27(state, unused, out_1863496012559826456);
}
void car_h_29(double *state, double *unused, double *out_5121338036945893449) {
  h_29(state, unused, out_5121338036945893449);
}
void car_H_29(double *state, double *unused, double *out_6373308519902476640) {
  H_29(state, unused, out_6373308519902476640);
}
void car_h_28(double *state, double *unused, double *out_4307520237050144928) {
  h_28(state, unused, out_4307520237050144928);
}
void car_H_28(double *state, double *unused, double *out_3132120817739504272) {
  H_28(state, unused, out_3132120817739504272);
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
  .kinds = { 25, 24, 30, 26, 27, 29, 28 },
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
  },
  .Hs = {
    { 25, car_H_25 },
    { 24, car_H_24 },
    { 30, car_H_30 },
    { 26, car_H_26 },
    { 27, car_H_27 },
    { 29, car_H_29 },
    { 28, car_H_28 },
  },
  .updates = {
    { 25, car_update_25 },
    { 24, car_update_24 },
    { 30, car_update_30 },
    { 26, car_update_26 },
    { 27, car_update_27 },
    { 29, car_update_29 },
    { 28, car_update_28 },
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
