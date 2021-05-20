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
 *                      Code generated with sympy 1.7.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_6392157582760322357) {
   out_6392157582760322357[0] = delta_x[0] + nom_x[0];
   out_6392157582760322357[1] = delta_x[1] + nom_x[1];
   out_6392157582760322357[2] = delta_x[2] + nom_x[2];
   out_6392157582760322357[3] = delta_x[3] + nom_x[3];
   out_6392157582760322357[4] = delta_x[4] + nom_x[4];
   out_6392157582760322357[5] = delta_x[5] + nom_x[5];
   out_6392157582760322357[6] = delta_x[6] + nom_x[6];
   out_6392157582760322357[7] = delta_x[7] + nom_x[7];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_3866323338818758786) {
   out_3866323338818758786[0] = -nom_x[0] + true_x[0];
   out_3866323338818758786[1] = -nom_x[1] + true_x[1];
   out_3866323338818758786[2] = -nom_x[2] + true_x[2];
   out_3866323338818758786[3] = -nom_x[3] + true_x[3];
   out_3866323338818758786[4] = -nom_x[4] + true_x[4];
   out_3866323338818758786[5] = -nom_x[5] + true_x[5];
   out_3866323338818758786[6] = -nom_x[6] + true_x[6];
   out_3866323338818758786[7] = -nom_x[7] + true_x[7];
}
void H_mod_fun(double *state, double *out_6503029791175311535) {
   out_6503029791175311535[0] = 1.0;
   out_6503029791175311535[1] = 0.0;
   out_6503029791175311535[2] = 0.0;
   out_6503029791175311535[3] = 0.0;
   out_6503029791175311535[4] = 0.0;
   out_6503029791175311535[5] = 0.0;
   out_6503029791175311535[6] = 0.0;
   out_6503029791175311535[7] = 0.0;
   out_6503029791175311535[8] = 0.0;
   out_6503029791175311535[9] = 1.0;
   out_6503029791175311535[10] = 0.0;
   out_6503029791175311535[11] = 0.0;
   out_6503029791175311535[12] = 0.0;
   out_6503029791175311535[13] = 0.0;
   out_6503029791175311535[14] = 0.0;
   out_6503029791175311535[15] = 0.0;
   out_6503029791175311535[16] = 0.0;
   out_6503029791175311535[17] = 0.0;
   out_6503029791175311535[18] = 1.0;
   out_6503029791175311535[19] = 0.0;
   out_6503029791175311535[20] = 0.0;
   out_6503029791175311535[21] = 0.0;
   out_6503029791175311535[22] = 0.0;
   out_6503029791175311535[23] = 0.0;
   out_6503029791175311535[24] = 0.0;
   out_6503029791175311535[25] = 0.0;
   out_6503029791175311535[26] = 0.0;
   out_6503029791175311535[27] = 1.0;
   out_6503029791175311535[28] = 0.0;
   out_6503029791175311535[29] = 0.0;
   out_6503029791175311535[30] = 0.0;
   out_6503029791175311535[31] = 0.0;
   out_6503029791175311535[32] = 0.0;
   out_6503029791175311535[33] = 0.0;
   out_6503029791175311535[34] = 0.0;
   out_6503029791175311535[35] = 0.0;
   out_6503029791175311535[36] = 1.0;
   out_6503029791175311535[37] = 0.0;
   out_6503029791175311535[38] = 0.0;
   out_6503029791175311535[39] = 0.0;
   out_6503029791175311535[40] = 0.0;
   out_6503029791175311535[41] = 0.0;
   out_6503029791175311535[42] = 0.0;
   out_6503029791175311535[43] = 0.0;
   out_6503029791175311535[44] = 0.0;
   out_6503029791175311535[45] = 1.0;
   out_6503029791175311535[46] = 0.0;
   out_6503029791175311535[47] = 0.0;
   out_6503029791175311535[48] = 0.0;
   out_6503029791175311535[49] = 0.0;
   out_6503029791175311535[50] = 0.0;
   out_6503029791175311535[51] = 0.0;
   out_6503029791175311535[52] = 0.0;
   out_6503029791175311535[53] = 0.0;
   out_6503029791175311535[54] = 1.0;
   out_6503029791175311535[55] = 0.0;
   out_6503029791175311535[56] = 0.0;
   out_6503029791175311535[57] = 0.0;
   out_6503029791175311535[58] = 0.0;
   out_6503029791175311535[59] = 0.0;
   out_6503029791175311535[60] = 0.0;
   out_6503029791175311535[61] = 0.0;
   out_6503029791175311535[62] = 0.0;
   out_6503029791175311535[63] = 1.0;
}
void f_fun(double *state, double dt, double *out_7152810423689958583) {
   out_7152810423689958583[0] = state[0];
   out_7152810423689958583[1] = state[1];
   out_7152810423689958583[2] = state[2];
   out_7152810423689958583[3] = state[3];
   out_7152810423689958583[4] = state[4];
   out_7152810423689958583[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_7152810423689958583[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_7152810423689958583[7] = state[7];
}
void F_fun(double *state, double dt, double *out_8168263035060162837) {
   out_8168263035060162837[0] = 1;
   out_8168263035060162837[1] = 0;
   out_8168263035060162837[2] = 0;
   out_8168263035060162837[3] = 0;
   out_8168263035060162837[4] = 0;
   out_8168263035060162837[5] = 0;
   out_8168263035060162837[6] = 0;
   out_8168263035060162837[7] = 0;
   out_8168263035060162837[8] = 0;
   out_8168263035060162837[9] = 1;
   out_8168263035060162837[10] = 0;
   out_8168263035060162837[11] = 0;
   out_8168263035060162837[12] = 0;
   out_8168263035060162837[13] = 0;
   out_8168263035060162837[14] = 0;
   out_8168263035060162837[15] = 0;
   out_8168263035060162837[16] = 0;
   out_8168263035060162837[17] = 0;
   out_8168263035060162837[18] = 1;
   out_8168263035060162837[19] = 0;
   out_8168263035060162837[20] = 0;
   out_8168263035060162837[21] = 0;
   out_8168263035060162837[22] = 0;
   out_8168263035060162837[23] = 0;
   out_8168263035060162837[24] = 0;
   out_8168263035060162837[25] = 0;
   out_8168263035060162837[26] = 0;
   out_8168263035060162837[27] = 1;
   out_8168263035060162837[28] = 0;
   out_8168263035060162837[29] = 0;
   out_8168263035060162837[30] = 0;
   out_8168263035060162837[31] = 0;
   out_8168263035060162837[32] = 0;
   out_8168263035060162837[33] = 0;
   out_8168263035060162837[34] = 0;
   out_8168263035060162837[35] = 0;
   out_8168263035060162837[36] = 1;
   out_8168263035060162837[37] = 0;
   out_8168263035060162837[38] = 0;
   out_8168263035060162837[39] = 0;
   out_8168263035060162837[40] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_8168263035060162837[41] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_8168263035060162837[42] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8168263035060162837[43] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8168263035060162837[44] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_8168263035060162837[45] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_8168263035060162837[46] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_8168263035060162837[47] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_8168263035060162837[48] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_8168263035060162837[49] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_8168263035060162837[50] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8168263035060162837[51] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8168263035060162837[52] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_8168263035060162837[53] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_8168263035060162837[54] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_8168263035060162837[55] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8168263035060162837[56] = 0;
   out_8168263035060162837[57] = 0;
   out_8168263035060162837[58] = 0;
   out_8168263035060162837[59] = 0;
   out_8168263035060162837[60] = 0;
   out_8168263035060162837[61] = 0;
   out_8168263035060162837[62] = 0;
   out_8168263035060162837[63] = 1;
}
void h_25(double *state, double *unused, double *out_3988360368053177208) {
   out_3988360368053177208[0] = state[6];
}
void H_25(double *state, double *unused, double *out_3893225050488880445) {
   out_3893225050488880445[0] = 0;
   out_3893225050488880445[1] = 0;
   out_3893225050488880445[2] = 0;
   out_3893225050488880445[3] = 0;
   out_3893225050488880445[4] = 0;
   out_3893225050488880445[5] = 0;
   out_3893225050488880445[6] = 1;
   out_3893225050488880445[7] = 0;
}
void h_24(double *state, double *unused, double *out_7142823418347609148) {
   out_7142823418347609148[0] = state[4];
   out_7142823418347609148[1] = state[5];
}
void H_24(double *state, double *unused, double *out_2581754386723611109) {
   out_2581754386723611109[0] = 0;
   out_2581754386723611109[1] = 0;
   out_2581754386723611109[2] = 0;
   out_2581754386723611109[3] = 0;
   out_2581754386723611109[4] = 1;
   out_2581754386723611109[5] = 0;
   out_2581754386723611109[6] = 0;
   out_2581754386723611109[7] = 0;
   out_2581754386723611109[8] = 0;
   out_2581754386723611109[9] = 0;
   out_2581754386723611109[10] = 0;
   out_2581754386723611109[11] = 0;
   out_2581754386723611109[12] = 0;
   out_2581754386723611109[13] = 1;
   out_2581754386723611109[14] = 0;
   out_2581754386723611109[15] = 0;
}
void h_30(double *state, double *unused, double *out_5320666420972523226) {
   out_5320666420972523226[0] = state[4];
}
void H_30(double *state, double *unused, double *out_5720673860460302835) {
   out_5720673860460302835[0] = 0;
   out_5720673860460302835[1] = 0;
   out_5720673860460302835[2] = 0;
   out_5720673860460302835[3] = 0;
   out_5720673860460302835[4] = 1;
   out_5720673860460302835[5] = 0;
   out_5720673860460302835[6] = 0;
   out_5720673860460302835[7] = 0;
}
void h_26(double *state, double *unused, double *out_3289596398024394460) {
   out_3289596398024394460[0] = state[7];
}
void H_26(double *state, double *unused, double *out_8848925868281461898) {
   out_8848925868281461898[0] = 0;
   out_8848925868281461898[1] = 0;
   out_8848925868281461898[2] = 0;
   out_8848925868281461898[3] = 0;
   out_8848925868281461898[4] = 0;
   out_8848925868281461898[5] = 0;
   out_8848925868281461898[6] = 0;
   out_8848925868281461898[7] = 1;
}
void h_27(double *state, double *unused, double *out_5572926824374950833) {
   out_5572926824374950833[0] = state[3];
}
void H_27(double *state, double *unused, double *out_7008255848296928147) {
   out_7008255848296928147[0] = 0;
   out_7008255848296928147[1] = 0;
   out_7008255848296928147[2] = 0;
   out_7008255848296928147[3] = 1;
   out_7008255848296928147[4] = 0;
   out_7008255848296928147[5] = 0;
   out_7008255848296928147[6] = 0;
   out_7008255848296928147[7] = 0;
}
void h_29(double *state, double *unused, double *out_4874996376036170089) {
   out_4874996376036170089[0] = state[1];
}
void H_29(double *state, double *unused, double *out_5146115246604766660) {
   out_5146115246604766660[0] = 0;
   out_5146115246604766660[1] = 1;
   out_5146115246604766660[2] = 0;
   out_5146115246604766660[3] = 0;
   out_5146115246604766660[4] = 0;
   out_5146115246604766660[5] = 0;
   out_5146115246604766660[6] = 0;
   out_5146115246604766660[7] = 0;
}
void h_28(double *state, double *unused, double *out_3302752609494473091) {
   out_3302752609494473091[0] = state[5];
   out_3302752609494473091[1] = state[6];
}
void H_28(double *state, double *unused, double *out_8847157208749445641) {
   out_8847157208749445641[0] = 0;
   out_8847157208749445641[1] = 0;
   out_8847157208749445641[2] = 0;
   out_8847157208749445641[3] = 0;
   out_8847157208749445641[4] = 0;
   out_8847157208749445641[5] = 1;
   out_8847157208749445641[6] = 0;
   out_8847157208749445641[7] = 0;
   out_8847157208749445641[8] = 0;
   out_8847157208749445641[9] = 0;
   out_8847157208749445641[10] = 0;
   out_8847157208749445641[11] = 0;
   out_8847157208749445641[12] = 0;
   out_8847157208749445641[13] = 0;
   out_8847157208749445641[14] = 1;
   out_8847157208749445641[15] = 0;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_6392157582760322357) {
  err_fun(nom_x, delta_x, out_6392157582760322357);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_3866323338818758786) {
  inv_err_fun(nom_x, true_x, out_3866323338818758786);
}
void car_H_mod_fun(double *state, double *out_6503029791175311535) {
  H_mod_fun(state, out_6503029791175311535);
}
void car_f_fun(double *state, double dt, double *out_7152810423689958583) {
  f_fun(state,  dt, out_7152810423689958583);
}
void car_F_fun(double *state, double dt, double *out_8168263035060162837) {
  F_fun(state,  dt, out_8168263035060162837);
}
void car_h_25(double *state, double *unused, double *out_3988360368053177208) {
  h_25(state, unused, out_3988360368053177208);
}
void car_H_25(double *state, double *unused, double *out_3893225050488880445) {
  H_25(state, unused, out_3893225050488880445);
}
void car_h_24(double *state, double *unused, double *out_7142823418347609148) {
  h_24(state, unused, out_7142823418347609148);
}
void car_H_24(double *state, double *unused, double *out_2581754386723611109) {
  H_24(state, unused, out_2581754386723611109);
}
void car_h_30(double *state, double *unused, double *out_5320666420972523226) {
  h_30(state, unused, out_5320666420972523226);
}
void car_H_30(double *state, double *unused, double *out_5720673860460302835) {
  H_30(state, unused, out_5720673860460302835);
}
void car_h_26(double *state, double *unused, double *out_3289596398024394460) {
  h_26(state, unused, out_3289596398024394460);
}
void car_H_26(double *state, double *unused, double *out_8848925868281461898) {
  H_26(state, unused, out_8848925868281461898);
}
void car_h_27(double *state, double *unused, double *out_5572926824374950833) {
  h_27(state, unused, out_5572926824374950833);
}
void car_H_27(double *state, double *unused, double *out_7008255848296928147) {
  H_27(state, unused, out_7008255848296928147);
}
void car_h_29(double *state, double *unused, double *out_4874996376036170089) {
  h_29(state, unused, out_4874996376036170089);
}
void car_H_29(double *state, double *unused, double *out_5146115246604766660) {
  H_29(state, unused, out_5146115246604766660);
}
void car_h_28(double *state, double *unused, double *out_3302752609494473091) {
  h_28(state, unused, out_3302752609494473091);
}
void car_H_28(double *state, double *unused, double *out_8847157208749445641) {
  H_28(state, unused, out_8847157208749445641);
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
