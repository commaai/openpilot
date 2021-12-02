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
void err_fun(double *nom_x, double *delta_x, double *out_6314627675369028590) {
   out_6314627675369028590[0] = delta_x[0] + nom_x[0];
   out_6314627675369028590[1] = delta_x[1] + nom_x[1];
   out_6314627675369028590[2] = delta_x[2] + nom_x[2];
   out_6314627675369028590[3] = delta_x[3] + nom_x[3];
   out_6314627675369028590[4] = delta_x[4] + nom_x[4];
   out_6314627675369028590[5] = delta_x[5] + nom_x[5];
   out_6314627675369028590[6] = delta_x[6] + nom_x[6];
   out_6314627675369028590[7] = delta_x[7] + nom_x[7];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_7796662908045054598) {
   out_7796662908045054598[0] = -nom_x[0] + true_x[0];
   out_7796662908045054598[1] = -nom_x[1] + true_x[1];
   out_7796662908045054598[2] = -nom_x[2] + true_x[2];
   out_7796662908045054598[3] = -nom_x[3] + true_x[3];
   out_7796662908045054598[4] = -nom_x[4] + true_x[4];
   out_7796662908045054598[5] = -nom_x[5] + true_x[5];
   out_7796662908045054598[6] = -nom_x[6] + true_x[6];
   out_7796662908045054598[7] = -nom_x[7] + true_x[7];
}
void H_mod_fun(double *state, double *out_7490673381729093055) {
   out_7490673381729093055[0] = 1.0;
   out_7490673381729093055[1] = 0.0;
   out_7490673381729093055[2] = 0.0;
   out_7490673381729093055[3] = 0.0;
   out_7490673381729093055[4] = 0.0;
   out_7490673381729093055[5] = 0.0;
   out_7490673381729093055[6] = 0.0;
   out_7490673381729093055[7] = 0.0;
   out_7490673381729093055[8] = 0.0;
   out_7490673381729093055[9] = 1.0;
   out_7490673381729093055[10] = 0.0;
   out_7490673381729093055[11] = 0.0;
   out_7490673381729093055[12] = 0.0;
   out_7490673381729093055[13] = 0.0;
   out_7490673381729093055[14] = 0.0;
   out_7490673381729093055[15] = 0.0;
   out_7490673381729093055[16] = 0.0;
   out_7490673381729093055[17] = 0.0;
   out_7490673381729093055[18] = 1.0;
   out_7490673381729093055[19] = 0.0;
   out_7490673381729093055[20] = 0.0;
   out_7490673381729093055[21] = 0.0;
   out_7490673381729093055[22] = 0.0;
   out_7490673381729093055[23] = 0.0;
   out_7490673381729093055[24] = 0.0;
   out_7490673381729093055[25] = 0.0;
   out_7490673381729093055[26] = 0.0;
   out_7490673381729093055[27] = 1.0;
   out_7490673381729093055[28] = 0.0;
   out_7490673381729093055[29] = 0.0;
   out_7490673381729093055[30] = 0.0;
   out_7490673381729093055[31] = 0.0;
   out_7490673381729093055[32] = 0.0;
   out_7490673381729093055[33] = 0.0;
   out_7490673381729093055[34] = 0.0;
   out_7490673381729093055[35] = 0.0;
   out_7490673381729093055[36] = 1.0;
   out_7490673381729093055[37] = 0.0;
   out_7490673381729093055[38] = 0.0;
   out_7490673381729093055[39] = 0.0;
   out_7490673381729093055[40] = 0.0;
   out_7490673381729093055[41] = 0.0;
   out_7490673381729093055[42] = 0.0;
   out_7490673381729093055[43] = 0.0;
   out_7490673381729093055[44] = 0.0;
   out_7490673381729093055[45] = 1.0;
   out_7490673381729093055[46] = 0.0;
   out_7490673381729093055[47] = 0.0;
   out_7490673381729093055[48] = 0.0;
   out_7490673381729093055[49] = 0.0;
   out_7490673381729093055[50] = 0.0;
   out_7490673381729093055[51] = 0.0;
   out_7490673381729093055[52] = 0.0;
   out_7490673381729093055[53] = 0.0;
   out_7490673381729093055[54] = 1.0;
   out_7490673381729093055[55] = 0.0;
   out_7490673381729093055[56] = 0.0;
   out_7490673381729093055[57] = 0.0;
   out_7490673381729093055[58] = 0.0;
   out_7490673381729093055[59] = 0.0;
   out_7490673381729093055[60] = 0.0;
   out_7490673381729093055[61] = 0.0;
   out_7490673381729093055[62] = 0.0;
   out_7490673381729093055[63] = 1.0;
}
void f_fun(double *state, double dt, double *out_1191313618120458023) {
   out_1191313618120458023[0] = state[0];
   out_1191313618120458023[1] = state[1];
   out_1191313618120458023[2] = state[2];
   out_1191313618120458023[3] = state[3];
   out_1191313618120458023[4] = state[4];
   out_1191313618120458023[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_1191313618120458023[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_1191313618120458023[7] = state[7];
}
void F_fun(double *state, double dt, double *out_3988374560714266677) {
   out_3988374560714266677[0] = 1;
   out_3988374560714266677[1] = 0;
   out_3988374560714266677[2] = 0;
   out_3988374560714266677[3] = 0;
   out_3988374560714266677[4] = 0;
   out_3988374560714266677[5] = 0;
   out_3988374560714266677[6] = 0;
   out_3988374560714266677[7] = 0;
   out_3988374560714266677[8] = 0;
   out_3988374560714266677[9] = 1;
   out_3988374560714266677[10] = 0;
   out_3988374560714266677[11] = 0;
   out_3988374560714266677[12] = 0;
   out_3988374560714266677[13] = 0;
   out_3988374560714266677[14] = 0;
   out_3988374560714266677[15] = 0;
   out_3988374560714266677[16] = 0;
   out_3988374560714266677[17] = 0;
   out_3988374560714266677[18] = 1;
   out_3988374560714266677[19] = 0;
   out_3988374560714266677[20] = 0;
   out_3988374560714266677[21] = 0;
   out_3988374560714266677[22] = 0;
   out_3988374560714266677[23] = 0;
   out_3988374560714266677[24] = 0;
   out_3988374560714266677[25] = 0;
   out_3988374560714266677[26] = 0;
   out_3988374560714266677[27] = 1;
   out_3988374560714266677[28] = 0;
   out_3988374560714266677[29] = 0;
   out_3988374560714266677[30] = 0;
   out_3988374560714266677[31] = 0;
   out_3988374560714266677[32] = 0;
   out_3988374560714266677[33] = 0;
   out_3988374560714266677[34] = 0;
   out_3988374560714266677[35] = 0;
   out_3988374560714266677[36] = 1;
   out_3988374560714266677[37] = 0;
   out_3988374560714266677[38] = 0;
   out_3988374560714266677[39] = 0;
   out_3988374560714266677[40] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_3988374560714266677[41] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_3988374560714266677[42] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_3988374560714266677[43] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_3988374560714266677[44] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_3988374560714266677[45] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_3988374560714266677[46] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_3988374560714266677[47] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_3988374560714266677[48] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_3988374560714266677[49] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_3988374560714266677[50] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3988374560714266677[51] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3988374560714266677[52] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_3988374560714266677[53] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_3988374560714266677[54] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_3988374560714266677[55] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3988374560714266677[56] = 0;
   out_3988374560714266677[57] = 0;
   out_3988374560714266677[58] = 0;
   out_3988374560714266677[59] = 0;
   out_3988374560714266677[60] = 0;
   out_3988374560714266677[61] = 0;
   out_3988374560714266677[62] = 0;
   out_3988374560714266677[63] = 1;
}
void h_25(double *state, double *unused, double *out_3077195206092010461) {
   out_3077195206092010461[0] = state[6];
}
void H_25(double *state, double *unused, double *out_8769480316130288681) {
   out_8769480316130288681[0] = 0;
   out_8769480316130288681[1] = 0;
   out_8769480316130288681[2] = 0;
   out_8769480316130288681[3] = 0;
   out_8769480316130288681[4] = 0;
   out_8769480316130288681[5] = 0;
   out_8769480316130288681[6] = 1;
   out_8769480316130288681[7] = 0;
}
void h_24(double *state, double *unused, double *out_357861895765212515) {
   out_357861895765212515[0] = state[4];
   out_357861895765212515[1] = state[5];
}
void H_24(double *state, double *unused, double *out_3828366448772007202) {
   out_3828366448772007202[0] = 0;
   out_3828366448772007202[1] = 0;
   out_3828366448772007202[2] = 0;
   out_3828366448772007202[3] = 0;
   out_3828366448772007202[4] = 1;
   out_3828366448772007202[5] = 0;
   out_3828366448772007202[6] = 0;
   out_3828366448772007202[7] = 0;
   out_3828366448772007202[8] = 0;
   out_3828366448772007202[9] = 0;
   out_3828366448772007202[10] = 0;
   out_3828366448772007202[11] = 0;
   out_3828366448772007202[12] = 0;
   out_3828366448772007202[13] = 1;
   out_3828366448772007202[14] = 0;
   out_3828366448772007202[15] = 0;
}
void h_30(double *state, double *unused, double *out_3089918833879123842) {
   out_3089918833879123842[0] = state[4];
}
void H_30(double *state, double *unused, double *out_63364846630079655) {
   out_63364846630079655[0] = 0;
   out_63364846630079655[1] = 0;
   out_63364846630079655[2] = 0;
   out_63364846630079655[3] = 0;
   out_63364846630079655[4] = 1;
   out_63364846630079655[5] = 0;
   out_63364846630079655[6] = 0;
   out_63364846630079655[7] = 0;
}
void h_26(double *state, double *unused, double *out_8091592101539969487) {
   out_8091592101539969487[0] = state[7];
}
void H_26(double *state, double *unused, double *out_7463244544175447536) {
   out_7463244544175447536[0] = 0;
   out_7463244544175447536[1] = 0;
   out_7463244544175447536[2] = 0;
   out_7463244544175447536[3] = 0;
   out_7463244544175447536[4] = 0;
   out_7463244544175447536[5] = 0;
   out_7463244544175447536[6] = 0;
   out_7463244544175447536[7] = 1;
}
void h_27(double *state, double *unused, double *out_4178948511001441248) {
   out_4178948511001441248[0] = state[3];
}
void H_27(double *state, double *unused, double *out_1224217141206545657) {
   out_1224217141206545657[0] = 0;
   out_1224217141206545657[1] = 0;
   out_1224217141206545657[2] = 0;
   out_1224217141206545657[3] = 1;
   out_1224217141206545657[4] = 0;
   out_1224217141206545657[5] = 0;
   out_1224217141206545657[6] = 0;
   out_1224217141206545657[7] = 0;
}
void h_29(double *state, double *unused, double *out_6730426071702673160) {
   out_6730426071702673160[0] = state[1];
}
void H_29(double *state, double *unused, double *out_3285595366136104527) {
   out_3285595366136104527[0] = 0;
   out_3285595366136104527[1] = 1;
   out_3285595366136104527[2] = 0;
   out_3285595366136104527[3] = 0;
   out_3285595366136104527[4] = 0;
   out_3285595366136104527[5] = 0;
   out_3285595366136104527[6] = 0;
   out_3285595366136104527[7] = 0;
}
void h_28(double *state, double *unused, double *out_917016906443250257) {
   out_917016906443250257[0] = state[5];
   out_917016906443250257[1] = state[6];
}
void H_28(double *state, double *unused, double *out_3202187763716681420) {
   out_3202187763716681420[0] = 0;
   out_3202187763716681420[1] = 0;
   out_3202187763716681420[2] = 0;
   out_3202187763716681420[3] = 0;
   out_3202187763716681420[4] = 0;
   out_3202187763716681420[5] = 1;
   out_3202187763716681420[6] = 0;
   out_3202187763716681420[7] = 0;
   out_3202187763716681420[8] = 0;
   out_3202187763716681420[9] = 0;
   out_3202187763716681420[10] = 0;
   out_3202187763716681420[11] = 0;
   out_3202187763716681420[12] = 0;
   out_3202187763716681420[13] = 0;
   out_3202187763716681420[14] = 1;
   out_3202187763716681420[15] = 0;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_6314627675369028590) {
  err_fun(nom_x, delta_x, out_6314627675369028590);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7796662908045054598) {
  inv_err_fun(nom_x, true_x, out_7796662908045054598);
}
void car_H_mod_fun(double *state, double *out_7490673381729093055) {
  H_mod_fun(state, out_7490673381729093055);
}
void car_f_fun(double *state, double dt, double *out_1191313618120458023) {
  f_fun(state,  dt, out_1191313618120458023);
}
void car_F_fun(double *state, double dt, double *out_3988374560714266677) {
  F_fun(state,  dt, out_3988374560714266677);
}
void car_h_25(double *state, double *unused, double *out_3077195206092010461) {
  h_25(state, unused, out_3077195206092010461);
}
void car_H_25(double *state, double *unused, double *out_8769480316130288681) {
  H_25(state, unused, out_8769480316130288681);
}
void car_h_24(double *state, double *unused, double *out_357861895765212515) {
  h_24(state, unused, out_357861895765212515);
}
void car_H_24(double *state, double *unused, double *out_3828366448772007202) {
  H_24(state, unused, out_3828366448772007202);
}
void car_h_30(double *state, double *unused, double *out_3089918833879123842) {
  h_30(state, unused, out_3089918833879123842);
}
void car_H_30(double *state, double *unused, double *out_63364846630079655) {
  H_30(state, unused, out_63364846630079655);
}
void car_h_26(double *state, double *unused, double *out_8091592101539969487) {
  h_26(state, unused, out_8091592101539969487);
}
void car_H_26(double *state, double *unused, double *out_7463244544175447536) {
  H_26(state, unused, out_7463244544175447536);
}
void car_h_27(double *state, double *unused, double *out_4178948511001441248) {
  h_27(state, unused, out_4178948511001441248);
}
void car_H_27(double *state, double *unused, double *out_1224217141206545657) {
  H_27(state, unused, out_1224217141206545657);
}
void car_h_29(double *state, double *unused, double *out_6730426071702673160) {
  h_29(state, unused, out_6730426071702673160);
}
void car_H_29(double *state, double *unused, double *out_3285595366136104527) {
  H_29(state, unused, out_3285595366136104527);
}
void car_h_28(double *state, double *unused, double *out_917016906443250257) {
  h_28(state, unused, out_917016906443250257);
}
void car_H_28(double *state, double *unused, double *out_3202187763716681420) {
  H_28(state, unused, out_3202187763716681420);
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
