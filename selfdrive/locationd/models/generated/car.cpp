
extern "C"{

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

}
extern "C" {
#include <math.h>
/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_2030068162292336593) {
   out_2030068162292336593[0] = delta_x[0] + nom_x[0];
   out_2030068162292336593[1] = delta_x[1] + nom_x[1];
   out_2030068162292336593[2] = delta_x[2] + nom_x[2];
   out_2030068162292336593[3] = delta_x[3] + nom_x[3];
   out_2030068162292336593[4] = delta_x[4] + nom_x[4];
   out_2030068162292336593[5] = delta_x[5] + nom_x[5];
   out_2030068162292336593[6] = delta_x[6] + nom_x[6];
   out_2030068162292336593[7] = delta_x[7] + nom_x[7];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_537171033141946999) {
   out_537171033141946999[0] = -nom_x[0] + true_x[0];
   out_537171033141946999[1] = -nom_x[1] + true_x[1];
   out_537171033141946999[2] = -nom_x[2] + true_x[2];
   out_537171033141946999[3] = -nom_x[3] + true_x[3];
   out_537171033141946999[4] = -nom_x[4] + true_x[4];
   out_537171033141946999[5] = -nom_x[5] + true_x[5];
   out_537171033141946999[6] = -nom_x[6] + true_x[6];
   out_537171033141946999[7] = -nom_x[7] + true_x[7];
}
void H_mod_fun(double *state, double *out_7461128603485662222) {
   out_7461128603485662222[0] = 1.0;
   out_7461128603485662222[1] = 0.0;
   out_7461128603485662222[2] = 0.0;
   out_7461128603485662222[3] = 0.0;
   out_7461128603485662222[4] = 0.0;
   out_7461128603485662222[5] = 0.0;
   out_7461128603485662222[6] = 0.0;
   out_7461128603485662222[7] = 0.0;
   out_7461128603485662222[8] = 0.0;
   out_7461128603485662222[9] = 1.0;
   out_7461128603485662222[10] = 0.0;
   out_7461128603485662222[11] = 0.0;
   out_7461128603485662222[12] = 0.0;
   out_7461128603485662222[13] = 0.0;
   out_7461128603485662222[14] = 0.0;
   out_7461128603485662222[15] = 0.0;
   out_7461128603485662222[16] = 0.0;
   out_7461128603485662222[17] = 0.0;
   out_7461128603485662222[18] = 1.0;
   out_7461128603485662222[19] = 0.0;
   out_7461128603485662222[20] = 0.0;
   out_7461128603485662222[21] = 0.0;
   out_7461128603485662222[22] = 0.0;
   out_7461128603485662222[23] = 0.0;
   out_7461128603485662222[24] = 0.0;
   out_7461128603485662222[25] = 0.0;
   out_7461128603485662222[26] = 0.0;
   out_7461128603485662222[27] = 1.0;
   out_7461128603485662222[28] = 0.0;
   out_7461128603485662222[29] = 0.0;
   out_7461128603485662222[30] = 0.0;
   out_7461128603485662222[31] = 0.0;
   out_7461128603485662222[32] = 0.0;
   out_7461128603485662222[33] = 0.0;
   out_7461128603485662222[34] = 0.0;
   out_7461128603485662222[35] = 0.0;
   out_7461128603485662222[36] = 1.0;
   out_7461128603485662222[37] = 0.0;
   out_7461128603485662222[38] = 0.0;
   out_7461128603485662222[39] = 0.0;
   out_7461128603485662222[40] = 0.0;
   out_7461128603485662222[41] = 0.0;
   out_7461128603485662222[42] = 0.0;
   out_7461128603485662222[43] = 0.0;
   out_7461128603485662222[44] = 0.0;
   out_7461128603485662222[45] = 1.0;
   out_7461128603485662222[46] = 0.0;
   out_7461128603485662222[47] = 0.0;
   out_7461128603485662222[48] = 0.0;
   out_7461128603485662222[49] = 0.0;
   out_7461128603485662222[50] = 0.0;
   out_7461128603485662222[51] = 0.0;
   out_7461128603485662222[52] = 0.0;
   out_7461128603485662222[53] = 0.0;
   out_7461128603485662222[54] = 1.0;
   out_7461128603485662222[55] = 0.0;
   out_7461128603485662222[56] = 0.0;
   out_7461128603485662222[57] = 0.0;
   out_7461128603485662222[58] = 0.0;
   out_7461128603485662222[59] = 0.0;
   out_7461128603485662222[60] = 0.0;
   out_7461128603485662222[61] = 0.0;
   out_7461128603485662222[62] = 0.0;
   out_7461128603485662222[63] = 1.0;
}
void f_fun(double *state, double dt, double *out_7427120013843204316) {
   out_7427120013843204316[0] = state[0];
   out_7427120013843204316[1] = state[1];
   out_7427120013843204316[2] = state[2];
   out_7427120013843204316[3] = state[3];
   out_7427120013843204316[4] = state[4];
   out_7427120013843204316[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_7427120013843204316[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_7427120013843204316[7] = state[7];
}
void F_fun(double *state, double dt, double *out_537719153909518512) {
   out_537719153909518512[0] = 1;
   out_537719153909518512[1] = 0;
   out_537719153909518512[2] = 0;
   out_537719153909518512[3] = 0;
   out_537719153909518512[4] = 0;
   out_537719153909518512[5] = 0;
   out_537719153909518512[6] = 0;
   out_537719153909518512[7] = 0;
   out_537719153909518512[8] = 0;
   out_537719153909518512[9] = 1;
   out_537719153909518512[10] = 0;
   out_537719153909518512[11] = 0;
   out_537719153909518512[12] = 0;
   out_537719153909518512[13] = 0;
   out_537719153909518512[14] = 0;
   out_537719153909518512[15] = 0;
   out_537719153909518512[16] = 0;
   out_537719153909518512[17] = 0;
   out_537719153909518512[18] = 1;
   out_537719153909518512[19] = 0;
   out_537719153909518512[20] = 0;
   out_537719153909518512[21] = 0;
   out_537719153909518512[22] = 0;
   out_537719153909518512[23] = 0;
   out_537719153909518512[24] = 0;
   out_537719153909518512[25] = 0;
   out_537719153909518512[26] = 0;
   out_537719153909518512[27] = 1;
   out_537719153909518512[28] = 0;
   out_537719153909518512[29] = 0;
   out_537719153909518512[30] = 0;
   out_537719153909518512[31] = 0;
   out_537719153909518512[32] = 0;
   out_537719153909518512[33] = 0;
   out_537719153909518512[34] = 0;
   out_537719153909518512[35] = 0;
   out_537719153909518512[36] = 1;
   out_537719153909518512[37] = 0;
   out_537719153909518512[38] = 0;
   out_537719153909518512[39] = 0;
   out_537719153909518512[40] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_537719153909518512[41] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_537719153909518512[42] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_537719153909518512[43] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_537719153909518512[44] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_537719153909518512[45] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_537719153909518512[46] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_537719153909518512[47] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_537719153909518512[48] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_537719153909518512[49] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_537719153909518512[50] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_537719153909518512[51] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_537719153909518512[52] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_537719153909518512[53] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_537719153909518512[54] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_537719153909518512[55] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_537719153909518512[56] = 0;
   out_537719153909518512[57] = 0;
   out_537719153909518512[58] = 0;
   out_537719153909518512[59] = 0;
   out_537719153909518512[60] = 0;
   out_537719153909518512[61] = 0;
   out_537719153909518512[62] = 0;
   out_537719153909518512[63] = 1;
}
void h_25(double *state, double *unused, double *out_5730183952204665335) {
   out_5730183952204665335[0] = state[6];
}
void H_25(double *state, double *unused, double *out_6850079430537253322) {
   out_6850079430537253322[0] = 0;
   out_6850079430537253322[1] = 0;
   out_6850079430537253322[2] = 0;
   out_6850079430537253322[3] = 0;
   out_6850079430537253322[4] = 0;
   out_6850079430537253322[5] = 0;
   out_6850079430537253322[6] = 1;
   out_6850079430537253322[7] = 0;
}
void h_24(double *state, double *unused, double *out_2156192023864214624) {
   out_2156192023864214624[0] = state[4];
   out_2156192023864214624[1] = state[5];
}
void H_24(double *state, double *unused, double *out_6745777559073209274) {
   out_6745777559073209274[0] = 0;
   out_6745777559073209274[1] = 0;
   out_6745777559073209274[2] = 0;
   out_6745777559073209274[3] = 0;
   out_6745777559073209274[4] = 1;
   out_6745777559073209274[5] = 0;
   out_6745777559073209274[6] = 0;
   out_6745777559073209274[7] = 0;
   out_6745777559073209274[8] = 0;
   out_6745777559073209274[9] = 0;
   out_6745777559073209274[10] = 0;
   out_6745777559073209274[11] = 0;
   out_6745777559073209274[12] = 0;
   out_6745777559073209274[13] = 1;
   out_6745777559073209274[14] = 0;
   out_6745777559073209274[15] = 0;
}
void h_30(double *state, double *unused, double *out_8532523385339766571) {
   out_8532523385339766571[0] = state[4];
}
void H_30(double *state, double *unused, double *out_2577225185951832736) {
   out_2577225185951832736[0] = 0;
   out_2577225185951832736[1] = 0;
   out_2577225185951832736[2] = 0;
   out_2577225185951832736[3] = 0;
   out_2577225185951832736[4] = 1;
   out_2577225185951832736[5] = 0;
   out_2577225185951832736[6] = 0;
   out_2577225185951832736[7] = 0;
}
void h_26(double *state, double *unused, double *out_2234723025716285968) {
   out_2234723025716285968[0] = state[7];
}
void H_26(double *state, double *unused, double *out_4924351094218360094) {
   out_4924351094218360094[0] = 0;
   out_4924351094218360094[1] = 0;
   out_4924351094218360094[2] = 0;
   out_4924351094218360094[3] = 0;
   out_4924351094218360094[4] = 0;
   out_4924351094218360094[5] = 0;
   out_4924351094218360094[6] = 0;
   out_4924351094218360094[7] = 1;
}
void h_27(double *state, double *unused, double *out_5815685395346792460) {
   out_5815685395346792460[0] = state[3];
}
void H_27(double *state, double *unused, double *out_2622377755614588354) {
   out_2622377755614588354[0] = 0;
   out_2622377755614588354[1] = 0;
   out_2622377755614588354[2] = 0;
   out_2622377755614588354[3] = 1;
   out_2622377755614588354[4] = 0;
   out_2622377755614588354[5] = 0;
   out_2622377755614588354[6] = 0;
   out_2622377755614588354[7] = 0;
}
void h_29(double *state, double *unused, double *out_2052512895026669290) {
   out_2052512895026669290[0] = state[1];
}
void H_29(double *state, double *unused, double *out_8160466959469793030) {
   out_8160466959469793030[0] = 0;
   out_8160466959469793030[1] = 1;
   out_8160466959469793030[2] = 0;
   out_8160466959469793030[3] = 0;
   out_8160466959469793030[4] = 0;
   out_8160466959469793030[5] = 0;
   out_8160466959469793030[6] = 0;
   out_8160466959469793030[7] = 0;
}
void h_28(double *state, double *unused, double *out_2232376890138292912) {
   out_2232376890138292912[0] = state[5];
   out_2232376890138292912[1] = state[6];
}
void H_28(double *state, double *unused, double *out_2704987191912913244) {
   out_2704987191912913244[0] = 0;
   out_2704987191912913244[1] = 0;
   out_2704987191912913244[2] = 0;
   out_2704987191912913244[3] = 0;
   out_2704987191912913244[4] = 0;
   out_2704987191912913244[5] = 1;
   out_2704987191912913244[6] = 0;
   out_2704987191912913244[7] = 0;
   out_2704987191912913244[8] = 0;
   out_2704987191912913244[9] = 0;
   out_2704987191912913244[10] = 0;
   out_2704987191912913244[11] = 0;
   out_2704987191912913244[12] = 0;
   out_2704987191912913244[13] = 0;
   out_2704987191912913244[14] = 1;
   out_2704987191912913244[15] = 0;
}
}

extern "C"{
#define DIM 8
#define EDIM 8
#define MEDIM 8
typedef void (*Hfun)(double *, double *, double *);

void predict(double *x, double *P, double *Q, double dt);
const static double MAHA_THRESH_25 = 3.841459;
void update_25(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_24 = 5.991465;
void update_24(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_30 = 3.841459;
void update_30(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_26 = 3.841459;
void update_26(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_27 = 3.841459;
void update_27(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_29 = 3.841459;
void update_29(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_28 = 5.991465;
void update_28(double *, double *, double *, double *, double *);
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



extern "C"{

      void update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<1,3,0>(in_x, in_P, h_25, H_25, NULL, in_z, in_R, in_ea, MAHA_THRESH_25);
      }
    
      void update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<2,3,0>(in_x, in_P, h_24, H_24, NULL, in_z, in_R, in_ea, MAHA_THRESH_24);
      }
    
      void update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<1,3,0>(in_x, in_P, h_30, H_30, NULL, in_z, in_R, in_ea, MAHA_THRESH_30);
      }
    
      void update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<1,3,0>(in_x, in_P, h_26, H_26, NULL, in_z, in_R, in_ea, MAHA_THRESH_26);
      }
    
      void update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<1,3,0>(in_x, in_P, h_27, H_27, NULL, in_z, in_R, in_ea, MAHA_THRESH_27);
      }
    
      void update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<1,3,0>(in_x, in_P, h_29, H_29, NULL, in_z, in_R, in_ea, MAHA_THRESH_29);
      }
    
      void update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
        update<2,3,0>(in_x, in_P, h_28, H_28, NULL, in_z, in_R, in_ea, MAHA_THRESH_28);
      }
    
}
