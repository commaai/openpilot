
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
 *                      Code generated with sympy 1.7.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_8693257057629833746) {
   out_8693257057629833746[0] = delta_x[0] + nom_x[0];
   out_8693257057629833746[1] = delta_x[1] + nom_x[1];
   out_8693257057629833746[2] = delta_x[2] + nom_x[2];
   out_8693257057629833746[3] = delta_x[3] + nom_x[3];
   out_8693257057629833746[4] = delta_x[4] + nom_x[4];
   out_8693257057629833746[5] = delta_x[5] + nom_x[5];
   out_8693257057629833746[6] = delta_x[6] + nom_x[6];
   out_8693257057629833746[7] = delta_x[7] + nom_x[7];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_8135511429353739328) {
   out_8135511429353739328[0] = -nom_x[0] + true_x[0];
   out_8135511429353739328[1] = -nom_x[1] + true_x[1];
   out_8135511429353739328[2] = -nom_x[2] + true_x[2];
   out_8135511429353739328[3] = -nom_x[3] + true_x[3];
   out_8135511429353739328[4] = -nom_x[4] + true_x[4];
   out_8135511429353739328[5] = -nom_x[5] + true_x[5];
   out_8135511429353739328[6] = -nom_x[6] + true_x[6];
   out_8135511429353739328[7] = -nom_x[7] + true_x[7];
}
void H_mod_fun(double *state, double *out_3693155764158352733) {
   out_3693155764158352733[0] = 1.0;
   out_3693155764158352733[1] = 0.0;
   out_3693155764158352733[2] = 0.0;
   out_3693155764158352733[3] = 0.0;
   out_3693155764158352733[4] = 0.0;
   out_3693155764158352733[5] = 0.0;
   out_3693155764158352733[6] = 0.0;
   out_3693155764158352733[7] = 0.0;
   out_3693155764158352733[8] = 0.0;
   out_3693155764158352733[9] = 1.0;
   out_3693155764158352733[10] = 0.0;
   out_3693155764158352733[11] = 0.0;
   out_3693155764158352733[12] = 0.0;
   out_3693155764158352733[13] = 0.0;
   out_3693155764158352733[14] = 0.0;
   out_3693155764158352733[15] = 0.0;
   out_3693155764158352733[16] = 0.0;
   out_3693155764158352733[17] = 0.0;
   out_3693155764158352733[18] = 1.0;
   out_3693155764158352733[19] = 0.0;
   out_3693155764158352733[20] = 0.0;
   out_3693155764158352733[21] = 0.0;
   out_3693155764158352733[22] = 0.0;
   out_3693155764158352733[23] = 0.0;
   out_3693155764158352733[24] = 0.0;
   out_3693155764158352733[25] = 0.0;
   out_3693155764158352733[26] = 0.0;
   out_3693155764158352733[27] = 1.0;
   out_3693155764158352733[28] = 0.0;
   out_3693155764158352733[29] = 0.0;
   out_3693155764158352733[30] = 0.0;
   out_3693155764158352733[31] = 0.0;
   out_3693155764158352733[32] = 0.0;
   out_3693155764158352733[33] = 0.0;
   out_3693155764158352733[34] = 0.0;
   out_3693155764158352733[35] = 0.0;
   out_3693155764158352733[36] = 1.0;
   out_3693155764158352733[37] = 0.0;
   out_3693155764158352733[38] = 0.0;
   out_3693155764158352733[39] = 0.0;
   out_3693155764158352733[40] = 0.0;
   out_3693155764158352733[41] = 0.0;
   out_3693155764158352733[42] = 0.0;
   out_3693155764158352733[43] = 0.0;
   out_3693155764158352733[44] = 0.0;
   out_3693155764158352733[45] = 1.0;
   out_3693155764158352733[46] = 0.0;
   out_3693155764158352733[47] = 0.0;
   out_3693155764158352733[48] = 0.0;
   out_3693155764158352733[49] = 0.0;
   out_3693155764158352733[50] = 0.0;
   out_3693155764158352733[51] = 0.0;
   out_3693155764158352733[52] = 0.0;
   out_3693155764158352733[53] = 0.0;
   out_3693155764158352733[54] = 1.0;
   out_3693155764158352733[55] = 0.0;
   out_3693155764158352733[56] = 0.0;
   out_3693155764158352733[57] = 0.0;
   out_3693155764158352733[58] = 0.0;
   out_3693155764158352733[59] = 0.0;
   out_3693155764158352733[60] = 0.0;
   out_3693155764158352733[61] = 0.0;
   out_3693155764158352733[62] = 0.0;
   out_3693155764158352733[63] = 1.0;
}
void f_fun(double *state, double dt, double *out_3062127823688626291) {
   out_3062127823688626291[0] = state[0];
   out_3062127823688626291[1] = state[1];
   out_3062127823688626291[2] = state[2];
   out_3062127823688626291[3] = state[3];
   out_3062127823688626291[4] = state[4];
   out_3062127823688626291[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_3062127823688626291[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_3062127823688626291[7] = state[7];
}
void F_fun(double *state, double dt, double *out_3519682903798662874) {
   out_3519682903798662874[0] = 1;
   out_3519682903798662874[1] = 0;
   out_3519682903798662874[2] = 0;
   out_3519682903798662874[3] = 0;
   out_3519682903798662874[4] = 0;
   out_3519682903798662874[5] = 0;
   out_3519682903798662874[6] = 0;
   out_3519682903798662874[7] = 0;
   out_3519682903798662874[8] = 0;
   out_3519682903798662874[9] = 1;
   out_3519682903798662874[10] = 0;
   out_3519682903798662874[11] = 0;
   out_3519682903798662874[12] = 0;
   out_3519682903798662874[13] = 0;
   out_3519682903798662874[14] = 0;
   out_3519682903798662874[15] = 0;
   out_3519682903798662874[16] = 0;
   out_3519682903798662874[17] = 0;
   out_3519682903798662874[18] = 1;
   out_3519682903798662874[19] = 0;
   out_3519682903798662874[20] = 0;
   out_3519682903798662874[21] = 0;
   out_3519682903798662874[22] = 0;
   out_3519682903798662874[23] = 0;
   out_3519682903798662874[24] = 0;
   out_3519682903798662874[25] = 0;
   out_3519682903798662874[26] = 0;
   out_3519682903798662874[27] = 1;
   out_3519682903798662874[28] = 0;
   out_3519682903798662874[29] = 0;
   out_3519682903798662874[30] = 0;
   out_3519682903798662874[31] = 0;
   out_3519682903798662874[32] = 0;
   out_3519682903798662874[33] = 0;
   out_3519682903798662874[34] = 0;
   out_3519682903798662874[35] = 0;
   out_3519682903798662874[36] = 1;
   out_3519682903798662874[37] = 0;
   out_3519682903798662874[38] = 0;
   out_3519682903798662874[39] = 0;
   out_3519682903798662874[40] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_3519682903798662874[41] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_3519682903798662874[42] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_3519682903798662874[43] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_3519682903798662874[44] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_3519682903798662874[45] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_3519682903798662874[46] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_3519682903798662874[47] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_3519682903798662874[48] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_3519682903798662874[49] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_3519682903798662874[50] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3519682903798662874[51] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3519682903798662874[52] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_3519682903798662874[53] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_3519682903798662874[54] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_3519682903798662874[55] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3519682903798662874[56] = 0;
   out_3519682903798662874[57] = 0;
   out_3519682903798662874[58] = 0;
   out_3519682903798662874[59] = 0;
   out_3519682903798662874[60] = 0;
   out_3519682903798662874[61] = 0;
   out_3519682903798662874[62] = 0;
   out_3519682903798662874[63] = 1;
}
void h_25(double *state, double *unused, double *out_4440999344614594369) {
   out_4440999344614594369[0] = state[6];
}
void H_25(double *state, double *unused, double *out_6446149650547874996) {
   out_6446149650547874996[0] = 0;
   out_6446149650547874996[1] = 0;
   out_6446149650547874996[2] = 0;
   out_6446149650547874996[3] = 0;
   out_6446149650547874996[4] = 0;
   out_6446149650547874996[5] = 0;
   out_6446149650547874996[6] = 1;
   out_6446149650547874996[7] = 0;
}
void h_24(double *state, double *unused, double *out_8962308475132917488) {
   out_8962308475132917488[0] = state[4];
   out_8962308475132917488[1] = state[5];
}
void H_24(double *state, double *unused, double *out_2050419563909869434) {
   out_2050419563909869434[0] = 0;
   out_2050419563909869434[1] = 0;
   out_2050419563909869434[2] = 0;
   out_2050419563909869434[3] = 0;
   out_2050419563909869434[4] = 1;
   out_2050419563909869434[5] = 0;
   out_2050419563909869434[6] = 0;
   out_2050419563909869434[7] = 0;
   out_2050419563909869434[8] = 0;
   out_2050419563909869434[9] = 0;
   out_2050419563909869434[10] = 0;
   out_2050419563909869434[11] = 0;
   out_2050419563909869434[12] = 0;
   out_2050419563909869434[13] = 1;
   out_2050419563909869434[14] = 0;
   out_2050419563909869434[15] = 0;
}
void h_30(double *state, double *unused, double *out_4888601670002484904) {
   out_4888601670002484904[0] = state[4];
}
void H_30(double *state, double *unused, double *out_3167749260401308284) {
   out_3167749260401308284[0] = 0;
   out_3167749260401308284[1] = 0;
   out_3167749260401308284[2] = 0;
   out_3167749260401308284[3] = 0;
   out_3167749260401308284[4] = 1;
   out_3167749260401308284[5] = 0;
   out_3167749260401308284[6] = 0;
   out_3167749260401308284[7] = 0;
}
void h_26(double *state, double *unused, double *out_8556469136046221067) {
   out_8556469136046221067[0] = state[7];
}
void H_26(double *state, double *unused, double *out_6296001268222467347) {
   out_6296001268222467347[0] = 0;
   out_6296001268222467347[1] = 0;
   out_6296001268222467347[2] = 0;
   out_6296001268222467347[3] = 0;
   out_6296001268222467347[4] = 0;
   out_6296001268222467347[5] = 0;
   out_6296001268222467347[6] = 0;
   out_6296001268222467347[7] = 1;
}
void h_27(double *state, double *unused, double *out_5423083437949892481) {
   out_5423083437949892481[0] = state[3];
}
void H_27(double *state, double *unused, double *out_4455331248237933596) {
   out_4455331248237933596[0] = 0;
   out_4455331248237933596[1] = 0;
   out_4455331248237933596[2] = 0;
   out_4455331248237933596[3] = 1;
   out_4455331248237933596[4] = 0;
   out_4455331248237933596[5] = 0;
   out_4455331248237933596[6] = 0;
   out_4455331248237933596[7] = 0;
}
void h_29(double *state, double *unused, double *out_1903641303629246875) {
   out_1903641303629246875[0] = state[1];
}
void H_29(double *state, double *unused, double *out_2593190646545772109) {
   out_2593190646545772109[0] = 0;
   out_2593190646545772109[1] = 1;
   out_2593190646545772109[2] = 0;
   out_2593190646545772109[3] = 0;
   out_2593190646545772109[4] = 0;
   out_2593190646545772109[5] = 0;
   out_2593190646545772109[6] = 0;
   out_2593190646545772109[7] = 0;
}
void h_28(double *state, double *unused, double *out_1463873990780735927) {
   out_1463873990780735927[0] = state[5];
   out_1463873990780735927[1] = state[6];
}
void H_28(double *state, double *unused, double *out_6433301870748069359) {
   out_6433301870748069359[0] = 0;
   out_6433301870748069359[1] = 0;
   out_6433301870748069359[2] = 0;
   out_6433301870748069359[3] = 0;
   out_6433301870748069359[4] = 0;
   out_6433301870748069359[5] = 1;
   out_6433301870748069359[6] = 0;
   out_6433301870748069359[7] = 0;
   out_6433301870748069359[8] = 0;
   out_6433301870748069359[9] = 0;
   out_6433301870748069359[10] = 0;
   out_6433301870748069359[11] = 0;
   out_6433301870748069359[12] = 0;
   out_6433301870748069359[13] = 0;
   out_6433301870748069359[14] = 1;
   out_6433301870748069359[15] = 0;
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
