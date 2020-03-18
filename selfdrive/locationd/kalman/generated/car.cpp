
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
void err_fun(double *nom_x, double *delta_x, double *out_7794183793822969988) {
   out_7794183793822969988[0] = delta_x[0] + nom_x[0];
   out_7794183793822969988[1] = delta_x[1] + nom_x[1];
   out_7794183793822969988[2] = delta_x[2] + nom_x[2];
   out_7794183793822969988[3] = delta_x[3] + nom_x[3];
   out_7794183793822969988[4] = delta_x[4] + nom_x[4];
   out_7794183793822969988[5] = delta_x[5] + nom_x[5];
   out_7794183793822969988[6] = delta_x[6] + nom_x[6];
   out_7794183793822969988[7] = delta_x[7] + nom_x[7];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_7355221333369501264) {
   out_7355221333369501264[0] = -nom_x[0] + true_x[0];
   out_7355221333369501264[1] = -nom_x[1] + true_x[1];
   out_7355221333369501264[2] = -nom_x[2] + true_x[2];
   out_7355221333369501264[3] = -nom_x[3] + true_x[3];
   out_7355221333369501264[4] = -nom_x[4] + true_x[4];
   out_7355221333369501264[5] = -nom_x[5] + true_x[5];
   out_7355221333369501264[6] = -nom_x[6] + true_x[6];
   out_7355221333369501264[7] = -nom_x[7] + true_x[7];
}
void H_mod_fun(double *state, double *out_5744445918609299259) {
   out_5744445918609299259[0] = 1.0;
   out_5744445918609299259[1] = 0.0;
   out_5744445918609299259[2] = 0.0;
   out_5744445918609299259[3] = 0.0;
   out_5744445918609299259[4] = 0.0;
   out_5744445918609299259[5] = 0.0;
   out_5744445918609299259[6] = 0.0;
   out_5744445918609299259[7] = 0.0;
   out_5744445918609299259[8] = 0.0;
   out_5744445918609299259[9] = 1.0;
   out_5744445918609299259[10] = 0.0;
   out_5744445918609299259[11] = 0.0;
   out_5744445918609299259[12] = 0.0;
   out_5744445918609299259[13] = 0.0;
   out_5744445918609299259[14] = 0.0;
   out_5744445918609299259[15] = 0.0;
   out_5744445918609299259[16] = 0.0;
   out_5744445918609299259[17] = 0.0;
   out_5744445918609299259[18] = 1.0;
   out_5744445918609299259[19] = 0.0;
   out_5744445918609299259[20] = 0.0;
   out_5744445918609299259[21] = 0.0;
   out_5744445918609299259[22] = 0.0;
   out_5744445918609299259[23] = 0.0;
   out_5744445918609299259[24] = 0.0;
   out_5744445918609299259[25] = 0.0;
   out_5744445918609299259[26] = 0.0;
   out_5744445918609299259[27] = 1.0;
   out_5744445918609299259[28] = 0.0;
   out_5744445918609299259[29] = 0.0;
   out_5744445918609299259[30] = 0.0;
   out_5744445918609299259[31] = 0.0;
   out_5744445918609299259[32] = 0.0;
   out_5744445918609299259[33] = 0.0;
   out_5744445918609299259[34] = 0.0;
   out_5744445918609299259[35] = 0.0;
   out_5744445918609299259[36] = 1.0;
   out_5744445918609299259[37] = 0.0;
   out_5744445918609299259[38] = 0.0;
   out_5744445918609299259[39] = 0.0;
   out_5744445918609299259[40] = 0.0;
   out_5744445918609299259[41] = 0.0;
   out_5744445918609299259[42] = 0.0;
   out_5744445918609299259[43] = 0.0;
   out_5744445918609299259[44] = 0.0;
   out_5744445918609299259[45] = 1.0;
   out_5744445918609299259[46] = 0.0;
   out_5744445918609299259[47] = 0.0;
   out_5744445918609299259[48] = 0.0;
   out_5744445918609299259[49] = 0.0;
   out_5744445918609299259[50] = 0.0;
   out_5744445918609299259[51] = 0.0;
   out_5744445918609299259[52] = 0.0;
   out_5744445918609299259[53] = 0.0;
   out_5744445918609299259[54] = 1.0;
   out_5744445918609299259[55] = 0.0;
   out_5744445918609299259[56] = 0.0;
   out_5744445918609299259[57] = 0.0;
   out_5744445918609299259[58] = 0.0;
   out_5744445918609299259[59] = 0.0;
   out_5744445918609299259[60] = 0.0;
   out_5744445918609299259[61] = 0.0;
   out_5744445918609299259[62] = 0.0;
   out_5744445918609299259[63] = 1.0;
}
void f_fun(double *state, double dt, double *out_9039064146508570210) {
   out_9039064146508570210[0] = state[0];
   out_9039064146508570210[1] = state[1];
   out_9039064146508570210[2] = state[2];
   out_9039064146508570210[3] = state[3];
   out_9039064146508570210[4] = state[4];
   out_9039064146508570210[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_9039064146508570210[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_9039064146508570210[7] = state[7];
}
void F_fun(double *state, double dt, double *out_8072223598153394438) {
   out_8072223598153394438[0] = 1;
   out_8072223598153394438[1] = 0;
   out_8072223598153394438[2] = 0;
   out_8072223598153394438[3] = 0;
   out_8072223598153394438[4] = 0;
   out_8072223598153394438[5] = 0;
   out_8072223598153394438[6] = 0;
   out_8072223598153394438[7] = 0;
   out_8072223598153394438[8] = 0;
   out_8072223598153394438[9] = 1;
   out_8072223598153394438[10] = 0;
   out_8072223598153394438[11] = 0;
   out_8072223598153394438[12] = 0;
   out_8072223598153394438[13] = 0;
   out_8072223598153394438[14] = 0;
   out_8072223598153394438[15] = 0;
   out_8072223598153394438[16] = 0;
   out_8072223598153394438[17] = 0;
   out_8072223598153394438[18] = 1;
   out_8072223598153394438[19] = 0;
   out_8072223598153394438[20] = 0;
   out_8072223598153394438[21] = 0;
   out_8072223598153394438[22] = 0;
   out_8072223598153394438[23] = 0;
   out_8072223598153394438[24] = 0;
   out_8072223598153394438[25] = 0;
   out_8072223598153394438[26] = 0;
   out_8072223598153394438[27] = 1;
   out_8072223598153394438[28] = 0;
   out_8072223598153394438[29] = 0;
   out_8072223598153394438[30] = 0;
   out_8072223598153394438[31] = 0;
   out_8072223598153394438[32] = 0;
   out_8072223598153394438[33] = 0;
   out_8072223598153394438[34] = 0;
   out_8072223598153394438[35] = 0;
   out_8072223598153394438[36] = 1;
   out_8072223598153394438[37] = 0;
   out_8072223598153394438[38] = 0;
   out_8072223598153394438[39] = 0;
   out_8072223598153394438[40] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_8072223598153394438[41] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_8072223598153394438[42] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8072223598153394438[43] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_8072223598153394438[44] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_8072223598153394438[45] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_8072223598153394438[46] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_8072223598153394438[47] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_8072223598153394438[48] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_8072223598153394438[49] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_8072223598153394438[50] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8072223598153394438[51] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8072223598153394438[52] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_8072223598153394438[53] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_8072223598153394438[54] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_8072223598153394438[55] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8072223598153394438[56] = 0;
   out_8072223598153394438[57] = 0;
   out_8072223598153394438[58] = 0;
   out_8072223598153394438[59] = 0;
   out_8072223598153394438[60] = 0;
   out_8072223598153394438[61] = 0;
   out_8072223598153394438[62] = 0;
   out_8072223598153394438[63] = 1;
}
void h_25(double *state, double *unused, double *out_4065255527585992767) {
   out_4065255527585992767[0] = state[6];
}
void H_25(double *state, double *unused, double *out_8165269990197921171) {
   out_8165269990197921171[0] = 0;
   out_8165269990197921171[1] = 0;
   out_8165269990197921171[2] = 0;
   out_8165269990197921171[3] = 0;
   out_8165269990197921171[4] = 0;
   out_8165269990197921171[5] = 0;
   out_8165269990197921171[6] = 1;
   out_8165269990197921171[7] = 0;
}
void h_24(double *state, double *unused, double *out_1976043640039013749) {
   out_1976043640039013749[0] = state[4];
   out_1976043640039013749[1] = state[5];
}
void H_24(double *state, double *unused, double *out_413298986514378767) {
   out_413298986514378767[0] = 0;
   out_413298986514378767[1] = 0;
   out_413298986514378767[2] = 0;
   out_413298986514378767[3] = 0;
   out_413298986514378767[4] = 1;
   out_413298986514378767[5] = 0;
   out_413298986514378767[6] = 0;
   out_413298986514378767[7] = 0;
   out_413298986514378767[8] = 0;
   out_413298986514378767[9] = 0;
   out_413298986514378767[10] = 0;
   out_413298986514378767[11] = 0;
   out_413298986514378767[12] = 0;
   out_413298986514378767[13] = 1;
   out_413298986514378767[14] = 0;
   out_413298986514378767[15] = 0;
}
void h_26(double *state, double *unused, double *out_8261513777496151526) {
   out_8261513777496151526[0] = state[7];
}
void H_26(double *state, double *unused, double *out_8923028299312733787) {
   out_8923028299312733787[0] = 0;
   out_8923028299312733787[1] = 0;
   out_8923028299312733787[2] = 0;
   out_8923028299312733787[3] = 0;
   out_8923028299312733787[4] = 0;
   out_8923028299312733787[5] = 0;
   out_8923028299312733787[6] = 0;
   out_8923028299312733787[7] = 1;
}
void h_27(double *state, double *unused, double *out_2457131146622328962) {
   out_2457131146622328962[0] = state[3];
}
void H_27(double *state, double *unused, double *out_1366834396023443895) {
   out_1366834396023443895[0] = 0;
   out_1366834396023443895[1] = 0;
   out_1366834396023443895[2] = 0;
   out_1366834396023443895[3] = 1;
   out_1366834396023443895[4] = 0;
   out_1366834396023443895[5] = 0;
   out_1366834396023443895[6] = 0;
   out_1366834396023443895[7] = 0;
}
void h_29(double *state, double *unused, double *out_189784329795850816) {
   out_189784329795850816[0] = state[1];
}
void H_29(double *state, double *unused, double *out_5664192392506963901) {
   out_5664192392506963901[0] = 0;
   out_5664192392506963901[1] = 1;
   out_5664192392506963901[2] = 0;
   out_5664192392506963901[3] = 0;
   out_5664192392506963901[4] = 0;
   out_5664192392506963901[5] = 0;
   out_5664192392506963901[6] = 0;
   out_5664192392506963901[7] = 0;
}
void h_28(double *state, double *unused, double *out_5232440403228794693) {
   out_5232440403228794693[0] = state[5];
   out_5232440403228794693[1] = state[6];
}
void H_28(double *state, double *unused, double *out_1801946368970198955) {
   out_1801946368970198955[0] = 0;
   out_1801946368970198955[1] = 0;
   out_1801946368970198955[2] = 0;
   out_1801946368970198955[3] = 0;
   out_1801946368970198955[4] = 0;
   out_1801946368970198955[5] = 1;
   out_1801946368970198955[6] = 0;
   out_1801946368970198955[7] = 0;
   out_1801946368970198955[8] = 0;
   out_1801946368970198955[9] = 0;
   out_1801946368970198955[10] = 0;
   out_1801946368970198955[11] = 0;
   out_1801946368970198955[12] = 0;
   out_1801946368970198955[13] = 0;
   out_1801946368970198955[14] = 1;
   out_1801946368970198955[15] = 0;
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
