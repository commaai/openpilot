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
void err_fun(double *nom_x, double *delta_x, double *out_6490269132271391301) {
   out_6490269132271391301[0] = delta_x[0] + nom_x[0];
   out_6490269132271391301[1] = delta_x[1] + nom_x[1];
   out_6490269132271391301[2] = delta_x[2] + nom_x[2];
   out_6490269132271391301[3] = delta_x[3] + nom_x[3];
   out_6490269132271391301[4] = delta_x[4] + nom_x[4];
   out_6490269132271391301[5] = delta_x[5] + nom_x[5];
   out_6490269132271391301[6] = delta_x[6] + nom_x[6];
   out_6490269132271391301[7] = delta_x[7] + nom_x[7];
   out_6490269132271391301[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_7265692863643532060) {
   out_7265692863643532060[0] = -nom_x[0] + true_x[0];
   out_7265692863643532060[1] = -nom_x[1] + true_x[1];
   out_7265692863643532060[2] = -nom_x[2] + true_x[2];
   out_7265692863643532060[3] = -nom_x[3] + true_x[3];
   out_7265692863643532060[4] = -nom_x[4] + true_x[4];
   out_7265692863643532060[5] = -nom_x[5] + true_x[5];
   out_7265692863643532060[6] = -nom_x[6] + true_x[6];
   out_7265692863643532060[7] = -nom_x[7] + true_x[7];
   out_7265692863643532060[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_5640827172746054554) {
   out_5640827172746054554[0] = 1.0;
   out_5640827172746054554[1] = 0.0;
   out_5640827172746054554[2] = 0.0;
   out_5640827172746054554[3] = 0.0;
   out_5640827172746054554[4] = 0.0;
   out_5640827172746054554[5] = 0.0;
   out_5640827172746054554[6] = 0.0;
   out_5640827172746054554[7] = 0.0;
   out_5640827172746054554[8] = 0.0;
   out_5640827172746054554[9] = 0.0;
   out_5640827172746054554[10] = 1.0;
   out_5640827172746054554[11] = 0.0;
   out_5640827172746054554[12] = 0.0;
   out_5640827172746054554[13] = 0.0;
   out_5640827172746054554[14] = 0.0;
   out_5640827172746054554[15] = 0.0;
   out_5640827172746054554[16] = 0.0;
   out_5640827172746054554[17] = 0.0;
   out_5640827172746054554[18] = 0.0;
   out_5640827172746054554[19] = 0.0;
   out_5640827172746054554[20] = 1.0;
   out_5640827172746054554[21] = 0.0;
   out_5640827172746054554[22] = 0.0;
   out_5640827172746054554[23] = 0.0;
   out_5640827172746054554[24] = 0.0;
   out_5640827172746054554[25] = 0.0;
   out_5640827172746054554[26] = 0.0;
   out_5640827172746054554[27] = 0.0;
   out_5640827172746054554[28] = 0.0;
   out_5640827172746054554[29] = 0.0;
   out_5640827172746054554[30] = 1.0;
   out_5640827172746054554[31] = 0.0;
   out_5640827172746054554[32] = 0.0;
   out_5640827172746054554[33] = 0.0;
   out_5640827172746054554[34] = 0.0;
   out_5640827172746054554[35] = 0.0;
   out_5640827172746054554[36] = 0.0;
   out_5640827172746054554[37] = 0.0;
   out_5640827172746054554[38] = 0.0;
   out_5640827172746054554[39] = 0.0;
   out_5640827172746054554[40] = 1.0;
   out_5640827172746054554[41] = 0.0;
   out_5640827172746054554[42] = 0.0;
   out_5640827172746054554[43] = 0.0;
   out_5640827172746054554[44] = 0.0;
   out_5640827172746054554[45] = 0.0;
   out_5640827172746054554[46] = 0.0;
   out_5640827172746054554[47] = 0.0;
   out_5640827172746054554[48] = 0.0;
   out_5640827172746054554[49] = 0.0;
   out_5640827172746054554[50] = 1.0;
   out_5640827172746054554[51] = 0.0;
   out_5640827172746054554[52] = 0.0;
   out_5640827172746054554[53] = 0.0;
   out_5640827172746054554[54] = 0.0;
   out_5640827172746054554[55] = 0.0;
   out_5640827172746054554[56] = 0.0;
   out_5640827172746054554[57] = 0.0;
   out_5640827172746054554[58] = 0.0;
   out_5640827172746054554[59] = 0.0;
   out_5640827172746054554[60] = 1.0;
   out_5640827172746054554[61] = 0.0;
   out_5640827172746054554[62] = 0.0;
   out_5640827172746054554[63] = 0.0;
   out_5640827172746054554[64] = 0.0;
   out_5640827172746054554[65] = 0.0;
   out_5640827172746054554[66] = 0.0;
   out_5640827172746054554[67] = 0.0;
   out_5640827172746054554[68] = 0.0;
   out_5640827172746054554[69] = 0.0;
   out_5640827172746054554[70] = 1.0;
   out_5640827172746054554[71] = 0.0;
   out_5640827172746054554[72] = 0.0;
   out_5640827172746054554[73] = 0.0;
   out_5640827172746054554[74] = 0.0;
   out_5640827172746054554[75] = 0.0;
   out_5640827172746054554[76] = 0.0;
   out_5640827172746054554[77] = 0.0;
   out_5640827172746054554[78] = 0.0;
   out_5640827172746054554[79] = 0.0;
   out_5640827172746054554[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_3709053776412774313) {
   out_3709053776412774313[0] = state[0];
   out_3709053776412774313[1] = state[1];
   out_3709053776412774313[2] = state[2];
   out_3709053776412774313[3] = state[3];
   out_3709053776412774313[4] = state[4];
   out_3709053776412774313[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_3709053776412774313[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_3709053776412774313[7] = state[7];
   out_3709053776412774313[8] = state[8];
}
void F_fun(double *state, double dt, double *out_3639689452659227219) {
   out_3639689452659227219[0] = 1;
   out_3639689452659227219[1] = 0;
   out_3639689452659227219[2] = 0;
   out_3639689452659227219[3] = 0;
   out_3639689452659227219[4] = 0;
   out_3639689452659227219[5] = 0;
   out_3639689452659227219[6] = 0;
   out_3639689452659227219[7] = 0;
   out_3639689452659227219[8] = 0;
   out_3639689452659227219[9] = 0;
   out_3639689452659227219[10] = 1;
   out_3639689452659227219[11] = 0;
   out_3639689452659227219[12] = 0;
   out_3639689452659227219[13] = 0;
   out_3639689452659227219[14] = 0;
   out_3639689452659227219[15] = 0;
   out_3639689452659227219[16] = 0;
   out_3639689452659227219[17] = 0;
   out_3639689452659227219[18] = 0;
   out_3639689452659227219[19] = 0;
   out_3639689452659227219[20] = 1;
   out_3639689452659227219[21] = 0;
   out_3639689452659227219[22] = 0;
   out_3639689452659227219[23] = 0;
   out_3639689452659227219[24] = 0;
   out_3639689452659227219[25] = 0;
   out_3639689452659227219[26] = 0;
   out_3639689452659227219[27] = 0;
   out_3639689452659227219[28] = 0;
   out_3639689452659227219[29] = 0;
   out_3639689452659227219[30] = 1;
   out_3639689452659227219[31] = 0;
   out_3639689452659227219[32] = 0;
   out_3639689452659227219[33] = 0;
   out_3639689452659227219[34] = 0;
   out_3639689452659227219[35] = 0;
   out_3639689452659227219[36] = 0;
   out_3639689452659227219[37] = 0;
   out_3639689452659227219[38] = 0;
   out_3639689452659227219[39] = 0;
   out_3639689452659227219[40] = 1;
   out_3639689452659227219[41] = 0;
   out_3639689452659227219[42] = 0;
   out_3639689452659227219[43] = 0;
   out_3639689452659227219[44] = 0;
   out_3639689452659227219[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_3639689452659227219[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_3639689452659227219[47] = -dt*stiffness_front*state[0]/state[1];
   out_3639689452659227219[48] = -dt*stiffness_front*state[0]/state[1];
   out_3639689452659227219[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_3639689452659227219[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_3639689452659227219[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_3639689452659227219[52] = dt*stiffness_front*state[0]/state[1];
   out_3639689452659227219[53] = -9.8100000000000005*dt;
   out_3639689452659227219[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_3639689452659227219[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_3639689452659227219[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3639689452659227219[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3639689452659227219[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_3639689452659227219[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_3639689452659227219[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_3639689452659227219[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_3639689452659227219[62] = 0;
   out_3639689452659227219[63] = 0;
   out_3639689452659227219[64] = 0;
   out_3639689452659227219[65] = 0;
   out_3639689452659227219[66] = 0;
   out_3639689452659227219[67] = 0;
   out_3639689452659227219[68] = 0;
   out_3639689452659227219[69] = 0;
   out_3639689452659227219[70] = 1;
   out_3639689452659227219[71] = 0;
   out_3639689452659227219[72] = 0;
   out_3639689452659227219[73] = 0;
   out_3639689452659227219[74] = 0;
   out_3639689452659227219[75] = 0;
   out_3639689452659227219[76] = 0;
   out_3639689452659227219[77] = 0;
   out_3639689452659227219[78] = 0;
   out_3639689452659227219[79] = 0;
   out_3639689452659227219[80] = 1;
}
void h_25(double *state, double *unused, double *out_2141167549731806248) {
   out_2141167549731806248[0] = state[6];
}
void H_25(double *state, double *unused, double *out_3612728675051259825) {
   out_3612728675051259825[0] = 0;
   out_3612728675051259825[1] = 0;
   out_3612728675051259825[2] = 0;
   out_3612728675051259825[3] = 0;
   out_3612728675051259825[4] = 0;
   out_3612728675051259825[5] = 0;
   out_3612728675051259825[6] = 1;
   out_3612728675051259825[7] = 0;
   out_3612728675051259825[8] = 0;
}
void h_24(double *state, double *unused, double *out_6454025621392139070) {
   out_6454025621392139070[0] = state[4];
   out_6454025621392139070[1] = state[5];
}
void H_24(double *state, double *unused, double *out_1440079076045760259) {
   out_1440079076045760259[0] = 0;
   out_1440079076045760259[1] = 0;
   out_1440079076045760259[2] = 0;
   out_1440079076045760259[3] = 0;
   out_1440079076045760259[4] = 1;
   out_1440079076045760259[5] = 0;
   out_1440079076045760259[6] = 0;
   out_1440079076045760259[7] = 0;
   out_1440079076045760259[8] = 0;
   out_1440079076045760259[9] = 0;
   out_1440079076045760259[10] = 0;
   out_1440079076045760259[11] = 0;
   out_1440079076045760259[12] = 0;
   out_1440079076045760259[13] = 0;
   out_1440079076045760259[14] = 1;
   out_1440079076045760259[15] = 0;
   out_1440079076045760259[16] = 0;
   out_1440079076045760259[17] = 0;
}
void h_30(double *state, double *unused, double *out_2298354218082935393) {
   out_2298354218082935393[0] = state[4];
}
void H_30(double *state, double *unused, double *out_7917325057166675036) {
   out_7917325057166675036[0] = 0;
   out_7917325057166675036[1] = 0;
   out_7917325057166675036[2] = 0;
   out_7917325057166675036[3] = 0;
   out_7917325057166675036[4] = 1;
   out_7917325057166675036[5] = 0;
   out_7917325057166675036[6] = 0;
   out_7917325057166675036[7] = 0;
   out_7917325057166675036[8] = 0;
}
void h_26(double *state, double *unused, double *out_514197885186917990) {
   out_514197885186917990[0] = state[7];
}
void H_26(double *state, double *unused, double *out_128774643822796399) {
   out_128774643822796399[0] = 0;
   out_128774643822796399[1] = 0;
   out_128774643822796399[2] = 0;
   out_128774643822796399[3] = 0;
   out_128774643822796399[4] = 0;
   out_128774643822796399[5] = 0;
   out_128774643822796399[6] = 0;
   out_128774643822796399[7] = 1;
   out_128774643822796399[8] = 0;
}
void h_27(double *state, double *unused, double *out_3304737207191273806) {
   out_3304737207191273806[0] = state[3];
}
void H_27(double *state, double *unused, double *out_1308626416107594844) {
   out_1308626416107594844[0] = 0;
   out_1308626416107594844[1] = 0;
   out_1308626416107594844[2] = 0;
   out_1308626416107594844[3] = 1;
   out_1308626416107594844[4] = 0;
   out_1308626416107594844[5] = 0;
   out_1308626416107594844[6] = 0;
   out_1308626416107594844[7] = 0;
   out_1308626416107594844[8] = 0;
}
void h_29(double *state, double *unused, double *out_3466098019159077130) {
   out_3466098019159077130[0] = state[1];
}
void H_29(double *state, double *unused, double *out_404736310761956189) {
   out_404736310761956189[0] = 0;
   out_404736310761956189[1] = 1;
   out_404736310761956189[2] = 0;
   out_404736310761956189[3] = 0;
   out_404736310761956189[4] = 0;
   out_404736310761956189[5] = 0;
   out_404736310761956189[6] = 0;
   out_404736310761956189[7] = 0;
   out_404736310761956189[8] = 0;
}
void h_28(double *state, double *unused, double *out_7926923236647382120) {
   out_7926923236647382120[0] = state[0];
}
void H_28(double *state, double *unused, double *out_1558893960803370062) {
   out_1558893960803370062[0] = 1;
   out_1558893960803370062[1] = 0;
   out_1558893960803370062[2] = 0;
   out_1558893960803370062[3] = 0;
   out_1558893960803370062[4] = 0;
   out_1558893960803370062[5] = 0;
   out_1558893960803370062[6] = 0;
   out_1558893960803370062[7] = 0;
   out_1558893960803370062[8] = 0;
}
void h_31(double *state, double *unused, double *out_6066329196351100602) {
   out_6066329196351100602[0] = state[8];
}
void H_31(double *state, double *unused, double *out_3643374636928220253) {
   out_3643374636928220253[0] = 0;
   out_3643374636928220253[1] = 0;
   out_3643374636928220253[2] = 0;
   out_3643374636928220253[3] = 0;
   out_3643374636928220253[4] = 0;
   out_3643374636928220253[5] = 0;
   out_3643374636928220253[6] = 0;
   out_3643374636928220253[7] = 0;
   out_3643374636928220253[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_6490269132271391301) {
  err_fun(nom_x, delta_x, out_6490269132271391301);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7265692863643532060) {
  inv_err_fun(nom_x, true_x, out_7265692863643532060);
}
void car_H_mod_fun(double *state, double *out_5640827172746054554) {
  H_mod_fun(state, out_5640827172746054554);
}
void car_f_fun(double *state, double dt, double *out_3709053776412774313) {
  f_fun(state,  dt, out_3709053776412774313);
}
void car_F_fun(double *state, double dt, double *out_3639689452659227219) {
  F_fun(state,  dt, out_3639689452659227219);
}
void car_h_25(double *state, double *unused, double *out_2141167549731806248) {
  h_25(state, unused, out_2141167549731806248);
}
void car_H_25(double *state, double *unused, double *out_3612728675051259825) {
  H_25(state, unused, out_3612728675051259825);
}
void car_h_24(double *state, double *unused, double *out_6454025621392139070) {
  h_24(state, unused, out_6454025621392139070);
}
void car_H_24(double *state, double *unused, double *out_1440079076045760259) {
  H_24(state, unused, out_1440079076045760259);
}
void car_h_30(double *state, double *unused, double *out_2298354218082935393) {
  h_30(state, unused, out_2298354218082935393);
}
void car_H_30(double *state, double *unused, double *out_7917325057166675036) {
  H_30(state, unused, out_7917325057166675036);
}
void car_h_26(double *state, double *unused, double *out_514197885186917990) {
  h_26(state, unused, out_514197885186917990);
}
void car_H_26(double *state, double *unused, double *out_128774643822796399) {
  H_26(state, unused, out_128774643822796399);
}
void car_h_27(double *state, double *unused, double *out_3304737207191273806) {
  h_27(state, unused, out_3304737207191273806);
}
void car_H_27(double *state, double *unused, double *out_1308626416107594844) {
  H_27(state, unused, out_1308626416107594844);
}
void car_h_29(double *state, double *unused, double *out_3466098019159077130) {
  h_29(state, unused, out_3466098019159077130);
}
void car_H_29(double *state, double *unused, double *out_404736310761956189) {
  H_29(state, unused, out_404736310761956189);
}
void car_h_28(double *state, double *unused, double *out_7926923236647382120) {
  h_28(state, unused, out_7926923236647382120);
}
void car_H_28(double *state, double *unused, double *out_1558893960803370062) {
  H_28(state, unused, out_1558893960803370062);
}
void car_h_31(double *state, double *unused, double *out_6066329196351100602) {
  h_31(state, unused, out_6066329196351100602);
}
void car_H_31(double *state, double *unused, double *out_3643374636928220253) {
  H_31(state, unused, out_3643374636928220253);
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
