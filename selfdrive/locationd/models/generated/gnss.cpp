#include "gnss.h"

namespace {
#define DIM 11
#define EDIM 11
#define MEDIM 11
typedef void (*Hfun)(double *, double *, double *);
const static double MAHA_THRESH_6 = 3.8414588206941227;
const static double MAHA_THRESH_20 = 3.8414588206941227;
const static double MAHA_THRESH_7 = 3.8414588206941227;
const static double MAHA_THRESH_21 = 3.8414588206941227;

/******************************************************************************
 *                      Code generated with SymPy 1.11.1                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_1108434360080389513) {
   out_1108434360080389513[0] = delta_x[0] + nom_x[0];
   out_1108434360080389513[1] = delta_x[1] + nom_x[1];
   out_1108434360080389513[2] = delta_x[2] + nom_x[2];
   out_1108434360080389513[3] = delta_x[3] + nom_x[3];
   out_1108434360080389513[4] = delta_x[4] + nom_x[4];
   out_1108434360080389513[5] = delta_x[5] + nom_x[5];
   out_1108434360080389513[6] = delta_x[6] + nom_x[6];
   out_1108434360080389513[7] = delta_x[7] + nom_x[7];
   out_1108434360080389513[8] = delta_x[8] + nom_x[8];
   out_1108434360080389513[9] = delta_x[9] + nom_x[9];
   out_1108434360080389513[10] = delta_x[10] + nom_x[10];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_6163332558245412373) {
   out_6163332558245412373[0] = -nom_x[0] + true_x[0];
   out_6163332558245412373[1] = -nom_x[1] + true_x[1];
   out_6163332558245412373[2] = -nom_x[2] + true_x[2];
   out_6163332558245412373[3] = -nom_x[3] + true_x[3];
   out_6163332558245412373[4] = -nom_x[4] + true_x[4];
   out_6163332558245412373[5] = -nom_x[5] + true_x[5];
   out_6163332558245412373[6] = -nom_x[6] + true_x[6];
   out_6163332558245412373[7] = -nom_x[7] + true_x[7];
   out_6163332558245412373[8] = -nom_x[8] + true_x[8];
   out_6163332558245412373[9] = -nom_x[9] + true_x[9];
   out_6163332558245412373[10] = -nom_x[10] + true_x[10];
}
void H_mod_fun(double *state, double *out_8120041661689163173) {
   out_8120041661689163173[0] = 1.0;
   out_8120041661689163173[1] = 0;
   out_8120041661689163173[2] = 0;
   out_8120041661689163173[3] = 0;
   out_8120041661689163173[4] = 0;
   out_8120041661689163173[5] = 0;
   out_8120041661689163173[6] = 0;
   out_8120041661689163173[7] = 0;
   out_8120041661689163173[8] = 0;
   out_8120041661689163173[9] = 0;
   out_8120041661689163173[10] = 0;
   out_8120041661689163173[11] = 0;
   out_8120041661689163173[12] = 1.0;
   out_8120041661689163173[13] = 0;
   out_8120041661689163173[14] = 0;
   out_8120041661689163173[15] = 0;
   out_8120041661689163173[16] = 0;
   out_8120041661689163173[17] = 0;
   out_8120041661689163173[18] = 0;
   out_8120041661689163173[19] = 0;
   out_8120041661689163173[20] = 0;
   out_8120041661689163173[21] = 0;
   out_8120041661689163173[22] = 0;
   out_8120041661689163173[23] = 0;
   out_8120041661689163173[24] = 1.0;
   out_8120041661689163173[25] = 0;
   out_8120041661689163173[26] = 0;
   out_8120041661689163173[27] = 0;
   out_8120041661689163173[28] = 0;
   out_8120041661689163173[29] = 0;
   out_8120041661689163173[30] = 0;
   out_8120041661689163173[31] = 0;
   out_8120041661689163173[32] = 0;
   out_8120041661689163173[33] = 0;
   out_8120041661689163173[34] = 0;
   out_8120041661689163173[35] = 0;
   out_8120041661689163173[36] = 1.0;
   out_8120041661689163173[37] = 0;
   out_8120041661689163173[38] = 0;
   out_8120041661689163173[39] = 0;
   out_8120041661689163173[40] = 0;
   out_8120041661689163173[41] = 0;
   out_8120041661689163173[42] = 0;
   out_8120041661689163173[43] = 0;
   out_8120041661689163173[44] = 0;
   out_8120041661689163173[45] = 0;
   out_8120041661689163173[46] = 0;
   out_8120041661689163173[47] = 0;
   out_8120041661689163173[48] = 1.0;
   out_8120041661689163173[49] = 0;
   out_8120041661689163173[50] = 0;
   out_8120041661689163173[51] = 0;
   out_8120041661689163173[52] = 0;
   out_8120041661689163173[53] = 0;
   out_8120041661689163173[54] = 0;
   out_8120041661689163173[55] = 0;
   out_8120041661689163173[56] = 0;
   out_8120041661689163173[57] = 0;
   out_8120041661689163173[58] = 0;
   out_8120041661689163173[59] = 0;
   out_8120041661689163173[60] = 1.0;
   out_8120041661689163173[61] = 0;
   out_8120041661689163173[62] = 0;
   out_8120041661689163173[63] = 0;
   out_8120041661689163173[64] = 0;
   out_8120041661689163173[65] = 0;
   out_8120041661689163173[66] = 0;
   out_8120041661689163173[67] = 0;
   out_8120041661689163173[68] = 0;
   out_8120041661689163173[69] = 0;
   out_8120041661689163173[70] = 0;
   out_8120041661689163173[71] = 0;
   out_8120041661689163173[72] = 1.0;
   out_8120041661689163173[73] = 0;
   out_8120041661689163173[74] = 0;
   out_8120041661689163173[75] = 0;
   out_8120041661689163173[76] = 0;
   out_8120041661689163173[77] = 0;
   out_8120041661689163173[78] = 0;
   out_8120041661689163173[79] = 0;
   out_8120041661689163173[80] = 0;
   out_8120041661689163173[81] = 0;
   out_8120041661689163173[82] = 0;
   out_8120041661689163173[83] = 0;
   out_8120041661689163173[84] = 1.0;
   out_8120041661689163173[85] = 0;
   out_8120041661689163173[86] = 0;
   out_8120041661689163173[87] = 0;
   out_8120041661689163173[88] = 0;
   out_8120041661689163173[89] = 0;
   out_8120041661689163173[90] = 0;
   out_8120041661689163173[91] = 0;
   out_8120041661689163173[92] = 0;
   out_8120041661689163173[93] = 0;
   out_8120041661689163173[94] = 0;
   out_8120041661689163173[95] = 0;
   out_8120041661689163173[96] = 1.0;
   out_8120041661689163173[97] = 0;
   out_8120041661689163173[98] = 0;
   out_8120041661689163173[99] = 0;
   out_8120041661689163173[100] = 0;
   out_8120041661689163173[101] = 0;
   out_8120041661689163173[102] = 0;
   out_8120041661689163173[103] = 0;
   out_8120041661689163173[104] = 0;
   out_8120041661689163173[105] = 0;
   out_8120041661689163173[106] = 0;
   out_8120041661689163173[107] = 0;
   out_8120041661689163173[108] = 1.0;
   out_8120041661689163173[109] = 0;
   out_8120041661689163173[110] = 0;
   out_8120041661689163173[111] = 0;
   out_8120041661689163173[112] = 0;
   out_8120041661689163173[113] = 0;
   out_8120041661689163173[114] = 0;
   out_8120041661689163173[115] = 0;
   out_8120041661689163173[116] = 0;
   out_8120041661689163173[117] = 0;
   out_8120041661689163173[118] = 0;
   out_8120041661689163173[119] = 0;
   out_8120041661689163173[120] = 1.0;
}
void f_fun(double *state, double dt, double *out_3540396834108163993) {
   out_3540396834108163993[0] = dt*state[3] + state[0];
   out_3540396834108163993[1] = dt*state[4] + state[1];
   out_3540396834108163993[2] = dt*state[5] + state[2];
   out_3540396834108163993[3] = state[3];
   out_3540396834108163993[4] = state[4];
   out_3540396834108163993[5] = state[5];
   out_3540396834108163993[6] = dt*state[7] + state[6];
   out_3540396834108163993[7] = dt*state[8] + state[7];
   out_3540396834108163993[8] = state[8];
   out_3540396834108163993[9] = state[9];
   out_3540396834108163993[10] = state[10];
}
void F_fun(double *state, double dt, double *out_1283585105264921564) {
   out_1283585105264921564[0] = 1;
   out_1283585105264921564[1] = 0;
   out_1283585105264921564[2] = 0;
   out_1283585105264921564[3] = dt;
   out_1283585105264921564[4] = 0;
   out_1283585105264921564[5] = 0;
   out_1283585105264921564[6] = 0;
   out_1283585105264921564[7] = 0;
   out_1283585105264921564[8] = 0;
   out_1283585105264921564[9] = 0;
   out_1283585105264921564[10] = 0;
   out_1283585105264921564[11] = 0;
   out_1283585105264921564[12] = 1;
   out_1283585105264921564[13] = 0;
   out_1283585105264921564[14] = 0;
   out_1283585105264921564[15] = dt;
   out_1283585105264921564[16] = 0;
   out_1283585105264921564[17] = 0;
   out_1283585105264921564[18] = 0;
   out_1283585105264921564[19] = 0;
   out_1283585105264921564[20] = 0;
   out_1283585105264921564[21] = 0;
   out_1283585105264921564[22] = 0;
   out_1283585105264921564[23] = 0;
   out_1283585105264921564[24] = 1;
   out_1283585105264921564[25] = 0;
   out_1283585105264921564[26] = 0;
   out_1283585105264921564[27] = dt;
   out_1283585105264921564[28] = 0;
   out_1283585105264921564[29] = 0;
   out_1283585105264921564[30] = 0;
   out_1283585105264921564[31] = 0;
   out_1283585105264921564[32] = 0;
   out_1283585105264921564[33] = 0;
   out_1283585105264921564[34] = 0;
   out_1283585105264921564[35] = 0;
   out_1283585105264921564[36] = 1;
   out_1283585105264921564[37] = 0;
   out_1283585105264921564[38] = 0;
   out_1283585105264921564[39] = 0;
   out_1283585105264921564[40] = 0;
   out_1283585105264921564[41] = 0;
   out_1283585105264921564[42] = 0;
   out_1283585105264921564[43] = 0;
   out_1283585105264921564[44] = 0;
   out_1283585105264921564[45] = 0;
   out_1283585105264921564[46] = 0;
   out_1283585105264921564[47] = 0;
   out_1283585105264921564[48] = 1;
   out_1283585105264921564[49] = 0;
   out_1283585105264921564[50] = 0;
   out_1283585105264921564[51] = 0;
   out_1283585105264921564[52] = 0;
   out_1283585105264921564[53] = 0;
   out_1283585105264921564[54] = 0;
   out_1283585105264921564[55] = 0;
   out_1283585105264921564[56] = 0;
   out_1283585105264921564[57] = 0;
   out_1283585105264921564[58] = 0;
   out_1283585105264921564[59] = 0;
   out_1283585105264921564[60] = 1;
   out_1283585105264921564[61] = 0;
   out_1283585105264921564[62] = 0;
   out_1283585105264921564[63] = 0;
   out_1283585105264921564[64] = 0;
   out_1283585105264921564[65] = 0;
   out_1283585105264921564[66] = 0;
   out_1283585105264921564[67] = 0;
   out_1283585105264921564[68] = 0;
   out_1283585105264921564[69] = 0;
   out_1283585105264921564[70] = 0;
   out_1283585105264921564[71] = 0;
   out_1283585105264921564[72] = 1;
   out_1283585105264921564[73] = dt;
   out_1283585105264921564[74] = 0;
   out_1283585105264921564[75] = 0;
   out_1283585105264921564[76] = 0;
   out_1283585105264921564[77] = 0;
   out_1283585105264921564[78] = 0;
   out_1283585105264921564[79] = 0;
   out_1283585105264921564[80] = 0;
   out_1283585105264921564[81] = 0;
   out_1283585105264921564[82] = 0;
   out_1283585105264921564[83] = 0;
   out_1283585105264921564[84] = 1;
   out_1283585105264921564[85] = dt;
   out_1283585105264921564[86] = 0;
   out_1283585105264921564[87] = 0;
   out_1283585105264921564[88] = 0;
   out_1283585105264921564[89] = 0;
   out_1283585105264921564[90] = 0;
   out_1283585105264921564[91] = 0;
   out_1283585105264921564[92] = 0;
   out_1283585105264921564[93] = 0;
   out_1283585105264921564[94] = 0;
   out_1283585105264921564[95] = 0;
   out_1283585105264921564[96] = 1;
   out_1283585105264921564[97] = 0;
   out_1283585105264921564[98] = 0;
   out_1283585105264921564[99] = 0;
   out_1283585105264921564[100] = 0;
   out_1283585105264921564[101] = 0;
   out_1283585105264921564[102] = 0;
   out_1283585105264921564[103] = 0;
   out_1283585105264921564[104] = 0;
   out_1283585105264921564[105] = 0;
   out_1283585105264921564[106] = 0;
   out_1283585105264921564[107] = 0;
   out_1283585105264921564[108] = 1;
   out_1283585105264921564[109] = 0;
   out_1283585105264921564[110] = 0;
   out_1283585105264921564[111] = 0;
   out_1283585105264921564[112] = 0;
   out_1283585105264921564[113] = 0;
   out_1283585105264921564[114] = 0;
   out_1283585105264921564[115] = 0;
   out_1283585105264921564[116] = 0;
   out_1283585105264921564[117] = 0;
   out_1283585105264921564[118] = 0;
   out_1283585105264921564[119] = 0;
   out_1283585105264921564[120] = 1;
}
void h_6(double *state, double *sat_pos, double *out_556130728595450510) {
   out_556130728595450510[0] = sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2)) + state[6];
}
void H_6(double *state, double *sat_pos, double *out_8202757098530980823) {
   out_8202757098530980823[0] = (-sat_pos[0] + state[0])/sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2));
   out_8202757098530980823[1] = (-sat_pos[1] + state[1])/sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2));
   out_8202757098530980823[2] = (-sat_pos[2] + state[2])/sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2));
   out_8202757098530980823[3] = 0;
   out_8202757098530980823[4] = 0;
   out_8202757098530980823[5] = 0;
   out_8202757098530980823[6] = 1;
   out_8202757098530980823[7] = 0;
   out_8202757098530980823[8] = 0;
   out_8202757098530980823[9] = 0;
   out_8202757098530980823[10] = 0;
}
void h_20(double *state, double *sat_pos, double *out_1089066492334367938) {
   out_1089066492334367938[0] = sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2)) + sat_pos[3]*state[10] + state[6] + state[9];
}
void H_20(double *state, double *sat_pos, double *out_2432760181549977611) {
   out_2432760181549977611[0] = (-sat_pos[0] + state[0])/sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2));
   out_2432760181549977611[1] = (-sat_pos[1] + state[1])/sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2));
   out_2432760181549977611[2] = (-sat_pos[2] + state[2])/sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2));
   out_2432760181549977611[3] = 0;
   out_2432760181549977611[4] = 0;
   out_2432760181549977611[5] = 0;
   out_2432760181549977611[6] = 1;
   out_2432760181549977611[7] = 0;
   out_2432760181549977611[8] = 0;
   out_2432760181549977611[9] = 1;
   out_2432760181549977611[10] = sat_pos[3];
}
void h_7(double *state, double *sat_pos_vel, double *out_3534781499084937780) {
   out_3534781499084937780[0] = (sat_pos_vel[0] - state[0])*(sat_pos_vel[3] - state[3])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2)) + (sat_pos_vel[1] - state[1])*(sat_pos_vel[4] - state[4])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2)) + (sat_pos_vel[2] - state[2])*(sat_pos_vel[5] - state[5])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2)) + state[7];
}
void H_7(double *state, double *sat_pos_vel, double *out_7925721890503771832) {
   out_7925721890503771832[0] = pow(sat_pos_vel[0] - state[0], 2)*(sat_pos_vel[3] - state[3])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[0] - state[0])*(sat_pos_vel[1] - state[1])*(sat_pos_vel[4] - state[4])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[0] - state[0])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[5] - state[5])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) - (sat_pos_vel[3] - state[3])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_7925721890503771832[1] = (sat_pos_vel[0] - state[0])*(sat_pos_vel[1] - state[1])*(sat_pos_vel[3] - state[3])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + pow(sat_pos_vel[1] - state[1], 2)*(sat_pos_vel[4] - state[4])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[1] - state[1])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[5] - state[5])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) - (sat_pos_vel[4] - state[4])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_7925721890503771832[2] = (sat_pos_vel[0] - state[0])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[3] - state[3])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[1] - state[1])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[4] - state[4])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + pow(sat_pos_vel[2] - state[2], 2)*(sat_pos_vel[5] - state[5])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) - (sat_pos_vel[5] - state[5])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_7925721890503771832[3] = -(sat_pos_vel[0] - state[0])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_7925721890503771832[4] = -(sat_pos_vel[1] - state[1])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_7925721890503771832[5] = -(sat_pos_vel[2] - state[2])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_7925721890503771832[6] = 0;
   out_7925721890503771832[7] = 1;
   out_7925721890503771832[8] = 0;
   out_7925721890503771832[9] = 0;
   out_7925721890503771832[10] = 0;
}
void h_21(double *state, double *sat_pos_vel, double *out_3534781499084937780) {
   out_3534781499084937780[0] = (sat_pos_vel[0] - state[0])*(sat_pos_vel[3] - state[3])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2)) + (sat_pos_vel[1] - state[1])*(sat_pos_vel[4] - state[4])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2)) + (sat_pos_vel[2] - state[2])*(sat_pos_vel[5] - state[5])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2)) + state[7];
}
void H_21(double *state, double *sat_pos_vel, double *out_7925721890503771832) {
   out_7925721890503771832[0] = pow(sat_pos_vel[0] - state[0], 2)*(sat_pos_vel[3] - state[3])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[0] - state[0])*(sat_pos_vel[1] - state[1])*(sat_pos_vel[4] - state[4])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[0] - state[0])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[5] - state[5])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) - (sat_pos_vel[3] - state[3])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_7925721890503771832[1] = (sat_pos_vel[0] - state[0])*(sat_pos_vel[1] - state[1])*(sat_pos_vel[3] - state[3])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + pow(sat_pos_vel[1] - state[1], 2)*(sat_pos_vel[4] - state[4])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[1] - state[1])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[5] - state[5])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) - (sat_pos_vel[4] - state[4])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_7925721890503771832[2] = (sat_pos_vel[0] - state[0])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[3] - state[3])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[1] - state[1])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[4] - state[4])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + pow(sat_pos_vel[2] - state[2], 2)*(sat_pos_vel[5] - state[5])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) - (sat_pos_vel[5] - state[5])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_7925721890503771832[3] = -(sat_pos_vel[0] - state[0])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_7925721890503771832[4] = -(sat_pos_vel[1] - state[1])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_7925721890503771832[5] = -(sat_pos_vel[2] - state[2])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_7925721890503771832[6] = 0;
   out_7925721890503771832[7] = 1;
   out_7925721890503771832[8] = 0;
   out_7925721890503771832[9] = 0;
   out_7925721890503771832[10] = 0;
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

void gnss_update_6(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_6, H_6, NULL, in_z, in_R, in_ea, MAHA_THRESH_6);
}
void gnss_update_20(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_20, H_20, NULL, in_z, in_R, in_ea, MAHA_THRESH_20);
}
void gnss_update_7(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_7, H_7, NULL, in_z, in_R, in_ea, MAHA_THRESH_7);
}
void gnss_update_21(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_21, H_21, NULL, in_z, in_R, in_ea, MAHA_THRESH_21);
}
void gnss_err_fun(double *nom_x, double *delta_x, double *out_1108434360080389513) {
  err_fun(nom_x, delta_x, out_1108434360080389513);
}
void gnss_inv_err_fun(double *nom_x, double *true_x, double *out_6163332558245412373) {
  inv_err_fun(nom_x, true_x, out_6163332558245412373);
}
void gnss_H_mod_fun(double *state, double *out_8120041661689163173) {
  H_mod_fun(state, out_8120041661689163173);
}
void gnss_f_fun(double *state, double dt, double *out_3540396834108163993) {
  f_fun(state,  dt, out_3540396834108163993);
}
void gnss_F_fun(double *state, double dt, double *out_1283585105264921564) {
  F_fun(state,  dt, out_1283585105264921564);
}
void gnss_h_6(double *state, double *sat_pos, double *out_556130728595450510) {
  h_6(state, sat_pos, out_556130728595450510);
}
void gnss_H_6(double *state, double *sat_pos, double *out_8202757098530980823) {
  H_6(state, sat_pos, out_8202757098530980823);
}
void gnss_h_20(double *state, double *sat_pos, double *out_1089066492334367938) {
  h_20(state, sat_pos, out_1089066492334367938);
}
void gnss_H_20(double *state, double *sat_pos, double *out_2432760181549977611) {
  H_20(state, sat_pos, out_2432760181549977611);
}
void gnss_h_7(double *state, double *sat_pos_vel, double *out_3534781499084937780) {
  h_7(state, sat_pos_vel, out_3534781499084937780);
}
void gnss_H_7(double *state, double *sat_pos_vel, double *out_7925721890503771832) {
  H_7(state, sat_pos_vel, out_7925721890503771832);
}
void gnss_h_21(double *state, double *sat_pos_vel, double *out_3534781499084937780) {
  h_21(state, sat_pos_vel, out_3534781499084937780);
}
void gnss_H_21(double *state, double *sat_pos_vel, double *out_7925721890503771832) {
  H_21(state, sat_pos_vel, out_7925721890503771832);
}
void gnss_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
}

const EKF gnss = {
  .name = "gnss",
  .kinds = { 6, 20, 7, 21 },
  .feature_kinds = {  },
  .f_fun = gnss_f_fun,
  .F_fun = gnss_F_fun,
  .err_fun = gnss_err_fun,
  .inv_err_fun = gnss_inv_err_fun,
  .H_mod_fun = gnss_H_mod_fun,
  .predict = gnss_predict,
  .hs = {
    { 6, gnss_h_6 },
    { 20, gnss_h_20 },
    { 7, gnss_h_7 },
    { 21, gnss_h_21 },
  },
  .Hs = {
    { 6, gnss_H_6 },
    { 20, gnss_H_20 },
    { 7, gnss_H_7 },
    { 21, gnss_H_21 },
  },
  .updates = {
    { 6, gnss_update_6 },
    { 20, gnss_update_20 },
    { 7, gnss_update_7 },
    { 21, gnss_update_21 },
  },
  .Hes = {
  },
  .sets = {
  },
  .extra_routines = {
  },
};

ekf_init(gnss);
