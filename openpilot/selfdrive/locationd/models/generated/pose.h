#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_3812274923922881427);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_3419763620583365922);
void pose_H_mod_fun(double *state, double *out_6689702553421276651);
void pose_f_fun(double *state, double dt, double *out_446039340896779864);
void pose_F_fun(double *state, double dt, double *out_642758158368098798);
void pose_h_4(double *state, double *unused, double *out_7695501649787619215);
void pose_H_4(double *state, double *unused, double *out_2075075516491960287);
void pose_h_10(double *state, double *unused, double *out_289310091903726368);
void pose_H_10(double *state, double *unused, double *out_2508005545533897472);
void pose_h_13(double *state, double *unused, double *out_2751864404123375741);
void pose_H_13(double *state, double *unused, double *out_1137198308840372514);
void pose_h_14(double *state, double *unused, double *out_2963815555134069202);
void pose_H_14(double *state, double *unused, double *out_1888165339847524242);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}