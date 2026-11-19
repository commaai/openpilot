#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_4484765240542640643);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2314530964318079998);
void pose_H_mod_fun(double *state, double *out_3727250654126270259);
void pose_f_fun(double *state, double dt, double *out_2032702142868839994);
void pose_F_fun(double *state, double dt, double *out_7670713034263855145);
void pose_h_4(double *state, double *unused, double *out_702391539073330728);
void pose_H_4(double *state, double *unused, double *out_5217111011931551083);
void pose_h_10(double *state, double *unused, double *out_4841844565248541973);
void pose_H_10(double *state, double *unused, double *out_2598878753773211359);
void pose_h_13(double *state, double *unused, double *out_3389735097065185273);
void pose_H_13(double *state, double *unused, double *out_9050866475234075107);
void pose_h_14(double *state, double *unused, double *out_293591342554501993);
void pose_H_14(double *state, double *unused, double *out_8299899444226923379);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}