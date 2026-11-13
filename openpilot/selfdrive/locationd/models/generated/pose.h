#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_1039242481979313707);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_5271632768120927696);
void pose_H_mod_fun(double *state, double *out_7382780514853116996);
void pose_f_fun(double *state, double dt, double *out_7524264837216844965);
void pose_F_fun(double *state, double dt, double *out_7682706624390115746);
void pose_h_4(double *state, double *unused, double *out_1857154749449920576);
void pose_H_4(double *state, double *unused, double *out_5215881752104475899);
void pose_h_10(double *state, double *unused, double *out_271235286086808204);
void pose_H_10(double *state, double *unused, double *out_2135374553292641286);
void pose_h_13(double *state, double *unused, double *out_9186029837407148995);
void pose_H_13(double *state, double *unused, double *out_8428155577436808700);
void pose_h_14(double *state, double *unused, double *out_2185818957137124734);
void pose_H_14(double *state, double *unused, double *out_9179122608443960428);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}