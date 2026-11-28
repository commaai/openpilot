#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_8093069977717794784);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2890339221041424080);
void pose_H_mod_fun(double *state, double *out_3258004078251541594);
void pose_f_fun(double *state, double dt, double *out_3851408165110359671);
void pose_F_fun(double *state, double dt, double *out_5176413023006983401);
void pose_h_4(double *state, double *unused, double *out_9158595119508026012);
void pose_H_4(double *state, double *unused, double *out_7087198156485597472);
void pose_h_10(double *state, double *unused, double *out_1976892447920578365);
void pose_H_10(double *state, double *unused, double *out_2856556414602702172);
void pose_h_13(double *state, double *unused, double *out_7124925249639989703);
void pose_H_13(double *state, double *unused, double *out_8147272091891621343);
void pose_h_14(double *state, double *unused, double *out_1921870948443217106);
void pose_H_14(double *state, double *unused, double *out_7396305060884469615);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}