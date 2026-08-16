#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_1364756071930907291);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_8303365725516605546);
void pose_H_mod_fun(double *state, double *out_3322466727120633125);
void pose_f_fun(double *state, double dt, double *out_4385019422503781050);
void pose_F_fun(double *state, double dt, double *out_3364185213448184257);
void pose_h_4(double *state, double *unused, double *out_277067005022287991);
void pose_H_4(double *state, double *unused, double *out_6474383267850384560);
void pose_h_10(double *state, double *unused, double *out_439254518827768136);
void pose_H_10(double *state, double *unused, double *out_4997374602163059806);
void pose_h_13(double *state, double *unused, double *out_5459722292722030835);
void pose_H_13(double *state, double *unused, double *out_3262109442518051759);
void pose_h_14(double *state, double *unused, double *out_3151095188830224548);
void pose_H_14(double *state, double *unused, double *out_6909499794495268159);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}