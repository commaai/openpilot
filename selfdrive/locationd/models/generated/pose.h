#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_1100575581318041579);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_6410724245809315096);
void pose_H_mod_fun(double *state, double *out_1238262041853718804);
void pose_f_fun(double *state, double dt, double *out_470659832336319089);
void pose_F_fun(double *state, double dt, double *out_7507724760190486926);
void pose_h_4(double *state, double *unused, double *out_8065105448835972349);
void pose_H_4(double *state, double *unused, double *out_1962147859247751220);
void pose_h_10(double *state, double *unused, double *out_365029382432652932);
void pose_H_10(double *state, double *unused, double *out_6475974887783404571);
void pose_h_13(double *state, double *unused, double *out_1501704762674320483);
void pose_H_13(double *state, double *unused, double *out_1397545939565907116);
void pose_h_14(double *state, double *unused, double *out_2046062482872930650);
void pose_H_14(double *state, double *unused, double *out_5044936291543123516);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}