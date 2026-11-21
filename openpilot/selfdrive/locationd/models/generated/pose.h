#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_5073279989451287369);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_5177693965196878277);
void pose_H_mod_fun(double *state, double *out_2613280836668770145);
void pose_f_fun(double *state, double dt, double *out_3683444243472113040);
void pose_F_fun(double *state, double dt, double *out_3103705100434820586);
void pose_h_4(double *state, double *unused, double *out_5982454180248270517);
void pose_H_4(double *state, double *unused, double *out_5813690737770240169);
void pose_h_10(double *state, double *unused, double *out_8845439129170142173);
void pose_H_10(double *state, double *unused, double *out_5995085404368594946);
void pose_h_13(double *state, double *unused, double *out_5280177107270464524);
void pose_H_13(double *state, double *unused, double *out_1796940470546460760);
void pose_h_14(double *state, double *unused, double *out_1941905627500734236);
void pose_H_14(double *state, double *unused, double *out_8896479170065612465);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}