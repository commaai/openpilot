#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_7049367403244546975);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_941402359144984825);
void pose_H_mod_fun(double *state, double *out_4100613614008638725);
void pose_f_fun(double *state, double dt, double *out_8397176677496298785);
void pose_F_fun(double *state, double dt, double *out_3499620883405643949);
void pose_h_4(double *state, double *unused, double *out_8172190577682765870);
void pose_H_4(double *state, double *unused, double *out_9193612074661832156);
void pose_h_10(double *state, double *unused, double *out_8347089001260123640);
void pose_H_10(double *state, double *unused, double *out_7557781302877688021);
void pose_h_13(double *state, double *unused, double *out_8271019048962668661);
void pose_H_13(double *state, double *unused, double *out_5981338249329499355);
void pose_h_14(double *state, double *unused, double *out_6097499650426216806);
void pose_H_14(double *state, double *unused, double *out_5230371218322347627);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}