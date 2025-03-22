#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_3867314615092444191);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_357340291324325113);
void pose_H_mod_fun(double *state, double *out_5335377228813010670);
void pose_f_fun(double *state, double dt, double *out_8449822447276776350);
void pose_F_fun(double *state, double dt, double *out_8918868180555199161);
void pose_h_4(double *state, double *unused, double *out_6617669203308538531);
void pose_H_4(double *state, double *unused, double *out_1875678236267692381);
void pose_h_10(double *state, double *unused, double *out_8387436857003783836);
void pose_H_10(double *state, double *unused, double *out_253999754152164366);
void pose_h_13(double *state, double *unused, double *out_3782328115152295086);
void pose_H_13(double *state, double *unused, double *out_5709433699570216405);
void pose_h_14(double *state, double *unused, double *out_8292081498849349712);
void pose_H_14(double *state, double *unused, double *out_4958466668563064677);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}