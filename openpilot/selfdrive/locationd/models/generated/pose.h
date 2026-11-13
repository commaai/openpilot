#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_695385451800373050);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_3513419238100027883);
void pose_H_mod_fun(double *state, double *out_6118050737020864777);
void pose_f_fun(double *state, double dt, double *out_4351988193126604514);
void pose_F_fun(double *state, double dt, double *out_2969389070201032483);
void pose_h_4(double *state, double *unused, double *out_1854639727209469544);
void pose_H_4(double *state, double *unused, double *out_269968930268906056);
void pose_h_10(double *state, double *unused, double *out_2214826177905303244);
void pose_H_10(double *state, double *unused, double *out_8815669168610743905);
void pose_h_13(double *state, double *unused, double *out_7189848073171455906);
void pose_H_13(double *state, double *unused, double *out_3482242755601238857);
void pose_h_14(double *state, double *unused, double *out_2380654778222354163);
void pose_H_14(double *state, double *unused, double *out_4233209786608390585);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}