#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_4336476086057343998);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_1258106400637610027);
void pose_H_mod_fun(double *state, double *out_6527990562688054560);
void pose_f_fun(double *state, double dt, double *out_8412414231100200223);
void pose_F_fun(double *state, double dt, double *out_5891562599511662399);
void pose_h_4(double *state, double *unused, double *out_5872972277401777544);
void pose_H_4(double *state, double *unused, double *out_2756183352844146988);
void pose_h_10(double *state, double *unused, double *out_2437929224325462368);
void pose_H_10(double *state, double *unused, double *out_2308350212782797437);
void pose_h_13(double *state, double *unused, double *out_5303446754481693594);
void pose_H_13(double *state, double *unused, double *out_4854447855472553941);
void pose_h_14(double *state, double *unused, double *out_7614535523248366498);
void pose_H_14(double *state, double *unused, double *out_1207057503495337541);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}