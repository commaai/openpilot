#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_409617722116612880);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2261331234596504125);
void pose_H_mod_fun(double *state, double *out_8946862073519308870);
void pose_f_fun(double *state, double dt, double *out_5554883369010406028);
void pose_F_fun(double *state, double dt, double *out_6041026006058863705);
void pose_h_4(double *state, double *unused, double *out_2384878000485610448);
void pose_H_4(double *state, double *unused, double *out_761685768107253608);
void pose_h_10(double *state, double *unused, double *out_1692252871197538134);
void pose_H_10(double *state, double *unused, double *out_4918497349089969484);
void pose_h_13(double *state, double *unused, double *out_3349492577274348733);
void pose_H_13(double *state, double *unused, double *out_2450588057225079193);
void pose_h_14(double *state, double *unused, double *out_5910242476345995022);
void pose_H_14(double *state, double *unused, double *out_8242831583386994032);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}