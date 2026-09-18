#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_5087435637205728933);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2851508015392009149);
void pose_H_mod_fun(double *state, double *out_3140738153404446410);
void pose_f_fun(double *state, double dt, double *out_8261207861188225982);
void pose_F_fun(double *state, double dt, double *out_2195864972581525307);
void pose_h_4(double *state, double *unused, double *out_7615763258081257669);
void pose_H_4(double *state, double *unused, double *out_4338685635287344514);
void pose_h_10(double *state, double *unused, double *out_4081004568469840555);
void pose_H_10(double *state, double *unused, double *out_2118589289993173556);
void pose_h_13(double *state, double *unused, double *out_7322787283816672919);
void pose_H_13(double *state, double *unused, double *out_504930171984820490);
void pose_h_14(double *state, double *unused, double *out_2194728518549899811);
void pose_H_14(double *state, double *unused, double *out_1255897202991972218);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}