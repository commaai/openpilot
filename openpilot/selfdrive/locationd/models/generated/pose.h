#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_3110422523440551038);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2741580585373072420);
void pose_H_mod_fun(double *state, double *out_3055547229087302731);
void pose_f_fun(double *state, double dt, double *out_2068541479278321289);
void pose_F_fun(double *state, double dt, double *out_1433385751187260512);
void pose_h_4(double *state, double *unused, double *out_5960296184483738033);
void pose_H_4(double *state, double *unused, double *out_8903629035839261452);
void pose_h_10(double *state, double *unused, double *out_2371720150447351138);
void pose_H_10(double *state, double *unused, double *out_8795871100057175473);
void pose_h_13(double *state, double *unused, double *out_790049311483924192);
void pose_H_13(double *state, double *unused, double *out_5691355210506928651);
void pose_h_14(double *state, double *unused, double *out_7499051543095232089);
void pose_H_14(double *state, double *unused, double *out_4940388179499776923);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}