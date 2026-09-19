#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_7210081177362107870);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_8354364799378035034);
void pose_H_mod_fun(double *state, double *out_1145354702242097965);
void pose_f_fun(double *state, double dt, double *out_7691519941823953160);
void pose_F_fun(double *state, double dt, double *out_1204100135177654472);
void pose_h_4(double *state, double *unused, double *out_7251927051155261244);
void pose_H_4(double *state, double *unused, double *out_7799006963815723377);
void pose_h_10(double *state, double *unused, double *out_3161621592790876862);
void pose_H_10(double *state, double *unused, double *out_8738726503938101473);
void pose_h_13(double *state, double *unused, double *out_5099159631779018063);
void pose_H_13(double *state, double *unused, double *out_4586733138483390576);
void pose_h_14(double *state, double *unused, double *out_204192348592699604);
void pose_H_14(double *state, double *unused, double *out_3835766107476238848);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}