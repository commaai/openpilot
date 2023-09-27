#pragma once
#include "rednose/helpers/common_ekf.h"
extern "C" {
void live_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_9(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_12(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_32(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_33(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_H(double *in_vec, double *out_453938190038079243);
void live_err_fun(double *nom_x, double *delta_x, double *out_2158047252761075543);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_8734297616248117664);
void live_H_mod_fun(double *state, double *out_2860066609604479351);
void live_f_fun(double *state, double dt, double *out_5602464115310423705);
void live_F_fun(double *state, double dt, double *out_6916573737423352276);
void live_h_4(double *state, double *unused, double *out_596332248622237516);
void live_H_4(double *state, double *unused, double *out_4979886627124493108);
void live_h_9(double *state, double *unused, double *out_3856253699158019501);
void live_H_9(double *state, double *unused, double *out_5221076273754083753);
void live_h_10(double *state, double *unused, double *out_8315774334350174408);
void live_H_10(double *state, double *unused, double *out_3413394562179663325);
void live_h_12(double *state, double *unused, double *out_7313259081701872739);
void live_H_12(double *state, double *unused, double *out_8447401038553096713);
void live_h_31(double *state, double *unused, double *out_140212771163044236);
void live_H_31(double *state, double *unused, double *out_8346548684497100484);
void live_h_32(double *state, double *unused, double *out_7302549768249459087);
void live_H_32(double *state, double *unused, double *out_4005543908763993596);
void live_h_13(double *state, double *unused, double *out_6683782575543378369);
void live_H_13(double *state, double *unused, double *out_5186093271929199825);
void live_h_14(double *state, double *unused, double *out_3856253699158019501);
void live_H_14(double *state, double *unused, double *out_5221076273754083753);
void live_h_33(double *state, double *unused, double *out_5701278751367177618);
void live_H_33(double *state, double *unused, double *out_6949638384573593528);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}