#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_716632891583450328);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_3104130222774257522);
void pose_H_mod_fun(double *state, double *out_8665327190051684991);
void pose_f_fun(double *state, double dt, double *out_1781769764659626549);
void pose_F_fun(double *state, double dt, double *out_3937534521909816001);
void pose_h_4(double *state, double *unused, double *out_4373223397940532151);
void pose_H_4(double *state, double *unused, double *out_7961718157751569764);
void pose_h_10(double *state, double *unused, double *out_6286469033191029799);
void pose_H_10(double *state, double *unused, double *out_5531647416424197146);
void pose_h_13(double *state, double *unused, double *out_5115075594500414792);
void pose_H_13(double *state, double *unused, double *out_7272752090625649051);
void pose_h_14(double *state, double *unused, double *out_7724198509017227969);
void pose_H_14(double *state, double *unused, double *out_6521785059618497323);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}