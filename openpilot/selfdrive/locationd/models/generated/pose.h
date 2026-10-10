#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_5727667037901716489);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_6198627751216921880);
void pose_H_mod_fun(double *state, double *out_8253589554410785672);
void pose_f_fun(double *state, double dt, double *out_6523760126628443948);
void pose_F_fun(double *state, double dt, double *out_2397429393732395161);
void pose_h_4(double *state, double *unused, double *out_8382275560088012236);
void pose_H_4(double *state, double *unused, double *out_6433890828504488811);
void pose_h_10(double *state, double *unused, double *out_3474458810352115326);
void pose_H_10(double *state, double *unused, double *out_3190334320092666102);
void pose_h_13(double *state, double *unused, double *out_1070052259346719453);
void pose_H_13(double *state, double *unused, double *out_8800579419872730004);
void pose_h_14(double *state, double *unused, double *out_3428767572042643384);
void pose_H_14(double *state, double *unused, double *out_8049612388865578276);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}