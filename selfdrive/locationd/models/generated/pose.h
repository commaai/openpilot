#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_3495798777798121399);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_6491638533999781681);
void pose_H_mod_fun(double *state, double *out_1193861252693843199);
void pose_f_fun(double *state, double dt, double *out_2162222754272876291);
void pose_F_fun(double *state, double dt, double *out_577996268896955016);
void pose_h_4(double *state, double *unused, double *out_2480001917824807916);
void pose_H_4(double *state, double *unused, double *out_4654220554058115522);
void pose_h_10(double *state, double *unused, double *out_6842199753574438663);
void pose_H_10(double *state, double *unused, double *out_8460558184635204487);
void pose_h_13(double *state, double *unused, double *out_7103896844424350049);
void pose_H_13(double *state, double *unused, double *out_1441946728725782721);
void pose_h_14(double *state, double *unused, double *out_2985573120985558604);
void pose_H_14(double *state, double *unused, double *out_690979697718630993);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}