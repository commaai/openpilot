#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_7351456117776073154);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_9109558368636035839);
void pose_H_mod_fun(double *state, double *out_595012964091947662);
void pose_f_fun(double *state, double dt, double *out_3792261713431509332);
void pose_F_fun(double *state, double dt, double *out_6928823179361542013);
void pose_h_4(double *state, double *unused, double *out_7683085519451864677);
void pose_H_4(double *state, double *unused, double *out_3149865311834297615);
void pose_h_10(double *state, double *unused, double *out_4760549729285226708);
void pose_H_10(double *state, double *unused, double *out_748214929053840095);
void pose_h_13(double *state, double *unused, double *out_6328698996174999134);
void pose_H_13(double *state, double *unused, double *out_6362139137166630416);
void pose_h_14(double *state, double *unused, double *out_4490584305148798359);
void pose_H_14(double *state, double *unused, double *out_7113106168173782144);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}