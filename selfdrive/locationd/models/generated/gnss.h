#pragma once
#include "rednose/helpers/common_ekf.h"
extern "C" {
void gnss_update_6(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void gnss_update_20(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void gnss_update_7(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void gnss_update_21(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void gnss_err_fun(double *nom_x, double *delta_x, double *out_1108434360080389513);
void gnss_inv_err_fun(double *nom_x, double *true_x, double *out_6163332558245412373);
void gnss_H_mod_fun(double *state, double *out_8120041661689163173);
void gnss_f_fun(double *state, double dt, double *out_3540396834108163993);
void gnss_F_fun(double *state, double dt, double *out_1283585105264921564);
void gnss_h_6(double *state, double *sat_pos, double *out_556130728595450510);
void gnss_H_6(double *state, double *sat_pos, double *out_8202757098530980823);
void gnss_h_20(double *state, double *sat_pos, double *out_1089066492334367938);
void gnss_H_20(double *state, double *sat_pos, double *out_2432760181549977611);
void gnss_h_7(double *state, double *sat_pos_vel, double *out_3534781499084937780);
void gnss_H_7(double *state, double *sat_pos_vel, double *out_7925721890503771832);
void gnss_h_21(double *state, double *sat_pos_vel, double *out_3534781499084937780);
void gnss_H_21(double *state, double *sat_pos_vel, double *out_7925721890503771832);
void gnss_predict(double *in_x, double *in_P, double *in_Q, double dt);
}