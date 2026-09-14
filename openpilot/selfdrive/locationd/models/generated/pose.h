#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_498200963581026915);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_191777889625927103);
void pose_H_mod_fun(double *state, double *out_8577665774918696216);
void pose_f_fun(double *state, double dt, double *out_5599332296746008014);
void pose_F_fun(double *state, double dt, double *out_1584783165911621436);
void pose_h_4(double *state, double *unused, double *out_8170273359625996454);
void pose_H_4(double *state, double *unused, double *out_7481347933629845101);
void pose_h_10(double *state, double *unused, double *out_4791397978297849235);
void pose_H_10(double *state, double *unused, double *out_7583350988764981753);
void pose_h_13(double *state, double *unused, double *out_5529921368018353304);
void pose_H_13(double *state, double *unused, double *out_3647592470327321077);
void pose_h_14(double *state, double *unused, double *out_4744126367926772826);
void pose_H_14(double *state, double *unused, double *out_4398559501334472805);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}