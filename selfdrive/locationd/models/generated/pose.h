#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_7464579881794296048);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2725303949871892042);
void pose_H_mod_fun(double *state, double *out_2728954879743535149);
void pose_f_fun(double *state, double dt, double *out_4998926688300100454);
void pose_F_fun(double *state, double dt, double *out_8226313807756837064);
void pose_h_4(double *state, double *unused, double *out_3272625519318058906);
void pose_H_4(double *state, double *unused, double *out_8577036686495493870);
void pose_h_10(double *state, double *unused, double *out_7478176691014094533);
void pose_H_10(double *state, double *unused, double *out_3894289231047876903);
void pose_h_13(double *state, double *unused, double *out_257622233170053408);
void pose_H_13(double *state, double *unused, double *out_5364762861163161069);
void pose_h_14(double *state, double *unused, double *out_2543715519988584578);
void pose_H_14(double *state, double *unused, double *out_4613795830156009341);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}