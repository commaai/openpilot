#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_4069511595981512906);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_6977798052227057242);
void pose_H_mod_fun(double *state, double *out_2633911611712179105);
void pose_f_fun(double *state, double dt, double *out_7060402792772463216);
void pose_F_fun(double *state, double dt, double *out_4609062820950943518);
void pose_h_4(double *state, double *unused, double *out_1304819822248380504);
void pose_H_4(double *state, double *unused, double *out_8481993418464137826);
void pose_h_10(double *state, double *unused, double *out_3476711082558028639);
void pose_H_10(double *state, double *unused, double *out_436246387152335361);
void pose_h_13(double *state, double *unused, double *out_5988377314978994220);
void pose_H_13(double *state, double *unused, double *out_5269719593131805025);
void pose_h_14(double *state, double *unused, double *out_4290951141142713711);
void pose_H_14(double *state, double *unused, double *out_4518752562124653297);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}