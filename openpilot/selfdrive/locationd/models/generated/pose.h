#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_348428071618360721);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2713115070426494842);
void pose_H_mod_fun(double *state, double *out_1740654701803477855);
void pose_f_fun(double *state, double dt, double *out_7474451683258418033);
void pose_F_fun(double *state, double dt, double *out_953892830646151789);
void pose_h_4(double *state, double *unused, double *out_44459550781696338);
void pose_H_4(double *state, double *unused, double *out_79044024102819006);
void pose_h_10(double *state, double *unused, double *out_81618196104347189);
void pose_H_10(double *state, double *unused, double *out_4021134994901400003);
void pose_h_13(double *state, double *unused, double *out_3818565266702701970);
void pose_H_13(double *state, double *unused, double *out_3133229801229513795);
void pose_h_14(double *state, double *unused, double *out_1807207524562730417);
void pose_H_14(double *state, double *unused, double *out_3161832456398191302);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}