#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_8566306649041746244);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_1696404042837177676);
void pose_H_mod_fun(double *state, double *out_4823007598658938475);
void pose_f_fun(double *state, double dt, double *out_8206133842831466424);
void pose_F_fun(double *state, double dt, double *out_2449703379901721914);
void pose_h_4(double *state, double *unused, double *out_1814135670413092001);
void pose_H_4(double *state, double *unused, double *out_8594814808502846047);
void pose_h_10(double *state, double *unused, double *out_977191010376645903);
void pose_H_10(double *state, double *unused, double *out_4157265640572721743);
void pose_h_13(double *state, double *unused, double *out_51042124798717067);
void pose_H_13(double *state, double *unused, double *out_9159416728184690151);
void pose_h_14(double *state, double *unused, double *out_6024313293648676072);
void pose_H_14(double *state, double *unused, double *out_5512026376207473751);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}