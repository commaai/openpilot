#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_8287156320329551527);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_152026319407410727);
void pose_H_mod_fun(double *state, double *out_3515917231783269738);
void pose_f_fun(double *state, double dt, double *out_8327036891744297669);
void pose_F_fun(double *state, double dt, double *out_6376990126270885036);
void pose_h_4(double *state, double *unused, double *out_1784339471610928199);
void pose_H_4(double *state, double *unused, double *out_4713864713666167842);
void pose_h_10(double *state, double *unused, double *out_7218205989387104649);
void pose_H_10(double *state, double *unused, double *out_664999596844092490);
void pose_h_13(double *state, double *unused, double *out_4772014765743314712);
void pose_H_13(double *state, double *unused, double *out_880109250363643818);
void pose_h_14(double *state, double *unused, double *out_7459610224657432703);
void pose_H_14(double *state, double *unused, double *out_1631076281370795546);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}