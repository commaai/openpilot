#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void live_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_9(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_12(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_35(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_32(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_33(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_H(double *in_vec, double *out_2771383200586010315);
void live_err_fun(double *nom_x, double *delta_x, double *out_7264884532825675944);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_216980306333165683);
void live_H_mod_fun(double *state, double *out_1286384807294278638);
void live_f_fun(double *state, double dt, double *out_197262717200513097);
void live_F_fun(double *state, double dt, double *out_4650870885846882768);
void live_h_4(double *state, double *unused, double *out_7917265329571743646);
void live_H_4(double *state, double *unused, double *out_2651857399384504582);
void live_h_9(double *state, double *unused, double *out_3070906275014437862);
void live_H_9(double *state, double *unused, double *out_4635361535879942888);
void live_h_10(double *state, double *unused, double *out_4313183360394379365);
void live_H_10(double *state, double *unused, double *out_3428032321684379279);
void live_h_12(double *state, double *unused, double *out_767506780943440081);
void live_H_12(double *state, double *unused, double *out_9033115776427237578);
void live_h_35(double *state, double *unused, double *out_6538064231220813596);
void live_H_35(double *state, double *unused, double *out_714804657988102794);
void live_h_32(double *state, double *unused, double *out_5102921810259401742);
void live_H_32(double *state, double *unused, double *out_845969442928843263);
void live_h_13(double *state, double *unused, double *out_7358619630987489117);
void live_H_13(double *state, double *unused, double *out_8328960665934086531);
void live_h_14(double *state, double *unused, double *out_3070906275014437862);
void live_H_14(double *state, double *unused, double *out_4635361535879942888);
void live_h_33(double *state, double *unused, double *out_2359446693134497674);
void live_H_33(double *state, double *unused, double *out_7535353122447734393);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}