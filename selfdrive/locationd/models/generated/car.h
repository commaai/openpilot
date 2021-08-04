#pragma once
#include "rednose/helpers/common_ekf.h"
extern "C" {
void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_err_fun(double *nom_x, double *delta_x, double *out_161625240143566849);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_6140887166081679633);
void car_H_mod_fun(double *state, double *out_7194905136312760084);
void car_f_fun(double *state, double dt, double *out_7617539522698236049);
void car_F_fun(double *state, double dt, double *out_8484164093019597553);
void car_h_25(double *state, double *unused, double *out_6067892585336549744);
void car_H_25(double *state, double *unused, double *out_4184736662998209975);
void car_h_24(double *state, double *unused, double *out_9057033175405661551);
void car_H_24(double *state, double *unused, double *out_8543405586499812728);
void car_h_30(double *state, double *unused, double *out_519935341139745419);
void car_H_30(double *state, double *unused, double *out_5429162247950973305);
void car_h_26(double *state, double *unused, double *out_4481737926521100226);
void car_H_26(double *state, double *unused, double *out_2843300529302562423);
void car_h_27(double *state, double *unused, double *out_7036984252486971309);
void car_H_27(double *state, double *unused, double *out_4683970549287096174);
void car_h_29(double *state, double *unused, double *out_7312178314771477198);
void car_H_29(double *state, double *unused, double *out_4854603634095437130);
void car_h_28(double *state, double *unused, double *out_895997352713156116);
void car_H_28(double *state, double *unused, double *out_2885506008973244022);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}