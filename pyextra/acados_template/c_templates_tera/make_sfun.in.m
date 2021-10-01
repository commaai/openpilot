%
% Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
% Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
% Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
% Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
%
% This file is part of acados.
%
% The 2-Clause BSD License
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.;
%

SOURCES = { ...
        {%- if solver_options.integrator_type == 'ERK' %}
            '{{ model.name }}_model/{{ model.name }}_expl_ode_fun.c', ...
            '{{ model.name }}_model/{{ model.name }}_expl_vde_forw.c',...
            {%- if solver_options.hessian_approx == 'EXACT' %}
            '{{ model.name }}_model/{{ model.name }}_expl_ode_hess.c',...
            {%- endif %}
        {%- elif solver_options.integrator_type == "IRK" %}
            '{{ model.name }}_model/{{ model.name }}_impl_dae_fun.c', ...
            '{{ model.name }}_model/{{ model.name }}_impl_dae_fun_jac_x_xdot_z.c', ...
            '{{ model.name }}_model/{{ model.name }}_impl_dae_jac_x_xdot_u_z.c', ...
            {%- if solver_options.hessian_approx == 'EXACT' %}
            '{{ model.name }}_model/{{ model.name }}_impl_dae_hess.c',...
            {%- endif %}
        {%- elif solver_options.integrator_type == "GNSF" %}
            '{{ model.name }}_model/{{ model.name }}_gnsf_phi_fun.c',...
            '{{ model.name }}_model/{{ model.name }}_gnsf_phi_fun_jac_y.c',...
            '{{ model.name }}_model/{{ model.name }}_gnsf_phi_jac_y_uhat.c',...
            '{{ model.name }}_model/{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz.c',...
            '{{ model.name }}_model/{{ model.name }}_gnsf_get_matrices_fun.c',...
        {%- elif solver_options.integrator_type == "DISCRETE" %}
            '{{ model.name }}_model/{{ model.name }}_dyn_disc_phi_fun.c',...
            '{{ model.name }}_model/{{ model.name }}_dyn_disc_phi_fun_jac.c',...
        {%- if solver_options.hessian_approx == "EXACT" %}
            '{{ model.name }}_model/{{ model.name }}_dyn_disc_phi_fun_jac_hess.c',...
        {%- endif %}
        {%- endif %}
        {%- if cost.cost_type_0 == "NONLINEAR_LS" %}
            '{{ model.name }}_cost/{{ model.name }}_cost_y_0_fun.c',...
            '{{ model.name }}_cost/{{ model.name }}_cost_y_0_fun_jac_ut_xt.c',...
            '{{ model.name }}_cost/{{ model.name }}_cost_y_0_hess.c',...
        {%- elif cost.cost_type_0 == "EXTERNAL" %}
            '{{ model.name }}_cost/{{ model.name }}_cost_ext_cost_0_fun.c',...
            '{{ model.name }}_cost/{{ model.name }}_cost_ext_cost_0_fun_jac.c',...
            '{{ model.name }}_cost/{{ model.name }}_cost_ext_cost_0_fun_jac_hess.c',...
        {%- endif %}

        {%- if cost.cost_type == "NONLINEAR_LS" %}
            '{{ model.name }}_cost/{{ model.name }}_cost_y_fun.c',...
            '{{ model.name }}_cost/{{ model.name }}_cost_y_fun_jac_ut_xt.c',...
            '{{ model.name }}_cost/{{ model.name }}_cost_y_hess.c',...
        {%- elif cost.cost_type == "EXTERNAL" %}
            '{{ model.name }}_cost/{{ model.name }}_cost_ext_cost_fun.c',...
            '{{ model.name }}_cost/{{ model.name }}_cost_ext_cost_fun_jac.c',...
            '{{ model.name }}_cost/{{ model.name }}_cost_ext_cost_fun_jac_hess.c',...
        {%- endif %}
        {%- if cost.cost_type_e == "NONLINEAR_LS" %}
            '{{ model.name }}_cost/{{ model.name }}_cost_y_e_fun.c',...
            '{{ model.name }}_cost/{{ model.name }}_cost_y_e_fun_jac_ut_xt.c',...
            '{{ model.name }}_cost/{{ model.name }}_cost_y_e_hess.c',...
        {%- elif cost.cost_type_e == "EXTERNAL" %}
            '{{ model.name }}_cost/{{ model.name }}_cost_ext_cost_e_fun.c',...
            '{{ model.name }}_cost/{{ model.name }}_cost_ext_cost_e_fun_jac.c',...
            '{{ model.name }}_cost/{{ model.name }}_cost_ext_cost_e_fun_jac_hess.c',...
        {%- endif %}
        {%- if constraints.constr_type == "BGH"  and dims.nh > 0 %}
            '{{ model.name }}_constraints/{{ model.name }}_constr_h_fun.c', ...
            '{{ model.name }}_constraints/{{ model.name }}_constr_h_fun_jac_uxt_zt_hess.c', ...
            '{{ model.name }}_constraints/{{ model.name }}_constr_h_fun_jac_uxt_zt.c', ...
        {%- elif constraints.constr_type == "BGP" and dims.nphi > 0 %}
            '{{ model.name }}_constraints/{{ model.name }}_phi_constraint.c', ...
        {%- endif %}
        {%- if constraints.constr_type_e == "BGH"  and dims.nh_e > 0 %}
            '{{ model.name }}_constraints/{{ model.name }}_constr_h_e_fun.c', ...
            '{{ model.name }}_constraints/{{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess.c', ...
            '{{ model.name }}_constraints/{{ model.name }}_constr_h_e_fun_jac_uxt_zt.c', ...
        {%- elif constraints.constr_type_e == "BGP" and dims.nphi_e > 0 %}
            '{{ model.name }}_constraints/{{ model.name }}_phi_e_constraint.c', ...
        {%- endif %}
            'acados_solver_sfunction_{{ model.name }}.c', ...
            'acados_solver_{{ model.name }}.c'
          };

INC_PATH = '{{ acados_include_path }}';

INCS = {['-I', fullfile(INC_PATH, 'blasfeo', 'include')], ...
        ['-I', fullfile(INC_PATH, 'hpipm', 'include')], ...
        ['-I', fullfile(INC_PATH, 'acados')], ...
        ['-I', fullfile(INC_PATH)]};

{% if solver_options.qp_solver is containing("QPOASES") %}
INCS{end+1} = ['-I', fullfile(INC_PATH, 'qpOASES_e')];
{% endif %}

CFLAGS = 'CFLAGS=$CFLAGS';
LDFLAGS = 'LDFLAGS=$LDFLAGS';
COMPFLAGS = 'COMPFLAGS=$COMPFLAGS';
COMPDEFINES = 'COMPDEFINES=$COMPDEFINES';

{% if solver_options.qp_solver is containing("QPOASES") %}
CFLAGS = [ CFLAGS, ' -DACADOS_WITH_QPOASES ' ];
COMPDEFINES = [ COMPDEFINES, ' -DACADOS_WITH_QPOASES ' ];
{%- elif solver_options.qp_solver is containing("OSQP") %}
CFLAGS = [ CFLAGS, ' -DACADOS_WITH_OSQP ' ];
COMPDEFINES = [ COMPDEFINES, ' -DACADOS_WITH_OSQP ' ];
{%- elif solver_options.qp_solver is containing("QPDUNES") %}
CFLAGS = [ CFLAGS, ' -DACADOS_WITH_QPDUNES ' ];
COMPDEFINES = [ COMPDEFINES, ' -DACADOS_WITH_QPDUNES ' ];
{%- elif solver_options.qp_solver is containing("HPMPC") %}
CFLAGS = [ CFLAGS, ' -DACADOS_WITH_HPMPC ' ];
COMPDEFINES = [ COMPDEFINES, ' -DACADOS_WITH_HPMPC ' ];
{% endif %}

LIB_PATH = ['-L', fullfile('{{ acados_lib_path }}')];

LIBS = {'-lacados', '-lhpipm', '-lblasfeo'};

% acados linking libraries and flags
{%- if acados_link_libs and os and os == "pc" %}
LDFLAGS = [LDFLAGS ' {{ acados_link_libs.openmp }}'];
COMPFLAGS = [COMPFLAGS ' {{ acados_link_libs.openmp }}'];
LIBS{end+1} = '{{ acados_link_libs.qpoases }}';
LIBS{end+1} = '{{ acados_link_libs.hpmpc }}';
LIBS{end+1} = '{{ acados_link_libs.osqp }}';
{%- else %}
    {% if solver_options.qp_solver is containing("QPOASES") %}
LIBS{end+1} = '-lqpOASES_e';
    {% endif %}
{%- endif %}

mex('-v', '-O', CFLAGS, LDFLAGS, COMPFLAGS, COMPDEFINES, INCS{:}, ...
    LIB_PATH, LIBS{:}, SOURCES{:}, ...
    '-output', 'acados_solver_sfunction_{{ model.name }}' );

fprintf( [ '\n\nSuccessfully created sfunction:\nacados_solver_sfunction_{{ model.name }}', '.', ...
    eval('mexext')] );


%% print note on usage of s-function
fprintf('\n\nNote: Usage of Sfunction is as follows:\n')
input_note = 'Inputs are:\n';
i_in = 1;


{%- if dims.nbx_0 > 0 and simulink_opts.inputs.lbx_0 -%}  {#- lbx_0 #}
input_note = strcat(input_note, num2str(i_in), ') lbx_0 - lower bound on x for stage 0,',...
                    ' size [{{ dims.nbx_0 }}]\n ');
i_in = i_in + 1;
{%- endif %}

{%- if dims.nbx_0 > 0 and simulink_opts.inputs.ubx_0 -%}  {#- ubx_0 #}
input_note = strcat(input_note, num2str(i_in), ') ubx_0 - upper bound on x for stage 0,',...
                    ' size [{{ dims.nbx_0 }}]\n ');
i_in = i_in + 1;
{%- endif %}

{%- if dims.np > 0 and simulink_opts.inputs.parameter_traj -%}  {#- parameter_traj #}
input_note = strcat(input_note, num2str(i_in), ') parameters - concatenated for all shooting nodes 0 to N+1,',...
                    ' size [{{ (dims.N+1)*dims.np }}]\n ');
i_in = i_in + 1;
{%- endif %}

{%- if dims.ny_0 > 0 and simulink_opts.inputs.y_ref_0 %}
input_note = strcat(input_note, num2str(i_in), ') y_ref_0, size [{{ dims.ny_0 }}]\n ');
i_in = i_in + 1;
{%- endif %}

{%- if dims.ny > 0 and dims.N > 1 and simulink_opts.inputs.y_ref %}
input_note = strcat(input_note, num2str(i_in), ') y_ref - concatenated for shooting nodes 1 to N-1,',...
                    ' size [{{ (dims.N-1) * dims.ny }}]\n ');
i_in = i_in + 1;
{%- endif %}

{%- if dims.ny_e > 0 and dims.N > 0 and simulink_opts.inputs.y_ref_e %}
input_note = strcat(input_note, num2str(i_in), ') y_ref_e, size [{{ dims.ny_e }}]\n ');
i_in = i_in + 1;
{%- endif %}

{%- if dims.nbx > 0 and dims.N > 1 and simulink_opts.inputs.lbx -%}  {#- lbx #}
input_note = strcat(input_note, num2str(i_in), ') lbx for shooting nodes 1 to N-1, size [{{ (dims.N-1) * dims.nbx }}]\n ');
i_in = i_in + 1;
{%- endif %}
{%- if dims.nbx > 0 and dims.N > 1 and simulink_opts.inputs.ubx -%}  {#- ubx #}
input_note = strcat(input_note, num2str(i_in), ') ubx for shooting nodes 1 to N-1, size [{{ (dims.N-1) * dims.nbx }}]\n ');
i_in = i_in + 1;
{%- endif %}


{%- if dims.nbx_e > 0 and dims.N > 0 and simulink_opts.inputs.lbx_e -%}  {#- lbx_e #}
input_note = strcat(input_note, num2str(i_in), ') lbx_e (lbx at shooting node N), size [{{ dims.nbx_e }}]\n ');
i_in = i_in + 1;
{%- endif %}
{%- if dims.nbx_e > 0 and dims.N > 0 and simulink_opts.inputs.ubx_e -%}  {#- ubx_e #}
input_note = strcat(input_note, num2str(i_in), ') ubx_e (ubx at shooting node N), size [{{ dims.nbx_e }}]\n ');
i_in = i_in + 1;
{%- endif %}

{%- if dims.nbu > 0 and dims.N > 0 and simulink_opts.inputs.lbu -%}  {#- lbu #}
input_note = strcat(input_note, num2str(i_in), ') lbu for shooting nodes 0 to N-1, size [{{ dims.N*dims.nbu }}]\n ');
i_in = i_in + 1;
{%- endif -%}
{%- if dims.nbu > 0 and dims.N > 0 and simulink_opts.inputs.ubu -%}  {#- ubu #}
input_note = strcat(input_note, num2str(i_in), ') ubu for shooting nodes 0 to N-1, size [{{ dims.N*dims.nbu }}]\n ');
i_in = i_in + 1;
{%- endif -%}

{%- if dims.ng > 0 and simulink_opts.inputs.lg -%}  {#- lg #}
input_note = strcat(input_note, num2str(i_in), ') lg, size [{{ dims.ng }}]\n ');
i_in = i_in + 1;
{%- endif %}
{%- if dims.ng > 0 and simulink_opts.inputs.ug -%}  {#- ug #}
input_note = strcat(input_note, num2str(i_in), ') ug, size [{{ dims.ng }}]\n ');
i_in = i_in + 1;
{%- endif %}

{%- if dims.nh > 0 and simulink_opts.inputs.lh -%}  {#- lh #}
input_note = strcat(input_note, num2str(i_in), ') lh, size [{{ dims.nh }}]\n ');
i_in = i_in + 1;
{%- endif %}
{%- if dims.nh > 0 and simulink_opts.inputs.uh -%}  {#- uh #}
input_note = strcat(input_note, num2str(i_in), ') uh, size [{{ dims.nh }}]\n ');
i_in = i_in + 1;
{%- endif %}

{%- if dims.ny_0 > 0 and simulink_opts.inputs.cost_W_0 %}  {#- cost_W_0 #}
input_note = strcat(input_note, num2str(i_in), ') cost_W_0 in column-major format, size [{{ dims.ny_0 * dims.ny_0 }}]\n ');
i_in = i_in + 1;
{%- endif %}

{%- if dims.ny > 0 and simulink_opts.inputs.cost_W %}  {#- cost_W #}
input_note = strcat(input_note, num2str(i_in), ') cost_W in column-major format, that is set for all intermediate shooting nodes: 1 to N-1, size [{{ dims.ny * dims.ny }}]\n ');
i_in = i_in + 1;
{%- endif %}

{%- if dims.ny_e > 0 and simulink_opts.inputs.cost_W_e %}  {#- cost_W_e #}
input_note = strcat(input_note, num2str(i_in), ') cost_W_e in column-major format, size [{{ dims.ny_e * dims.ny_e }}]\n ');
i_in = i_in + 1;
{%- endif %}

{%- if simulink_opts.inputs.x_init %}  {#- x_init #}
input_note = strcat(input_note, num2str(i_in), ') initialization of x for all shooting nodes, size [{{ dims.nx * (dims.N+1) }}]\n ');
i_in = i_in + 1;
{%- endif %}

{%- if simulink_opts.inputs.u_init %}  {#- u_init #}
input_note = strcat(input_note, num2str(i_in), ') initialization of u for shooting nodes 0 to N-1, size [{{ dims.nu * (dims.N) }}]\n ');
i_in = i_in + 1;
{%- endif %}

fprintf(input_note)

disp(' ')

output_note = 'Outputs are:\n';
i_out = 0;

{%- if dims.nu > 0 and simulink_opts.outputs.u0 == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') u0, control input at node 0, size [{{ dims.nu }}]\n ');
{%- endif %}

{%- if simulink_opts.outputs.utraj == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') utraj, control input concatenated for nodes 0 to N-1, size [{{ dims.nu * dims.N }}]\n ');
{%- endif %}

{%- if simulink_opts.outputs.xtraj == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') xtraj, state concatenated for nodes 0 to N, size [{{ dims.nx * (dims.N + 1) }}]\n ');
{%- endif %}

{%- if simulink_opts.outputs.solver_status == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') acados solver status (0 = SUCCESS)\n ');
{%- endif %}

{%- if simulink_opts.outputs.KKT_residual == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') KKT residual\n ');
{%- endif %}

{%- if dims.N > 0 and simulink_opts.outputs.x1 == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') x1, state at node 1\n ');
{%- endif %}

{%- if simulink_opts.outputs.CPU_time == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') CPU time\n ');
{%- endif %}

{%- if simulink_opts.outputs.CPU_time_sim == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') CPU time integrator\n ');
{%- endif %}

{%- if simulink_opts.outputs.CPU_time_qp == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') CPU time QP solution\n ');
{%- endif %}

{%- if simulink_opts.outputs.CPU_time_lin == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') CPU time linearization (including integrator)\n ');
{%- endif %}

{%- if simulink_opts.outputs.sqp_iter == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') SQP iterations\n ');
{%- endif %}

fprintf(output_note)
