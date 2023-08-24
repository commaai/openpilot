%
% Copyright (c) The acados authors.
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
            {% if model.gnsf.purely_linear != 1 %}
            '{{ model.name }}_model/{{ model.name }}_gnsf_phi_fun.c',...
            '{{ model.name }}_model/{{ model.name }}_gnsf_phi_fun_jac_y.c',...
            '{{ model.name }}_model/{{ model.name }}_gnsf_phi_jac_y_uhat.c',...
            {% if model.gnsf.nontrivial_f_LO == 1 %}
            '{{ model.name }}_model/{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz.c',...
            {%- endif %}
            {%- endif %}
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
{%- elif solver_options.qp_solver is containing("DAQP") %}
CFLAGS = [ CFLAGS, ' -DACADOS_WITH_DAQP' ];
COMPDEFINES = [ COMPDEFINES, ' -DACADOS_WITH_DAQP' ];
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
    {% if solver_options.qp_solver is containing("DAQP") %}
LIBS{end+1} = '-ldaqp';
    {% endif %}
{%- endif %}


try
    %     mex('-v', '-O', CFLAGS, LDFLAGS, COMPFLAGS, COMPDEFINES, INCS{:}, ...
    mex('-O', CFLAGS, LDFLAGS, COMPFLAGS, COMPDEFINES, INCS{:}, ...
            LIB_PATH, LIBS{:}, SOURCES{:}, ...
            '-output', 'acados_solver_sfunction_{{ model.name }}' );
catch exception
    disp('make_sfun failed with the following exception:')
    disp(exception);
    disp('Try adding -v to the mex command above to get more information.')
    keyboard
end

fprintf( [ '\n\nSuccessfully created sfunction:\nacados_solver_sfunction_{{ model.name }}', '.', ...
    eval('mexext')] );


%% print note on usage of s-function, and create I/O port names vectors
fprintf('\n\nNote: Usage of Sfunction is as follows:\n')
input_note = 'Inputs are:\n';
i_in = 1;

global sfun_input_names
sfun_input_names = {};

{%- if dims.nbx_0 > 0 and simulink_opts.inputs.lbx_0 -%}  {#- lbx_0 #}
input_note = strcat(input_note, num2str(i_in), ') lbx_0 - lower bound on x for stage 0,',...
                    ' size [{{ dims.nbx_0 }}]\n ');
sfun_input_names = [sfun_input_names; 'lbx_0 [{{ dims.nbx_0 }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if dims.nbx_0 > 0 and simulink_opts.inputs.ubx_0 -%}  {#- ubx_0 #}
input_note = strcat(input_note, num2str(i_in), ') ubx_0 - upper bound on x for stage 0,',...
                    ' size [{{ dims.nbx_0 }}]\n ');
sfun_input_names = [sfun_input_names; 'ubx_0 [{{ dims.nbx_0 }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if dims.np > 0 and simulink_opts.inputs.parameter_traj -%}  {#- parameter_traj #}
input_note = strcat(input_note, num2str(i_in), ') parameters - concatenated for all shooting nodes 0 to N,',...
                    ' size [{{ (dims.N+1)*dims.np }}]\n ');
sfun_input_names = [sfun_input_names; 'parameter_traj [{{ (dims.N+1)*dims.np }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if dims.ny_0 > 0 and simulink_opts.inputs.y_ref_0 %}
input_note = strcat(input_note, num2str(i_in), ') y_ref_0, size [{{ dims.ny_0 }}]\n ');
sfun_input_names = [sfun_input_names; 'y_ref_0 [{{ dims.ny_0 }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if dims.ny > 0 and dims.N > 1 and simulink_opts.inputs.y_ref %}
input_note = strcat(input_note, num2str(i_in), ') y_ref - concatenated for shooting nodes 1 to N-1,',...
                    ' size [{{ (dims.N-1) * dims.ny }}]\n ');
sfun_input_names = [sfun_input_names; 'y_ref [{{ (dims.N-1) * dims.ny }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if dims.ny_e > 0 and dims.N > 0 and simulink_opts.inputs.y_ref_e %}
input_note = strcat(input_note, num2str(i_in), ') y_ref_e, size [{{ dims.ny_e }}]\n ');
sfun_input_names = [sfun_input_names; 'y_ref_e [{{ dims.ny_e }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if dims.nbx > 0 and dims.N > 1 and simulink_opts.inputs.lbx -%}  {#- lbx #}
input_note = strcat(input_note, num2str(i_in), ') lbx for shooting nodes 1 to N-1, size [{{ (dims.N-1) * dims.nbx }}]\n ');
sfun_input_names = [sfun_input_names; 'lbx [{{ (dims.N-1) * dims.nbx }}]'];
i_in = i_in + 1;
{%- endif %}
{%- if dims.nbx > 0 and dims.N > 1 and simulink_opts.inputs.ubx -%}  {#- ubx #}
input_note = strcat(input_note, num2str(i_in), ') ubx for shooting nodes 1 to N-1, size [{{ (dims.N-1) * dims.nbx }}]\n ');
sfun_input_names = [sfun_input_names; 'ubx [{{ (dims.N-1) * dims.nbx }}]'];
i_in = i_in + 1;
{%- endif %}


{%- if dims.nbx_e > 0 and dims.N > 0 and simulink_opts.inputs.lbx_e -%}  {#- lbx_e #}
input_note = strcat(input_note, num2str(i_in), ') lbx_e (lbx at shooting node N), size [{{ dims.nbx_e }}]\n ');
sfun_input_names = [sfun_input_names; 'lbx_e [{{ dims.nbx_e }}]'];
i_in = i_in + 1;
{%- endif %}
{%- if dims.nbx_e > 0 and dims.N > 0 and simulink_opts.inputs.ubx_e -%}  {#- ubx_e #}
input_note = strcat(input_note, num2str(i_in), ') ubx_e (ubx at shooting node N), size [{{ dims.nbx_e }}]\n ');
sfun_input_names = [sfun_input_names; 'ubx_e [{{ dims.nbx_e }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if dims.nbu > 0 and dims.N > 0 and simulink_opts.inputs.lbu -%}  {#- lbu #}
input_note = strcat(input_note, num2str(i_in), ') lbu for shooting nodes 0 to N-1, size [{{ dims.N*dims.nbu }}]\n ');
sfun_input_names = [sfun_input_names; 'lbu [{{ dims.N*dims.nbu }}]'];
i_in = i_in + 1;
{%- endif -%}
{%- if dims.nbu > 0 and dims.N > 0 and simulink_opts.inputs.ubu -%}  {#- ubu #}
input_note = strcat(input_note, num2str(i_in), ') ubu for shooting nodes 0 to N-1, size [{{ dims.N*dims.nbu }}]\n ');
sfun_input_names = [sfun_input_names; 'ubu [{{ dims.N*dims.nbu }}]'];
i_in = i_in + 1;
{%- endif -%}

{%- if dims.ng > 0 and simulink_opts.inputs.lg -%}  {#- lg #}
input_note = strcat(input_note, num2str(i_in), ') lg for shooting nodes 0 to N-1, size [{{ dims.N*dims.ng }}]\n ');
sfun_input_names = [sfun_input_names; 'lg [{{ dims.N*dims.ng }}]'];
i_in = i_in + 1;
{%- endif %}
{%- if dims.ng > 0 and simulink_opts.inputs.ug -%}  {#- ug #}
input_note = strcat(input_note, num2str(i_in), ') ug for shooting nodes 0 to N-1, size [{{ dims.N*dims.ng }}]\n ');
sfun_input_names = [sfun_input_names; 'ug [{{ dims.N*dims.ng }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if dims.nh > 0 and simulink_opts.inputs.lh -%}  {#- lh #}
input_note = strcat(input_note, num2str(i_in), ') lh for shooting nodes 0 to N-1, size [{{ dims.N*dims.nh }}]\n ');
sfun_input_names = [sfun_input_names; 'lh [{{ dims.N*dims.nh }}]'];
i_in = i_in + 1;
{%- endif %}
{%- if dims.nh > 0 and simulink_opts.inputs.uh -%}  {#- uh #}
input_note = strcat(input_note, num2str(i_in), ') uh for shooting nodes 0 to N-1, size [{{ dims.N*dims.nh }}]\n ');
sfun_input_names = [sfun_input_names; 'uh [{{ dims.N*dims.nh }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if dims.nh_e > 0 and simulink_opts.inputs.lh_e -%}  {#- lh_e #}
input_note = strcat(input_note, num2str(i_in), ') lh_e, size [{{ dims.nh_e }}]\n ');
sfun_input_names = [sfun_input_names; 'lh_e [{{ dims.nh_e }}]'];
i_in = i_in + 1;
{%- endif %}
{%- if dims.nh_e > 0 and simulink_opts.inputs.uh_e -%}  {#- uh_e #}
input_note = strcat(input_note, num2str(i_in), ') uh_e, size [{{ dims.nh_e }}]\n ');
sfun_input_names = [sfun_input_names; 'uh_e [{{ dims.nh_e }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if dims.ny_0 > 0 and simulink_opts.inputs.cost_W_0 %}  {#- cost_W_0 #}
input_note = strcat(input_note, num2str(i_in), ') cost_W_0 in column-major format, size [{{ dims.ny_0 * dims.ny_0 }}]\n ');
sfun_input_names = [sfun_input_names; 'cost_W_0 [{{ dims.ny_0 * dims.ny_0 }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if dims.ny > 0 and simulink_opts.inputs.cost_W %}  {#- cost_W #}
input_note = strcat(input_note, num2str(i_in), ') cost_W in column-major format, that is set for all intermediate shooting nodes: 1 to N-1, size [{{ dims.ny * dims.ny }}]\n ');
sfun_input_names = [sfun_input_names; 'cost_W [{{ dims.ny * dims.ny }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if dims.ny_e > 0 and simulink_opts.inputs.cost_W_e %}  {#- cost_W_e #}
input_note = strcat(input_note, num2str(i_in), ') cost_W_e in column-major format, size [{{ dims.ny_e * dims.ny_e }}]\n ');
sfun_input_names = [sfun_input_names; 'cost_W_e [{{ dims.ny_e * dims.ny_e }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if simulink_opts.inputs.reset_solver %}  {#- reset_solver #}
input_note = strcat(input_note, num2str(i_in), ') reset_solver determines if iterate is set to all zeros before other initializations (x_init, u_init) are set and before solver is called, size [1]\n ');
sfun_input_names = [sfun_input_names; 'reset_solver [1]'];
i_in = i_in + 1;
{%- endif %}

{%- if simulink_opts.inputs.x_init %}  {#- x_init #}
input_note = strcat(input_note, num2str(i_in), ') initialization of x for all shooting nodes, size [{{ dims.nx * (dims.N+1) }}]\n ');
sfun_input_names = [sfun_input_names; 'x_init [{{ dims.nx * (dims.N+1) }}]'];
i_in = i_in + 1;
{%- endif %}

{%- if simulink_opts.inputs.u_init %}  {#- u_init #}
input_note = strcat(input_note, num2str(i_in), ') initialization of u for shooting nodes 0 to N-1, size [{{ dims.nu * (dims.N) }}]\n ');
sfun_input_names = [sfun_input_names; 'u_init [{{ dims.nu * (dims.N) }}]'];
i_in = i_in + 1;
{%- endif %}

fprintf(input_note)

disp(' ')

output_note = 'Outputs are:\n';
i_out = 0;

global sfun_output_names
sfun_output_names = {};

{%- if dims.nu > 0 and simulink_opts.outputs.u0 == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') u0, control input at node 0, size [{{ dims.nu }}]\n ');
sfun_output_names = [sfun_output_names; 'u0 [{{ dims.nu }}]'];
{%- endif %}

{%- if simulink_opts.outputs.utraj == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') utraj, control input concatenated for nodes 0 to N-1, size [{{ dims.nu * dims.N }}]\n ');
sfun_output_names = [sfun_output_names; 'utraj [{{ dims.nu * dims.N }}]'];
{%- endif %}

{%- if simulink_opts.outputs.xtraj == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') xtraj, state concatenated for nodes 0 to N, size [{{ dims.nx * (dims.N + 1) }}]\n ');
sfun_output_names = [sfun_output_names; 'xtraj [{{ dims.nx * (dims.N + 1) }}]'];
{%- endif %}

{%- if simulink_opts.outputs.solver_status == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') acados solver status (0 = SUCCESS)\n ');
sfun_output_names = [sfun_output_names; 'solver_status'];
{%- endif %}

{%- if simulink_opts.outputs.cost_value == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') cost function value\n ');
sfun_output_names = [sfun_output_names; 'cost_value'];
{%- endif %}


{%- if simulink_opts.outputs.KKT_residual == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') KKT residual\n ');
sfun_output_names = [sfun_output_names; 'KKT_residual'];
{%- endif %}

{%- if simulink_opts.outputs.KKT_residuals == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') KKT residuals, size [4] (stat, eq, ineq, comp)\n ');
sfun_output_names = [sfun_output_names; 'KKT_residuals [4]'];
{%- endif %}

{%- if dims.N > 0 and simulink_opts.outputs.x1 == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') x1, state at node 1\n ');
sfun_output_names = [sfun_output_names; 'x1'];
{%- endif %}

{%- if simulink_opts.outputs.CPU_time == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') CPU time\n ');
sfun_output_names = [sfun_output_names; 'CPU_time'];
{%- endif %}

{%- if simulink_opts.outputs.CPU_time_sim == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') CPU time integrator\n ');
sfun_output_names = [sfun_output_names; 'CPU_time_sim'];
{%- endif %}

{%- if simulink_opts.outputs.CPU_time_qp == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') CPU time QP solution\n ');
sfun_output_names = [sfun_output_names; 'CPU_time_qp'];
{%- endif %}

{%- if simulink_opts.outputs.CPU_time_lin == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') CPU time linearization (including integrator)\n ');
sfun_output_names = [sfun_output_names; 'CPU_time_lin'];
{%- endif %}

{%- if simulink_opts.outputs.sqp_iter == 1 %}
i_out = i_out + 1;
output_note = strcat(output_note, num2str(i_out), ') SQP iterations\n ');
sfun_output_names = [sfun_output_names; 'sqp_iter'];
{%- endif %}

fprintf(output_note)

% The mask drawing command is:
% ---
% global sfun_input_names sfun_output_names
% for i = 1:length(sfun_input_names)
% 	port_label('input', i, sfun_input_names{i})
% end
% for i = 1:length(sfun_output_names)
% 	port_label('output', i, sfun_output_names{i})
% end
% ---
% It can be used by copying it in sfunction/Mask/Edit mask/Icon drawing commands
%   (you can access it wirth ctrl+M on the s-function)